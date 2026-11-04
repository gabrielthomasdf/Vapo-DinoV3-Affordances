#!/usr/bin/env python3
"""Compare DINOv3 with the original VAPO UNet at their native resolutions.

Both models are evaluated on the same validation samples, but each receives
the preprocessing used by its own training run. The official gripper VAPO
checkpoint uses 64x64 inputs and radius 10; the DINOv3 values come from its
saved Hydra configuration. The script reports dataset-global binary
segmentation metrics, VAPO's foreground IoU reduction, and forward latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vapo.affordance.affordance_model import AffordanceModel
from vapo.affordance.dataloader.datasets import VREnvData


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.bool()
        target = target.bool()
        self.tp += int((pred & target).sum().item())
        self.fp += int((pred & ~target).sum().item())
        self.fn += int((~pred & target).sum().item())
        self.tn += int((~pred & ~target).sum().item())


class OriginalVapoAffordance(nn.Module):
    """Original ResNet18 UNet reconstructed from the baseline checkpoint."""

    def __init__(
        self,
        input_channels: int,
        n_classes: int,
        feature_dim: int,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=input_channels,
            classes=n_classes,
            encoder_depth=len(decoder_channels),
            decoder_channels=decoder_channels,
            activation=None,
        )
        self.center_direction_net = nn.Conv2d(
            feature_dim,
            2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def _prepare_input(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.input_channels == rgb.shape[1]:
            return rgb
        if self.input_channels == 1 and rgb.shape[1] == 3:
            return tvf.rgb_to_grayscale(rgb, num_output_channels=1)
        raise ValueError(
            "Não foi possível adaptar a entrada RGB com "
            f"{rgb.shape[1]} canais para o baseline com "
            f"{self.input_channels} canais."
        )

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare_input(rgb)
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        logits = self.unet.segmentation_head(decoder_output)
        directions = self.center_direction_net(decoder_output)
        return logits, directions


class DinoAffordanceForEvaluation(nn.Module):
    """Thin wrapper that avoids Hough voting during segmentation evaluation."""

    def __init__(self, model: AffordanceModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.unet.encoder(x)
        decoder_output = self.model.unet.decoder(*features)
        logits = self.model.unet.segmentation_head(decoder_output)
        directions = self.model.center_direction_net(decoder_output)
        return logits, directions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dino-config",
        type=Path,
        default=Path(
            "reproduction/dinov3_affordance/gripper_seed0/.hydra/config.yaml"
        ),
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        default=Path(
            "reproduction/dinov3_affordance/gripper_seed0/"
            "trained_models/best_val_miou.ckpt"
        ),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("trained_models/affordance/gripper_tabletop.ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction/eval_dinov3_vs_vapo_native"),
    )
    parser.add_argument(
        "--baseline-img-size",
        type=int,
        default=64,
        help="Resolução nativa do checkpoint VAPO gripper_tabletop.",
    )
    parser.add_argument(
        "--baseline-radius",
        type=int,
        default=10,
        help="Raio nativo dos rótulos gripper do VAPO oficial.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use somente para smoke test; omita na avaliação final.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use autocast FP16 nos dois modelos.",
    )
    return parser.parse_args()


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def require_state_tensor(
    state: dict[str, torch.Tensor],
    candidates: tuple[str, ...],
) -> torch.Tensor:
    for key in candidates:
        if key in state:
            return state[key]
    raise KeyError(
        "Nenhuma das chaves esperadas foi encontrada no checkpoint: "
        + ", ".join(candidates)
    )


def infer_decoder_channels(
    state: dict[str, torch.Tensor],
) -> tuple[int, ...]:
    """Infer the original UNet decoder layout from checkpoint tensor shapes."""
    pattern = re.compile(
        r"^unet\.decoder\.blocks\.(\d+)\.conv1\.0\.weight$"
    )
    channels_by_block: dict[int, int] = {}

    for key, tensor in state.items():
        match = pattern.match(key)
        if match is not None:
            channels_by_block[int(match.group(1))] = int(tensor.shape[0])

    if not channels_by_block:
        raise KeyError(
            "Não foi possível inferir decoder_channels do checkpoint "
            "baseline: nenhuma chave no formato "
            "'unet.decoder.blocks.N.conv1.0.weight' foi encontrada."
        )

    expected_indices = list(range(max(channels_by_block) + 1))
    actual_indices = sorted(channels_by_block)
    if actual_indices != expected_indices:
        raise RuntimeError(
            "Os blocos do decoder no checkpoint baseline não são "
            f"contíguos: encontrados {actual_indices}."
        )

    return tuple(channels_by_block[index] for index in expected_indices)


def load_original_baseline(path: Path) -> OriginalVapoAffordance:
    checkpoint = load_torch_checkpoint(path)
    state = checkpoint.get("state_dict", checkpoint)

    first_conv = require_state_tensor(
        state,
        (
            "unet.encoder.conv1.weight",
            "unet.encoder.model.conv1.weight",
        ),
    )
    segmentation_head = require_state_tensor(
        state,
        (
            "unet.segmentation_head.0.weight",
            "unet.segmentation_head.weight",
        ),
    )
    center_head = require_state_tensor(
        state,
        ("center_direction_net.weight",),
    )
    decoder_channels = infer_decoder_channels(state)

    model = OriginalVapoAffordance(
        input_channels=int(first_conv.shape[1]),
        n_classes=int(segmentation_head.shape[0]),
        feature_dim=int(center_head.shape[1]),
        decoder_channels=decoder_channels,
    )

    expected = model.state_dict()
    relevant_state = {key: value for key, value in state.items() if key in expected}
    missing, unexpected = model.load_state_dict(relevant_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "O checkpoint baseline não corresponde à UNet original.\n"
            f"Chaves ausentes: {missing}\n"
            f"Chaves inesperadas: {unexpected}"
        )
    print(
        "Baseline reconstruído com "
        f"encoder_depth={len(decoder_channels)} e "
        f"decoder_channels={decoder_channels}"
    )
    return model


def load_dino(path: Path, cfg: Any, input_channels: int) -> nn.Module:
    checkpoint = load_torch_checkpoint(path)
    state = checkpoint.get("state_dict", checkpoint)
    model = AffordanceModel(
        cfg.model_cfg,
        n_classes=cfg.model_cfg.n_classes,
        input_channels=input_channels,
        cmd_log=None,
    )
    model.load_state_dict(state, strict=True)
    return DinoAffordanceForEvaluation(model)


def build_original_baseline_dataset_cfg(
    dino_dataset_cfg: Any,
    camera: str,
    image_size: int,
    radius: int,
    input_channels: int,
) -> Any:
    """Build the validation preprocessing from the official VAPO config."""
    baseline_cfg = OmegaConf.create(
        OmegaConf.to_container(dino_dataset_cfg, resolve=True)
    )
    baseline_cfg.img_resize[camera] = image_size
    baseline_cfg.radius[camera] = radius

    validation_transforms: list[dict[str, Any]] = [
        {
            "_target_": "torchvision.transforms.Resize",
            "size": image_size,
        }
    ]
    if input_channels == 1:
        validation_transforms.append(
            {
                "_target_": "torchvision.transforms.Grayscale",
                "num_output_channels": 1,
            }
        )
    elif input_channels != 3:
        raise ValueError(
            "O checkpoint baseline deveria receber 1 ou 3 canais, "
            f"mas recebe {input_channels}."
        )
    validation_transforms.extend(
        [
            {
                "_target_": (
                    "vapo.affordance.dataloader.transforms.ScaleImageTensor"
                )
            },
            {
                "_target_": "torchvision.transforms.Normalize",
                "mean": [0.5],
                "std": [0.5],
            },
        ]
    )
    baseline_cfg.transforms_cfg.validation = OmegaConf.create(
        validation_transforms
    )
    baseline_cfg.transforms_cfg.masks = OmegaConf.create(
        [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": image_size,
            }
        ]
    )
    return baseline_cfg


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def metrics_from_counts(counts: Counts) -> dict[str, float | int]:
    iou_aff = safe_div(counts.tp, counts.tp + counts.fp + counts.fn)
    iou_bg = safe_div(counts.tn, counts.tn + counts.fp + counts.fn)
    return {
        **asdict(counts),
        "iou_affordance_global": iou_aff,
        "iou_background_global": iou_bg,
        "miou_binary_global": (iou_aff + iou_bg) / 2,
        "dice_f1_global": safe_div(
            2 * counts.tp,
            2 * counts.tp + counts.fp + counts.fn,
        ),
        "precision_global": safe_div(counts.tp, counts.tp + counts.fp),
        "recall_global": safe_div(counts.tp, counts.tp + counts.fn),
        "accuracy_global": safe_div(
            counts.tp + counts.tn,
            counts.tp + counts.fp + counts.fn + counts.tn,
        ),
    }


def per_image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    counts = Counts()
    counts.add(pred, target)
    metrics = metrics_from_counts(counts)
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "tn": counts.tn,
        "iou_affordance": metrics["iou_affordance_global"],
        "dice_f1": metrics["dice_f1_global"],
        "precision": metrics["precision_global"],
        "recall": metrics["recall_global"],
    }


def mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_and_time(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, float]:
    synchronize(device)
    start = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp,
    ):
        logits, _ = model(image)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    pred = logits.argmax(dim=1)
    return pred, elapsed_ms


def warm_up(
    model_inputs: tuple[tuple[nn.Module, torch.Tensor], ...],
    device: torch.device,
    amp: bool,
    iterations: int,
) -> None:
    if iterations <= 0:
        return
    with torch.inference_mode():
        for _ in range(iterations):
            for model, image in model_inputs:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp,
                ):
                    model(image)
    synchronize(device)


def model_summary(
    counts: Counts,
    per_image_ious: list[float],
    latencies_ms: list[float],
) -> dict[str, float | int]:
    result = metrics_from_counts(counts)
    result["vapo_iou_mean_per_image"] = mean_finite(per_image_ious)
    result["latency_ms_mean"] = mean_finite(latencies_ms)
    result["latency_ms_median"] = float(
        torch.tensor(latencies_ms).median().item()
    )
    result["fps_from_mean_latency"] = safe_div(
        1000.0,
        float(result["latency_ms_mean"]),
    )
    return result


def vapo_validation_iou(
    rows: list[dict[str, Any]],
    model_name: str,
    batch_size: int = 2,
) -> float:
    """Reproduce VAPO's validation IoU reduction with training batch size 2."""
    weighted_sum = 0.0
    sample_count = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        tp = sum(int(row[f"{model_name}_tp"]) for row in batch)
        fp = sum(int(row[f"{model_name}_fp"]) for row in batch)
        fn = sum(int(row[f"{model_name}_fn"]) for row in batch)
        batch_iou = safe_div(tp, tp + fp + fn)
        if math.isfinite(batch_iou):
            weighted_sum += batch_iou * len(batch)
            sample_count += len(batch)
    return safe_div(weighted_sum, sample_count)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Esta avaliação requer uma GPU CUDA.")

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    cfg = OmegaConf.load(args.dino_config)

    # During training, train_affordance.py selects the resize associated with
    # the active camera before Hydra resolves ${img_resize}. The persisted
    # .hydra/config.yaml still contains only dataset.img_resize as a mapping,
    # so evaluation must reproduce that runtime selection.
    camera = str(cfg.dataset.cam)
    if camera not in cfg.dataset.img_resize:
        available = ", ".join(str(key) for key in cfg.dataset.img_resize.keys())
        raise KeyError(
            f"Não há img_resize para a câmera '{camera}'. "
            f"Opções disponíveis: {available}"
        )
    cfg.img_resize = cfg.dataset.img_resize[camera]
    OmegaConf.resolve(cfg)

    dino_dataset = VREnvData(
        split="validation",
        log=None,
        **cfg.dataset,
    )

    print("Carregando baseline original...")
    baseline = load_original_baseline(args.baseline_checkpoint).to(device).eval()
    baseline_dataset_cfg = build_original_baseline_dataset_cfg(
        cfg.dataset,
        camera=camera,
        image_size=args.baseline_img_size,
        radius=args.baseline_radius,
        input_channels=baseline.input_channels,
    )
    baseline_dataset = VREnvData(
        split="validation",
        log=None,
        **baseline_dataset_cfg,
    )

    if baseline_dataset.data != dino_dataset.data:
        raise RuntimeError(
            "Os datasets baseline e DINOv3 não possuem as mesmas amostras "
            "na mesma ordem. A comparação foi interrompida."
        )

    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dino_loader = DataLoader(
        dino_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    baseline_sample_image, _ = baseline_dataset[0]
    dino_sample_image, _ = dino_dataset[0]
    dino_input_channels = int(dino_sample_image.shape[0])
    if int(baseline_sample_image.shape[0]) != baseline.input_channels:
        raise RuntimeError(
            "O pipeline oficial do baseline produziu um número incorreto "
            "de canais."
        )
    if dino_input_channels != 3:
        raise RuntimeError(
            "A configuração DINO deveria produzir RGB (3 canais), "
            f"mas produziu {dino_input_channels}."
        )

    print("Carregando DINOv3...")
    dino = load_dino(
        args.dino_checkpoint,
        cfg,
        dino_input_channels,
    ).to(device).eval()

    baseline_first_image = baseline_sample_image.unsqueeze(0).to(device)
    dino_first_image = dino_sample_image.unsqueeze(0).to(device)
    print(
        "Pipeline nativo VAPO: "
        f"entrada={tuple(baseline_sample_image.shape)}, "
        f"radius={args.baseline_radius}"
    )
    print(
        "Pipeline nativo DINOv3: "
        f"entrada={tuple(dino_sample_image.shape)}, "
        f"radius={cfg.dataset.radius[camera]}"
    )
    warm_up(
        (
            (baseline, baseline_first_image),
            (dino, dino_first_image),
        ),
        device,
        args.amp,
        args.warmup,
    )

    counts = {"baseline": Counts(), "dinov3": Counts()}
    image_ious: dict[str, list[float]] = {"baseline": [], "dinov3": []}
    latencies: dict[str, list[float]] = {"baseline": [], "dinov3": []}
    rows: list[dict[str, Any]] = []

    total = len(dino_dataset)
    if args.max_samples is not None:
        total = min(total, args.max_samples)

    with torch.inference_mode():
        paired_loader = zip(baseline_loader, dino_loader)
        for index, (baseline_batch, dino_batch) in enumerate(paired_loader):
            if index >= total:
                break

            baseline_image, baseline_labels = baseline_batch
            dino_image, dino_labels = dino_batch
            baseline_image = baseline_image.to(device, non_blocking=True)
            dino_image = dino_image.to(device, non_blocking=True)
            baseline_target = baseline_labels["affordance"].to(
                device,
                non_blocking=True,
            )
            dino_target = dino_labels["affordance"].to(
                device,
                non_blocking=True,
            )

            row: dict[str, Any] = {
                "index": index,
                "sample": dino_dataset.data[index],
                "baseline_positive_pixels_gt": int(
                    (baseline_target == 1).sum().item()
                ),
                "dinov3_positive_pixels_gt": int(
                    (dino_target == 1).sum().item()
                ),
            }

            model_batches = (
                ("baseline", baseline, baseline_image, baseline_target),
                ("dinov3", dino, dino_image, dino_target),
            )
            for name, model, image, target in model_batches:
                pred, latency_ms = predict_and_time(
                    model,
                    image,
                    device,
                    args.amp,
                )
                counts[name].add(pred, target)
                metrics = per_image_metrics(pred, target)
                image_ious[name].append(float(metrics["iou_affordance"]))
                latencies[name].append(latency_ms)

                for metric_name, value in metrics.items():
                    row[f"{name}_{metric_name}"] = value
                row[f"{name}_latency_ms"] = latency_ms

            row["delta_iou_dinov3_minus_baseline"] = (
                row["dinov3_iou_affordance"]
                - row["baseline_iou_affordance"]
            )
            rows.append(row)

            if (index + 1) % 1000 == 0 or index + 1 == total:
                print(f"Avaliadas {index + 1}/{total} imagens")

    summaries = {
        name: model_summary(counts[name], image_ious[name], latencies[name])
        for name in ("baseline", "dinov3")
    }
    for name in ("baseline", "dinov3"):
        summaries[name]["vapo_val_iou_batch2"] = vapo_validation_iou(
            rows,
            name,
            batch_size=int(cfg.dataloader.batch_size),
        )
    summaries["comparison"] = {
        "samples": len(rows),
        "amp": args.amp,
        "device": torch.cuda.get_device_name(device),
        "evaluation_mode": "native_preprocessing",
        "baseline_input_size": args.baseline_img_size,
        "baseline_radius": args.baseline_radius,
        "dinov3_input_size": int(dino_sample_image.shape[-1]),
        "dinov3_radius": int(cfg.dataset.radius[camera]),
        "delta_iou_affordance_global": (
            summaries["dinov3"]["iou_affordance_global"]
            - summaries["baseline"]["iou_affordance_global"]
        ),
        "delta_miou_binary_global": (
            summaries["dinov3"]["miou_binary_global"]
            - summaries["baseline"]["miou_binary_global"]
        ),
        "delta_vapo_iou_mean_per_image": (
            summaries["dinov3"]["vapo_iou_mean_per_image"]
            - summaries["baseline"]["vapo_iou_mean_per_image"]
        ),
        "delta_vapo_val_iou_batch2": (
            summaries["dinov3"]["vapo_val_iou_batch2"]
            - summaries["baseline"]["vapo_val_iou_batch2"]
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "metricas_por_imagem.csv"
    json_path = args.output_dir / "resumo.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(summaries, json_file, indent=2, ensure_ascii=False)

    print("\n=== RESULTADO ===")
    for name in ("baseline", "dinov3"):
        result = summaries[name]
        print(
            f"{name:8s} | "
            f"IoU affordance global nativo="
            f"{result['iou_affordance_global']:.6f} | "
            f"IoU VAPO batch=2={result['vapo_val_iou_batch2']:.6f} | "
            f"mIoU binário global={result['miou_binary_global']:.6f} | "
            f"Dice={result['dice_f1_global']:.6f} | "
            f"Precision={result['precision_global']:.6f} | "
            f"Recall={result['recall_global']:.6f} | "
            f"latência={result['latency_ms_mean']:.3f} ms"
        )
    print(
        "Delta IoU global DINOv3 - baseline: "
        f"{summaries['comparison']['delta_iou_affordance_global']:+.6f}"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
