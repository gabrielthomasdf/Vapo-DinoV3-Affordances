"""Audit recovered checkpoint hashes, split manifests and the reported command.

Only reads saved artifacts; never imports models or executes the evaluator.
Exports sample-list fingerprints and counts, not raw sample IDs or private paths.
"""

import argparse
import ast
import hashlib
import json
import re
import shlex
import tarfile
from pathlib import Path

from analyze_recovered_results import ValidationError, integer, read_csv, require, sha256


CHECKPOINTS = {
    "resnet18_gripper": "trained_models/affordance/gripper_tabletop.ckpt",
    "static_affordance": "trained_models/affordance/static_tabletop.ckpt",
    "dinov3_gripper": "reproduction/dinov3_affordance/gripper_seed0/trained_models/best_val_miou.ckpt",
    "sac_vapo": "trained_models/policy/tabletop/vapo/trained_models/most_tasks_from_15.pth",
    "sac_no_affordance_ablation": "trained_models/policy/tabletop/baseline/trained_models/most_tasks_from_15.pth",
    "dinov3_backbone": "dinov3_weights/dinov3_vits16plus_pretrain_lvd1689m.pth",
}


def checkpoint_manifest(path):
    records = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([a-fA-F0-9]{64})\s+\*?(.+)", line)
        require(match is not None, "Invalid SHA-256 manifest line")
        digest, private_path = match.groups()
        normalized = private_path.replace("\\", "/")
        roles = [role for role, suffix in CHECKPOINTS.items()
                 if normalized == suffix or normalized.endswith("/" + suffix)]
        require(len(roles) == 1, "Unknown or ambiguous checkpoint role")
        role = roles[0]
        require(role not in records, "Duplicate checkpoint role")
        records[role] = {"sha256": digest.lower(), "public_reference": CHECKPOINTS[role],
                         "selected_setup_reference": role != "sac_no_affordance_ablation"}
    require(set(records) == set(CHECKPOINTS), "Incomplete checkpoint identity manifest")
    return {"source_sha256": sha256(path), "artifacts": records,
            "verification": "SHA-256 identities recorded in the recovered manifest; checkpoint binaries were not rehashed locally.",
            "runtime_identity_limit": "Logical references match recovered code/configs; hashes do not independently attest what was loaded at historical runtime."}


def ordered_fingerprint(samples):
    return hashlib.sha256(("\n".join(samples) + "\n").encode("utf-8")).hexdigest()


def split_samples(manifest, split, prefix=""):
    require(isinstance(manifest, dict) and isinstance(manifest.get(split), dict), "Invalid split schema")
    result = []
    # Match VREnvData._get_split_data: preserve episode insertion order,
    # sort files within each episode, and apply its substring camera filter.
    for episode, files in manifest[split].items():
        require(isinstance(episode, str) and isinstance(files, list) and all(isinstance(f, str) for f in files),
                "Invalid split episode/file list")
        require(".." not in episode.split("/"), "Unexpected episode identifier")
        result.extend(prefix + episode + "/" + f for f in sorted(files) if "gripper" in f)
    require(len(result) == len(set(result)), "Duplicate split sample ID")
    return result


def audit_splits(archive_path, image_csv):
    rows = read_csv(image_csv, ("index", "sample"))
    require(sorted(integer(r["index"]) for r in rows) == list(range(len(rows))), "Invalid image index sequence")
    observed = [r["sample"] for r in sorted(rows, key=lambda r: integer(r["index"]))]
    require(len(observed) == len(set(observed)), "Duplicate evaluated sample ID")
    manifests, hashes = {}, {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            require(bool(re.fullmatch(r"(?:[A-Za-z0-9_-]+/)?episodes_split\.json", member.name)),
                    "Unexpected split archive member")
            require(member.name not in manifests and member.size < 64 * 1024 * 1024, "Invalid/duplicate split member")
            raw = archive.extractfile(member).read()
            manifests[member.name] = json.loads(raw)
            hashes[member.name] = hashlib.sha256(raw).hexdigest()
    require("episodes_split.json" in manifests and len(manifests) == 4, "Expected merged and three scene manifests")
    root = manifests["episodes_split.json"]
    training, validation = (split_samples(root, split) for split in ("training", "validation"))
    require(not set(training).intersection(validation), "Training/validation sample overlap")
    require(validation == observed, "Merged validation IDs/order differ from the evaluated CSV")
    scenes, joined = [], {"training": [], "validation": []}
    for name in sorted(n for n in manifests if n != "episodes_split.json"):
        scene = name.split("/")[0]
        prefix = "/" + scene + "/"
        parts = {split: split_samples(manifests[name], split, prefix) for split in joined}
        for split, root_samples in (("training", training), ("validation", validation)):
            require(parts[split] == [x for x in root_samples if x.startswith(prefix)],
                    "Scene manifest disagrees with merged ordered subset")
            joined[split].extend(parts[split])
        scenes.append({"scene": scene, "manifest_sha256": hashes[name],
                       "training_gripper_samples": len(parts["training"]),
                       "validation_gripper_samples": len(parts["validation"]),
                       "ordered_validation_ids_sha256": ordered_fingerprint(parts["validation"]),
                       "matches_merged_and_evaluated_scene_subset_in_order": True})
    require(len(joined["training"]) == len(training) and set(joined["training"]) == set(training), "Scene training union mismatch")
    require(len(joined["validation"]) == len(validation) and set(joined["validation"]) == set(validation), "Scene validation union mismatch")
    return {"archive_sha256": sha256(archive_path), "member_sha256": hashes, "camera": "gripper",
            "training_gripper_samples": len(training), "validation_gripper_samples": len(validation),
            "training_validation_overlap": 0, "matches_evaluated_ids_in_order": True,
            "ordered_validation_ids_sha256": ordered_fingerprint(validation), "scenes": scenes,
            "fingerprint_encoding": "UTF-8 of sample IDs joined by newline, followed by one final newline",
            "conclusion": "The merged manifest and scene subsets reproduce all evaluated sample IDs in order using the recovered loader semantics.",
            "limits": "Sample membership/order verified; image/label file contents, dataset release and historical on-disk manifest bytes are not independently attested.",
            "raw_sample_ids_exported": False}


def static_defaults(source):
    defaults = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            name = node.args[0].value
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[name] = kw.value.value
                elif kw.arg == "default" and isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name):
                    if kw.value.func.id == "Path" and len(kw.value.args) == 1 and isinstance(kw.value.args[0], ast.Constant):
                        defaults[name] = kw.value.args[0].value
                elif kw.arg == "action" and isinstance(kw.value, ast.Constant) and kw.value.value == "store_true":
                    defaults[name] = False
    return defaults


def audit_command(command_path, evaluator_path, recovered_summary):
    recorded = command_path.read_text(encoding="utf-8-sig").strip()
    tokens = shlex.split(recorded)
    require(len(tokens) >= 2 and tokens[0] in ("python", "python3") and
            Path(tokens[1]).name == "eval_affordance_dinov3_vs_baseline.py", "Unexpected recorded command")
    defaults = static_defaults(evaluator_path.read_text(encoding="utf-8-sig"))
    require("--warmup" in defaults and "--amp" in defaults, "Evaluator lacks required options")
    warmup, amp, seen, index = defaults["--warmup"], defaults["--amp"], set(), 2
    while index < len(tokens):
        flag = tokens[index]
        require(flag not in seen, "Duplicate command option")
        seen.add(flag)
        if flag == "--amp":
            amp = True
        elif flag == "--warmup":
            require(index + 1 < len(tokens), "Warmup value missing")
            index += 1
            warmup = integer(tokens[index])
        else:
            raise ValidationError("Unexpected option in recovered command")
        index += 1
    require(amp == recovered_summary["comparison"]["amp"], "Command AMP differs from recovered evaluation summary")
    return {"command_file_sha256": sha256(command_path), "evaluator_sha256": sha256(evaluator_path),
            "reported_historical_command": "python eval_affordance_dinov3_vs_baseline.py --warmup " + str(warmup) + (" --amp" if amp else ""),
            "warmup_iterations_recorded": warmup, "warmup_default_in_source": defaults["--warmup"],
            "amp_recorded": amp, "amp_matches_recovered_summary": True, "arguments_supported_by_source": True,
            "checkpoint_config_defaults": {name: defaults.get(name) for name in ("--dino-config", "--dino-checkpoint", "--baseline-checkpoint")},
            "evidence_level": "Command supplied in a recovered text artifact; consistent with code and AMP summary, without a contemporaneous execution trace.",
            "actual_runtime_warmup_independently_attested": False}


def audit_identity(audit_dir, metrics_dir, evaluator_path):
    audit_dir, metrics_dir, evaluator_path = Path(audit_dir), Path(metrics_dir), Path(evaluator_path)
    summary = json.loads((metrics_dir / "resumo.json").read_text(encoding="utf-8-sig"))
    return {"schema_version": 1, "audit_script_sha256": sha256(Path(__file__)),
            "metric_sources_sha256": {name: sha256(metrics_dir / name) for name in ("metricas_por_imagem.csv", "resumo.json", "ab_results.csv")},
            "checkpoint_identities": checkpoint_manifest(audit_dir / "vapo_checkpoint_hashes_20261104.txt"),
            "dataset_split_audit": audit_splits(audit_dir / "vapo_dataset_splits_20261104.tar.gz", metrics_dir / "metricas_por_imagem.csv"),
            "historical_command_audit": audit_command(audit_dir / "vapo_comando_percepcao_20261104.txt", evaluator_path, summary),
            "execution_scope": "Saved artifact inspection only; no training, model inference or robot evaluation was performed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=Path(__file__).with_name("eval_affordance_dinov3_vs_baseline.py"))
    args = parser.parse_args()
    try:
        report = audit_identity(args.audit_dir, args.metrics_dir, args.evaluator)
    except (ValidationError, OSError, ValueError, KeyError, tarfile.TarError):
        parser.exit(2, "Identity audit failed. Check the recovered artifact schemas and consistency.\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "identity_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    identities = report["checkpoint_identities"]["artifacts"]
    (args.output_dir / "checkpoints.sha256").write_text(
        "".join(identities[role]["sha256"] + "  " + CHECKPOINTS[role] + "\n" for role in CHECKPOINTS), encoding="utf-8")
    print(f"Verified ordered validation partition: {report['dataset_split_audit']['validation_gripper_samples']} images.")
    print(f"Checkpoint identities recorded: {len(identities)}; binaries not rehashed locally.")


if __name__ == "__main__":
    main()
