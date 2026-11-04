"""Audit recovered VAPO CSVs without importing models or publishing raw rows.

Core analysis uses only the Python standard library. Plotting requires Matplotlib.
See docs/experimental-reproduction.md for formulas, assumptions, and CLI examples.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import statistics
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


MODELS = ("baseline", "dinov3")
LABELS = {"baseline": "Baseline", "dinov3": "DINOv3"}
COUNTS = ("tp", "fp", "fn", "tn")
PER_IMAGE = ("iou_affordance", "dice_f1", "precision", "recall")
SOURCE_NAMES = ("metricas_por_imagem.csv", "resumo.json", "ab_results.csv", "git_commit.txt")


class ValidationError(ValueError):
    """An input violates the documented recovered-artifact schema."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        require(len(header) == len(set(header)), "Duplicate CSV column names")
        require(set(required) <= set(header), "Missing required CSV columns")
        rows = list(reader)
    require(bool(rows), "CSV is empty")
    require(all(None not in r and all(v not in (None, "") for v in r.values()) for r in rows),
            "CSV has missing fields or malformed rows")
    return rows


def integer(value):
    require(not isinstance(value, bool), "Expected an integer, not a boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Expected an integer") from exc
    require(result >= 0 and (not isinstance(value, float) or value == result), "Expected a nonnegative integer")
    return result


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Expected a number") from exc
    require(math.isfinite(result), "Unexpected nonfinite number")
    return result


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def scores(counts):
    tp, fp, fn, tn = (counts[k] for k in COUNTS)
    fg = ratio(tp, tp + fp + fn)
    bg = ratio(tn, tn + fp + fn)
    return {
        "iou_affordance_global": fg,
        "iou_background_global": bg,
        "miou_binary_global": (fg + bg) / 2 if fg is not None and bg is not None else None,
        "dice_f1_global": ratio(2 * tp, 2 * tp + fp + fn),
        "precision_global": ratio(tp, tp + fp),
        "recall_global": ratio(tp, tp + fn),
        "accuracy_global": ratio(tp + tn, tp + fp + fn + tn),
    }


def check_derived(value, expected):
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Invalid derived metric") from exc
    if expected is None:
        require(math.isnan(observed), "Zero-denominator metric must be NaN")
    else:
        require(math.isfinite(observed) and math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10),
                "Derived metric disagrees with confusion counts")


def analyze_segmentation(path, batch_size=2):
    require(batch_size > 0, "Batch size must be positive")
    columns = ["index", "sample", "delta_iou_dinov3_minus_baseline"]
    for model in MODELS:
        columns += [model + "_" + key for key in (*COUNTS, *PER_IMAGE, "positive_pixels_gt", "latency_ms")]
    rows = read_csv(path, columns)
    require(len({r["sample"] for r in rows}) == len(rows), "Duplicate segmentation sample")
    indices = [integer(r["index"]) for r in rows]
    require(sorted(indices) == list(range(len(rows))), "Missing or duplicate segmentation index")
    rows.sort(key=lambda r: integer(r["index"]))
    result = {"samples": len(rows), "models": {}, "batch_size_for_iou_reconstruction": batch_size}
    per_model_counts = {}
    for model in MODELS:
        totals = dict.fromkeys(COUNTS, 0)
        nan_counts = Counter(dict.fromkeys(PER_IMAGE, 0))
        per_image_iou, latencies, confusion_rows, pixel_totals = [], [], [], set()
        for row in rows:
            counts = {k: integer(row[model + "_" + k]) for k in COUNTS}
            pixels = sum(counts.values())
            require(pixels > 0, "Empty confusion matrix")
            pixel_totals.add(pixels)
            tp, fp, fn, _ = (counts[k] for k in COUNTS)
            require(integer(row[model + "_positive_pixels_gt"]) == tp + fn,
                    "Ground-truth positive count disagrees with TP + FN")
            expected = dict(zip(PER_IMAGE, (ratio(tp, tp + fp + fn), ratio(2 * tp, 2 * tp + fp + fn),
                                           ratio(tp, tp + fp), ratio(tp, tp + fn))))
            for metric, value in expected.items():
                check_derived(row[model + "_" + metric], value)
                nan_counts[metric] += value is None
            if expected["iou_affordance"] is not None:
                per_image_iou.append(expected["iou_affordance"])
            latency = finite(row[model + "_latency_ms"])
            require(latency > 0, "Latency must be positive")
            latencies.append(latency)
            confusion_rows.append(counts)
            for key in COUNTS:
                totals[key] += counts[key]
        batch_ious = []
        for start in range(0, len(rows), batch_size):
            group = confusion_rows[start:start + batch_size]
            total = {k: sum(r[k] for r in group) for k in COUNTS}
            batch_ious.append(scores(total)["iou_affordance_global"])
        defined_batches = [x for x in batch_ious if x is not None]
        mean_latency = statistics.mean(latencies)
        lower_middle = sorted(latencies)[(len(latencies) - 1) // 2]
        result["models"][model] = {
            **totals, **scores(totals),
            "pixels_per_image": sorted(pixel_totals),
            "undefined_per_image_metrics": dict(nan_counts),
            "vapo_iou_mean_per_image": statistics.mean(per_image_iou) if per_image_iou else None,
            "vapo_val_iou_batch2": statistics.mean(defined_batches) if defined_batches else None,
            "training_foreground_iou_batch_mean_empty_zero": statistics.mean(x or 0.0 for x in batch_ious),
            "empty_union_batches": len(batch_ious) - len(defined_batches),
            "batch_count": len(batch_ious),
            "latency_ms_mean": mean_latency,
            "latency_ms_median": statistics.median(latencies),
            "latency_ms_lower_middle_float32": struct.unpack("f", struct.pack("f", lower_middle))[0],
            "latency_ms_min": min(latencies), "latency_ms_max": max(latencies),
            "fps_from_mean_latency": 1000 / mean_latency,
        }
        per_model_counts[model] = confusion_rows
    undefined_deltas = 0
    for row, base, dino in zip(rows, per_model_counts["baseline"], per_model_counts["dinov3"]):
        biou, diou = (scores(c)["iou_affordance_global"] for c in (base, dino))
        delta = diou - biou if biou is not None and diou is not None else None
        check_derived(row["delta_iou_dinov3_minus_baseline"], delta)
        undefined_deltas += delta is None
    result["undefined_per_image_deltas"] = undefined_deltas
    result["integrity"] = {"duplicate_samples": 0, "missing_indices": 0,
                           "missing_fields": 0, "derived_metrics_consistent": True}
    return result


def aggregate_episodes(rows):
    successes = sum(r["success"] for r in rows)
    return {"episodes": len(rows), "successes": successes, "success_rate": successes / len(rows),
            "mean_return": statistics.mean(r["return"] for r in rows),
            "mean_steps": statistics.mean(r["steps"] for r in rows)}


def analyze_robotics(path):
    rows = read_csv(path, ("model", "seed", "episode_index", "object", "success", "return", "steps"))
    require({r["model"] for r in rows} == set(MODELS), "Expected baseline and dinov3 model labels")
    keys = {m: set() for m in MODELS}
    by_object, by_seed = defaultdict(list), defaultdict(list)
    for row in rows:
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", row["object"])), "Unsafe object identifier")
        for name in ("seed", "episode_index", "success", "steps"):
            row[name] = integer(row[name])
        require(row["success"] in (0, 1), "Success must be 0 or 1")
        require(row["steps"] > 0, "Steps must be positive")
        row["return"] = finite(row["return"])
        key = (row["seed"], row["object"], row["episode_index"])
        require(key not in keys[row["model"]], "Duplicate robotic episode key")
        keys[row["model"]].add(key)
        by_object[(row["model"], row["object"])].append(row)
        by_seed[(row["model"], row["seed"])].append(row)
    require(keys["baseline"] == keys["dinov3"], "Unpaired robotic episode keys across models")
    seeds = sorted({r["seed"] for r in rows})
    objects = sorted({r["object"] for r in rows})
    object_index_sets = [{(r["object"], r["episode_index"]) for r in by_seed[("baseline", s)]} for s in seeds]
    balanced = all(x == object_index_sets[0] for x in object_index_sets)
    models = {m: aggregate_episodes([r for r in rows if r["model"] == m]) for m in MODELS}
    return {
        "rows": len(rows), "models": models, "seed_labels": seeds, "objects": objects,
        "by_object": [{"object": obj, **{m: aggregate_episodes(by_object[(m, obj)]) for m in MODELS}}
                      for obj in objects],
        "by_seed": [{"seed": seed, **{m: aggregate_episodes(by_seed[(m, seed)]) for m in MODELS}}
                    for seed in seeds],
        "delta_success_rate_dinov3_minus_baseline": models["dinov3"]["success_rate"] - models["baseline"]["success_rate"],
        "integrity": {"duplicate_episode_keys": 0, "missing_fields": 0, "paired_keys_across_models": True,
                      "same_object_index_set_per_seed": balanced},
        "confidence_intervals": None,
        "uncertainty_note": "Seed labels alone do not establish independent evaluation runs or training replications.",
    }


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_seed_bootstrap(robotics, replicates, rng_seed):
    """Conditional inference: resample whole paired seed blocks, preserving all objects.

    Call only after explicitly accepting independent/exchangeable seed clusters.
    The same sampled seed indices are used for both models; never sample episodes.
    """
    require(replicates >= 100, "Bootstrap requires at least 100 replicates")
    groups = robotics["by_seed"]
    require(len(groups) >= 2, "Bootstrap needs at least two seed clusters")
    rng = random.Random(rng_seed)
    draws = {m: [] for m in (*MODELS, "delta")}
    for _ in range(replicates):
        sampled = rng.choices(groups, k=len(groups))
        rates = {}
        for model in MODELS:
            rates[model] = sum(g[model]["successes"] for g in sampled) / sum(g[model]["episodes"] for g in sampled)
            draws[model].append(rates[model])
        draws["delta"].append(rates["dinov3"] - rates["baseline"])
    return {"method": "paired seed-cluster percentile bootstrap", "level": 0.95,
            "replicates": replicates, "rng_seed": rng_seed,
            "assumption": "Independent exchangeable evaluation-seed clusters; fixed object set and policies. Not established by CSV.",
            "intervals": {m: [percentile(v, 0.025), percentile(v, 0.975)] for m, v in draws.items()}}


def compare_summary(segmentation, recovered):
    require(all(isinstance(recovered.get(m), dict) for m in MODELS), "Summary lacks model sections")
    comparison = recovered.get("comparison", {})
    require(isinstance(comparison, dict), "Invalid comparison metadata")
    matches, mismatches = [], []

    def compare(key, observed, expected):
        equal = (observed is None and expected is None) or (
            isinstance(observed, (int, float)) and isinstance(expected, (int, float))
            and math.isfinite(expected) and math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10)
        )
        if equal:
            matches.append(key)
        else:
            # Export numeric discrepancies only; do not echo arbitrary input strings.
            mismatches.append({"metric": key, "recomputed": observed,
                               "recovered": expected if isinstance(expected, (int, float)) and math.isfinite(expected) else None})

    for model in MODELS:
        for key, expected in recovered[model].items():
            if key in segmentation["models"][model]:
                compare(model + "." + key, segmentation["models"][model][key], expected)
    for mismatch in mismatches:
        model, metric = mismatch["metric"].split(".", 1)
        if metric == "latency_ms_median":
            alternate = segmentation["models"][model]["latency_ms_lower_middle_float32"]
            mismatch["compatible_with_lower_middle_float32"] = (
                mismatch["recovered"] is not None and math.isclose(alternate, mismatch["recovered"], abs_tol=1e-10)
            )
    if "samples" in comparison:
        compare("comparison.samples", segmentation["samples"], comparison["samples"])
    for key, expected in comparison.items():
        if key.startswith("delta_"):
            metric = key[len("delta_"):]
            if metric in segmentation["models"]["baseline"]:
                base, dino = (segmentation["models"][m][metric] for m in MODELS)
                compare("comparison." + key, dino - base if base is not None and dino is not None else None, expected)
    return {"numeric_tolerance": {"relative": 1e-10, "absolute": 1e-10},
            "matching_fields": matches, "mismatches": mismatches}


def recovered_conditions(recovered, segmentation):
    """Export only inspected, typed metadata; never copy arbitrary JSON content."""
    source = recovered.get("comparison", {})
    conditions = {}
    for key in ("device", "evaluation_mode"):
        if key in source:
            value = source[key]
            require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9 _().+-]{1,100}", value)),
                    "Unsafe public metadata string")
            conditions[key] = value
    if "amp" in source:
        require(isinstance(source["amp"], bool), "AMP flag must be boolean")
        conditions["amp"] = source["amp"]
    for model in MODELS:
        for suffix in ("input_size", "radius"):
            key = model + "_" + suffix
            if key in source:
                value = integer(source[key])
                require(value > 0, "Input size and radius must be positive")
                conditions[key] = value
        if model + "_input_size" in conditions:
            require(segmentation["models"][model]["pixels_per_image"] == [conditions[model + "_input_size"] ** 2],
                    "Summary input size disagrees with CSV pixel counts")
    return conditions


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(inputs):
    recorded = (inputs / "git_commit.txt").read_text(encoding="utf-8-sig").strip()
    require(bool(re.fullmatch(r"[0-9a-fA-F]{40}", recorded)), "Recorded commit must be a full Git SHA")
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"source_files": {name: {"sha256": sha256(inputs / name), "bytes": (inputs / name).stat().st_size}
                             for name in SOURCE_NAMES},
            "recorded_experiment_commit": recorded,
            "analysis_base_commit": revision, "analysis_script_sha256": sha256(Path(__file__)),
            "python_version": platform.python_version(),
            "scope": "Recalculation from recovered artifacts; training, inference and simulation were not rerun.",
            "raw_rows_published": False}


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "vapo-results"})
    colors = ("#2563a6", "#db7824")

    def save(fig, name):
        fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        fig.savefig(output / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)

    seg = report["segmentation"]["models"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    metrics = ("miou_binary_global", "dice_f1_global", "recall_global")
    for j, model in enumerate(MODELS):
        bars = ax.bar([i + (j - .5) * .34 for i in range(len(metrics))],
                      [seg[model][k] for k in metrics], .34, label=LABELS[model], color=colors[j])
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(xticks=range(len(metrics)), xticklabels=["Binary mIoU", "Foreground Dice", "Foreground recall"],
           ylim=(0, 1), ylabel="Score", title="Perception metrics from recovered confusion counts")
    ax.legend(loc="upper left", ncols=2)
    conditions = report["recovered_conditions"]
    native = " vs ".join(f"{conditions.get(m + '_input_size', '?')} px / radius {conditions.get(m + '_radius', '?')}"
                         for m in MODELS)
    fig.supxlabel(f"{report['segmentation']['samples']:,} paired images; native preprocessing: {native}", fontsize=9)
    # All plotted values come from the inputs; conditions are detailed in the report.
    save(fig, "perception")

    robot = report["robotics"]
    fig, ax = plt.subplots(figsize=(7, 4.7), layout="constrained")
    bars = ax.bar([LABELS[m] for m in MODELS], [robot["models"][m]["success_rate"] for m in MODELS],
                  width=.55, color=colors)
    for bar, model in zip(bars, MODELS):
        result = robot["models"][model]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .015,
                f"{result['success_rate']:.2%}\n{result['successes']:,} / {result['episodes']:,}", ha="center")
    ax.set(ylim=(0, 1), ylabel="Task success rate", title="Robotic task success from recovered episode records")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.supxlabel(f"{len(robot['seed_labels'])} labeled seeds; fixed object set; no independence assumption or CI", fontsize=9)
    save(fig, "robotic_success")

    objects = robot["by_object"]
    fig, ax = plt.subplots(figsize=(9, max(4, len(objects) * .42)), layout="constrained")
    for j, model in enumerate(MODELS):
        bars = ax.barh([i + (j - .5) * .35 for i in range(len(objects))],
                       [o[model]["success_rate"] for o in objects], .35, label=LABELS[model], color=colors[j])
        ax.bar_label(bars, labels=[f"{o[model]['success_rate']:.0%}" for o in objects], padding=3, fontsize=8)
    ax.set(yticks=range(len(objects)), yticklabels=[o["object"] for o in objects], xlim=(0, 1.16),
           xlabel="Success rate", title="Success by object under the recorded evaluation")
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.supxlabel("Object labels from CSV; counts and rates in robotics_by_object.csv", fontsize=9)
    save(fig, "robotic_by_object")
    return {"matplotlib_version": matplotlib.__version__,
            "latency_plot": "Not generated: recorded native-pipeline timings are descriptive; source protocol and historical-command evidence are in the separate provenance and identity audits."}


def run_analysis(inputs, output, plots=False, bootstrap_replicates=0, bootstrap_seed=0):
    inputs, output = Path(inputs), Path(output)
    require(not output.resolve().is_relative_to(inputs.resolve()), "Output must be outside the source directory")
    # Validate everything before creating or replacing outputs.
    prov = provenance(inputs)
    segmentation = analyze_segmentation(inputs / "metricas_por_imagem.csv")
    robotics = analyze_robotics(inputs / "ab_results.csv")
    recovered = json.loads((inputs / "resumo.json").read_text(encoding="utf-8-sig"))
    require(isinstance(recovered, dict), "Summary must be a JSON object")
    conditions = recovered_conditions(recovered, segmentation)
    consistency = compare_summary(segmentation, recovered)
    if bootstrap_replicates:
        robotics["confidence_intervals"] = paired_seed_bootstrap(robotics, bootstrap_replicates, bootstrap_seed)
    report = {"schema_version": 1, "provenance": prov, "recovered_conditions": conditions,
              "segmentation": segmentation, "robotics": robotics, "summary_consistency": consistency,
              "limitations": ["Native resolutions and label radii differ between approaches.",
                              "The four metric files alone lack model/policy checkpoint identities and evaluation commands.",
                              "Effective simulator seed propagation is not established by the CSV alone; see the separate provenance audit.",
                              "The four metric files lack timing protocol details; source-level protocol is in the separate provenance audit.",
                              "The standalone commit file alone does not establish runtime provenance; see the recovered run logs audit."]}
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "perception_metrics.csv", [{"model": m, **{k: v for k, v in segmentation["models"][m].items()
                                                                  if isinstance(v, (int, float)) or v is None}}
                                                   for m in MODELS])
    write_table(output / "robotics_overall.csv", [{"model": m, **robotics["models"][m]} for m in MODELS])
    for group, identifier in (("by_object", "object"), ("by_seed", "seed")):
        write_table(output / ("robotics_" + group + ".csv"),
                    [{identifier: row[identifier], "model": m, **row[m]} for row in robotics[group] for m in MODELS])
    if plots:
        report["plots"] = make_plots(report, output)
    (output / "analysis_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Private folder containing the four recovered files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for aggregate artifacts")
    parser.add_argument("--plots", action="store_true", help="Generate three Matplotlib PNG/SVG figures")
    parser.add_argument("--assume-independent-seed-clusters", action="store_true",
                        help="Explicitly accept the conditional paired seed bootstrap assumption")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    try:
        report = run_analysis(args.input_dir, args.output_dir, args.plots,
                              args.bootstrap_replicates if args.assume_independent_seed_clusters else 0,
                              args.bootstrap_seed)
    except (ValidationError, OSError, json.JSONDecodeError, ImportError) as exc:
        # Avoid exposing private paths or input content in logs intended for publication.
        parser.exit(2, "Analysis failed (" + type(exc).__name__ + "). Check the local inputs and dependencies.\n")
    print(f"Validated {report['segmentation']['samples']} images and {report['robotics']['rows']} robotic records.")
    print(f"Summary numeric mismatches: {len(report['summary_consistency']['mismatches'])}.")


if __name__ == "__main__":
    main()
