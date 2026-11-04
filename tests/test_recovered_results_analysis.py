"""Calculation and integrity tests; no Torch, simulation, or private files needed."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_recovered_results.py"
spec = importlib.util.spec_from_file_location("recovered_results", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def segmentation_row(index, tp, fp, fn, tn):
    row = {"index": index, "sample": f"/private/sample_{index}", "delta_iou_dinov3_minus_baseline": 0}
    fg = analysis.scores(dict(zip(analysis.COUNTS, (tp, fp, fn, tn))))
    for model in analysis.MODELS:
        row.update({model + "_" + k: v for k, v in zip(analysis.COUNTS, (tp, fp, fn, tn))})
        for name in analysis.PER_IMAGE:
            value = fg[name + "_global"]
            row[model + "_" + name] = value if value is not None else "nan"
        row[model + "_positive_pixels_gt"] = tp + fn
        row[model + "_latency_ms"] = index + 1
    if fg["iou_affordance_global"] is None:
        row["delta_iou_dinov3_minus_baseline"] = "nan"
    return row


def robot_rows():
    # Two objects share a seed; cluster resampling must keep them together.
    return [{"model": model, "seed": seed, "episode_index": index, "object": obj,
             "success": seed, "return": 10, "steps": 3}
            for model in analysis.MODELS for seed in (0, 1) for index, obj in enumerate(("a", "b"))]


def test_global_scores_are_ratios_of_summed_counts_not_means(tmp_path):
    path = tmp_path / "seg.csv"
    write_csv(path, [segmentation_row(0, 1, 1, 0, 2), segmentation_row(1, 1, 0, 0, 3)])
    result = analysis.analyze_segmentation(path)["models"]["baseline"]
    assert result["iou_affordance_global"] == pytest.approx(2 / 3)
    assert result["vapo_iou_mean_per_image"] == pytest.approx(3 / 4)
    assert result["iou_background_global"] == pytest.approx(5 / 6)
    assert result["miou_binary_global"] == pytest.approx(3 / 4)
    assert result["dice_f1_global"] == pytest.approx(4 / 5)
    assert result["precision_global"] == pytest.approx(2 / 3)
    assert result["recall_global"] == 1


def test_empty_unions_are_reported_and_training_convention_is_separate(tmp_path):
    path = tmp_path / "seg.csv"
    write_csv(path, [segmentation_row(0, 0, 0, 0, 4), segmentation_row(1, 0, 0, 0, 4),
                     segmentation_row(2, 2, 0, 0, 2)])
    result = analysis.analyze_segmentation(path)["models"]["dinov3"]
    assert result["undefined_per_image_metrics"]["iou_affordance"] == 2
    assert result["empty_union_batches"] == 1
    assert result["batch_count"] == 2
    assert result["vapo_val_iou_batch2"] == 1
    assert result["training_foreground_iou_batch_mean_empty_zero"] == .5


@pytest.mark.parametrize("mutation", ("duplicate_sample", "missing_index", "incorrect_score", "infinity", "negative_count", "missing_value"))
def test_invalid_segmentation_is_rejected(tmp_path, mutation):
    rows = [segmentation_row(0, 1, 1, 0, 2), segmentation_row(1, 1, 0, 0, 3)]
    if mutation == "duplicate_sample":
        rows[1]["sample"] = rows[0]["sample"]
    elif mutation == "missing_index":
        rows[1]["index"] = 3
    elif mutation == "incorrect_score":
        rows[0]["baseline_dice_f1"] = 1
    elif mutation == "infinity":
        rows[0]["baseline_latency_ms"] = "inf"
    elif mutation == "negative_count":
        rows[0]["baseline_tn"] = -1
    else:
        rows[0]["baseline_tn"] = ""
    path = tmp_path / "seg.csv"
    write_csv(path, rows)
    with pytest.raises(analysis.ValidationError):
        analysis.analyze_segmentation(path)


def test_robotic_keys_include_object_because_episode_indices_restart(tmp_path):
    rows = robot_rows()
    for row in rows:
        row["episode_index"] = 0
    path = tmp_path / "robot.csv"
    write_csv(path, rows)
    result = analysis.analyze_robotics(path)
    assert result["models"]["baseline"]["successes"] == 2
    assert result["models"]["dinov3"]["success_rate"] == .5
    assert result["integrity"]["same_object_index_set_per_seed"]
    assert result["confidence_intervals"] is None


@pytest.mark.parametrize("mutation", ("duplicate", "unpaired", "bad_success", "unsafe_label"))
def test_invalid_robotic_records_are_rejected(tmp_path, mutation):
    rows = robot_rows()
    if mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "unpaired":
        rows.pop()
    elif mutation == "bad_success":
        rows[0]["success"] = 2
    else:
        rows[0]["object"] = "private/path"
    path = tmp_path / "robot.csv"
    write_csv(path, rows)
    with pytest.raises(analysis.ValidationError):
        analysis.analyze_robotics(path)


def test_bootstrap_is_paired_by_seed_and_preserves_perfect_model_matching(tmp_path):
    path = tmp_path / "robot.csv"
    write_csv(path, robot_rows())
    result = analysis.analyze_robotics(path)
    ci = analysis.paired_seed_bootstrap(result, 200, 7)
    assert ci == analysis.paired_seed_bootstrap(result, 200, 7)
    assert ci["intervals"]["delta"] == [0, 0]
    assert ci["intervals"]["baseline"] == [0, 1]


def test_summary_median_mismatch_is_not_hidden(tmp_path):
    path = tmp_path / "seg.csv"
    write_csv(path, [segmentation_row(0, 1, 1, 0, 2)])
    seg = analysis.analyze_segmentation(path)
    recovered = {"baseline": {"latency_ms_mean": 1, "latency_ms_median": 2}, "dinov3": {}, "comparison": {"samples": 1}}
    result = analysis.compare_summary(seg, recovered)
    assert result["matching_fields"] == ["baseline.latency_ms_mean", "comparison.samples"]
    assert result["mismatches"] == [{"metric": "baseline.latency_ms_median", "recomputed": 1, "recovered": 2,
                                     "compatible_with_lower_middle_float32": False}]


def test_even_sample_median_convention_explains_lower_middle_summary(tmp_path):
    path = tmp_path / "seg.csv"
    write_csv(path, [segmentation_row(0, 1, 1, 0, 2), segmentation_row(1, 1, 0, 0, 3)])
    seg = analysis.analyze_segmentation(path)
    result = analysis.compare_summary(seg, {"baseline": {"latency_ms_median": 1}, "dinov3": {}})
    assert result["mismatches"][0]["recomputed"] == 1.5
    assert result["mismatches"][0]["compatible_with_lower_middle_float32"]


def test_end_to_end_export_is_strict_json_and_contains_no_raw_sample_paths(tmp_path):
    inputs = tmp_path / "private"
    inputs.mkdir()
    write_csv(inputs / "metricas_por_imagem.csv", [segmentation_row(0, 1, 1, 0, 2)])
    write_csv(inputs / "ab_results.csv", robot_rows())
    (inputs / "resumo.json").write_text(json.dumps({"baseline": {}, "dinov3": {}, "comparison": {}}))
    (inputs / "git_commit.txt").write_text("a" * 40)
    output = tmp_path / "public"
    result = analysis.run_analysis(inputs, output)
    text = (output / "analysis_summary.json").read_text()
    assert "/private/sample" not in text
    assert str(inputs) not in text
    assert "NaN" not in text
    assert json.loads(text)["segmentation"]["samples"] == 1
    assert result["provenance"]["source_files"]["ab_results.csv"]["sha256"] == analysis.sha256(inputs / "ab_results.csv")
    assert len(list(output.glob("*.csv"))) == 4
