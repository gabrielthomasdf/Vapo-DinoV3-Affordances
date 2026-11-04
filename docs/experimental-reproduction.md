# Recalculating the recovered experiments

This workflow reproduces calculations from saved experimental records. It does not run training, model inference, or robot simulation. Source CSVs remain private; aggregate outputs, scripts, tests, and figures are public.

## Inputs and execution

Use Python 3.10 or later. Core analysis needs only the standard library. Supply a directory containing these exact filenames:

| File | Purpose |
| --- | --- |
| `metricas_por_imagem.csv` | Paired sample IDs, per-model confusion counts, derived scores, and timings |
| `resumo.json` | Recovered summary to cross-check, plus device and native-preprocessing metadata |
| `ab_results.csv` | Model label, seed label, episode index, object, binary success, return, and steps |
| `git_commit.txt` | Recorded full Git SHA; association with specific weights remains unverified |

Run from the repository root. The input and output directories must be separate:

```bash
python scripts/analyze_recovered_results.py --input-dir /path/to/private/recovered-files --output-dir /path/to/derived-results
```

For the portfolio figures:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/analyze_recovered_results.py --input-dir /path/to/private/recovered-files --output-dir /path/to/derived-results --plots
```

Paths are command-line arguments and work on Windows, Linux, and WSL. The checked-in outputs are in [experiments/recovered](experiments/recovered/analysis_summary.json). The recalculation was executed with Python 3.14.6 and Matplotlib 3.11.2 on Windows; it takes seconds on CPU. Re-rendering figures can vary across Matplotlib/font versions. The mathematical analysis does not require NumPy, pandas, PyTorch, GPU access, datasets, or checkpoints.

## Outputs and provenance

| Output | Contents |
| --- | --- |
| `analysis_summary.json` | Source filenames, byte counts, SHA-256 hashes, recorded experiment SHA, analysis base SHA and script hash, recovered conditions, integrity checks, numeric results and discrepancies |
| `perception_metrics.csv` | Counts, global scores, per-image/batch aggregation, descriptive latency statistics |
| `robotics_overall.csv` | Episodes, successes, rates, mean return and steps per approach |
| `robotics_by_object.csv` | Aggregates per object and approach |
| `robotics_by_seed.csv` | Aggregates per labeled seed and approach, preserving evaluation blocks |
| `perception.png/.svg` | Global binary mIoU, foreground Dice and recall |
| `robotic_success.png/.svg` | Overall robotic success counts and rates |
| `robotic_by_object.png/.svg` | Success rates by recorded object identifier |

The exporter does not copy sample identifiers, raw rows, input-directory paths, arbitrary JSON strings, weights, or datasets. Device and preprocessing metadata are exported through a small typed allowlist. Source hashes identify the exact private inputs. The analysis base SHA is the checkout used during execution; the separate script hash identifies the implementation, including uncommitted changes. Machine/Python/plotting versions can change provenance fields without changing scores.

## Metric definitions

Let `TP`, `FP`, `FN`, and `TN` denote foreground confusion counts summed over all records **within one approach**. Counts are never pooled across the baseline and DINOv3 grids.

- Foreground IoU = `TP / (TP + FP + FN)`.
- Background IoU = `TN / (TN + FP + FN)`.
- Binary global mIoU = `(foreground IoU + background IoU) / 2`.
- Foreground Dice/F1 = `2 TP / (2 TP + FP + FN)`.
- Precision = `TP / (TP + FP)`; recall = `TP / (TP + FN)`.
- Accuracy = `(TP + TN) / (TP + FP + FN + TN)`.
- Robotic success rate = `sum(success) / number of episode records`.

The per-image CSV includes foreground IoU/Dice/precision/recall, not binary mIoU. Its derived values are checked against each row's integer counts at absolute/relative tolerance `1e-10`. Zero denominators require a `NaN` in the CSV and are reported as undefined, not silently imputed. Exported JSON uses `null`, never nonstandard JSON `NaN`.

`vapo_iou_mean_per_image` is the arithmetic mean of **defined** per-image foreground IoUs. It gives each defined image equal weight, unlike a ratio of global counts.

`vapo_val_iou_batch2` reconstructs the recovered summary by sorting on `index`, combining consecutive groups of two images, taking foreground intersection/union for each batch, and averaging **defined** batch IoUs. It is a reconstruction based on CSV order; the original evaluator's ordering is not available.

The actual `compute_mIoU` in [losses.py](../vapo/affordance/utils/losses.py) computes **foreground-only** IoU over a batch and uses `max(union, 1e-6)`, giving zero when the union is empty. Therefore the analysis also exports `training_foreground_iou_batch_mean_empty_zero`. For DINOv3, the recovered convention gives 0.728521 after excluding 28 empty batches, while the training convention gives 0.722132 over all 3,193 batches. These are different from binary global mIoU 0.819045. The training logger also omits the final batch from its console mean; checkpoint, distributed reduction, and sample order remain needed to match a specific epoch exactly.

Latency means and medians are computed directly from each image's positive finite millisecond value. The conventional median averages the two central values when sample count is even. The JSON's two stored medians use the lower middle sample rounded to float32; the report retains the conventional median discrepancy and an explicit compatibility flag. The supplementary recovered evaluator confirms this implementation with `torch.tensor(latencies).median()`.

## Validation and interpretation

The validator rejects missing required columns, duplicate column names, malformed/empty records, duplicate sample IDs, missing/duplicate image indices, negative confusion counts, nonfinite timings, inconsistent derived metrics, invalid success flags, duplicate episode keys, and unpaired robotic keys across approaches. It checks `positive_pixels_gt = TP + FN` and verifies the summary's input sizes against CSV pixel totals.

Robotic keys use `(model, seed, object, episode_index)` because episode indices restart within each seed. Matching keys establish a paired **record structure**; they do not prove matching physical simulator states, identical policies, or effective use of the seed. Completeness is checked against internal indices and cross-model pairing; absent records in both files cannot be detected without the original evaluation plan.

The recovered setup compares native preprocessing: baseline 64 px / radius 10 versus DINOv3 128 px / radius 20. The foreground definitions are consequently related but not the exact same pixel labels. No resized-mask comparison or controlled encoder-only effect can be reconstructed from confusion counts alone.

The recorded device is NVIDIA RTX A5500 and AMP is enabled. The supplementary source confirms batch size 1, synchronized forward-only timing and a warmup default of 20. The actual warmup override and full invocation are missing. Recorded latencies remain descriptive; the conditions do not support an isolated encoder benchmark and the fourth latency figure is omitted.

## Recovered provenance package

The private package `ic_vapo_proveniencia_20261104.tar.gz` contains the perception evaluator, robotic evaluator, runners for both approaches, CSV parser, DINOv3 configuration, metadata, run logs and a nested archive with **200 configs + 200 overrides**. Audit it without extracting or executing its contents:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/audit_recovered_provenance.py --archive /path/to/private/provenance.tar.gz --metrics-dir /path/to/private/recovered-files --output /path/to/derived-results/provenance_audit.json
```

When the second saved-policy/training package is available, add `--supplement /path/to/private/pendencias.tar.gz`. This checks evaluator-source agreement, audits the saved configs and streams the large training console without extracting it. The checked-in report includes both packages.

The [checked report](experiments/recovered/provenance_audit.json) verifies all 3,000 typed CSV rows against 200 logs, final success summaries, exit status zero for every run, and seed overrides matching every per-seed Hydra snapshot. It exports source hashes, config-difference field names, selected package versions and checkpoint filenames; it excludes private absolute paths and raw logs.

Batch metadata and all run logs report `4caeecf338c9a758607f1737adf222150806f9cb`; the archived evaluator matches that Git version. Earlier captures still record `bbded78`. The final code applies `test.eval_seed` to the environment before construction, explaining why the saved pre-execution Hydra `env.seed` remains null while the requested evaluation seed varies. The runner starts a new process for each seed. Neither this nor the repeated object IDs proves identical physical initial states or control of every global RNG.

Both approaches reference the same SAC path/name (`most_tasks_from_15.pth`). Recovered gripper references are `gripper_tabletop.ckpt` and `gripper_seed0/trained_models/best_val_miou.ckpt`. Static affordance references are unchanged. Content hashes for these weights remain absent. The composed CLI configs differ in encoder, decoder widths, gripper checkpoint, preprocessing, resize and affordance-preservation settings; these differences must be retained when interpreting the comparison. The second package supplies the saved SAC config loaded by `load_eval_cfg`: `vapo/.hydra/config.yaml`, with affordance inputs, distance and reward shaping enabled and `env.use_egl=false`. A separate no-affordance config named `baseline` is not selected by the A/B references. The CLI files are still not complete final runtime dumps, but the recovered saved environment now resolves the earlier EGL-setting uncertainty.

The environment snapshot records Python 3.11.15, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Lightning 1.5.9, segmentation-models-pytorch 0.3.3, NumPy 1.26.4 and PyBullet 3.2.7. It was recovered rather than reinstalled or revalidated. The old upstream environment instructions remain historical; they are not the recovered run's environment specification.

The second package also supplies the DINOv3 training Hydra config/overrides: simulation/gripper, 128 px, radius 20, batch size 2, four workers, 50 epochs, one GPU, precision 16 and no distributed strategy. It uses historical environment variables `DINOV3_REPO` and `DINOV3_WEIGHTS`; current `_DIR`/`_PATH` names do not resolve those old interpolations automatically. The console identifies the best checkpoint save at epoch 28, step 1,936,155, with `val_miou=0.72213`. No training-seed override is recorded despite the directory label `gripper_seed0`.

### Recovered perception evaluator

The reviewed script is preserved verbatim at [scripts/eval_affordance_dinov3_vs_baseline.py](../scripts/eval_affordance_dinov3_vs_baseline.py). Its SHA-256 matches the private archive. It requires CUDA and the research dependencies, config, data and weights. A reconstructed command using the recovered script defaults is:

```bash
python scripts/eval_affordance_dinov3_vs_baseline.py --dino-config /path/to/dino-training/.hydra/config.yaml --dino-checkpoint /path/to/dino-best.ckpt --baseline-checkpoint /path/to/gripper_tabletop.ckpt --output-dir /path/to/new-inference-results --amp --warmup 20 --num-workers 1
```

This is a template rather than a recovered exact invocation. Do not treat its default 20 warmup iterations as proof of the original command. The source selects the validation split, checks paired sample ordering, uses `shuffle=False`, and evaluates each model at its native preprocessing settings. Models are in eval/inference mode with optional FP16 autocast. Each timing uses CUDA synchronization before/after the forward pass and `perf_counter`; it includes encoder, decoder, segmentation and direction heads and excludes data loading, image/device transfer, argmax, Hough voting and SAC. Baseline always runs before DINOv3 for each image. cuDNN benchmarking is enabled; no general determinism guarantee follows from the source's `torch.manual_seed(0)`.

## Statistical uncertainty

No confidence intervals are generated by default. Consecutive perception frames can be correlated. Robotic episodes share seeds, objects and policies, and the 100 seed labels are not independent training repetitions. The supplementary code/configs support environment-seed propagation and a separate process per seed, but do not establish control of all RNG streams or paired physical initial states.

If independent, exchangeable evaluation-seed blocks can later be justified from the runner and logs, an explicitly conditional analysis is available:

```bash
python scripts/analyze_recovered_results.py --input-dir /path/to/private/recovered-files --output-dir /path/to/conditional-results --assume-independent-seed-clusters --bootstrap-replicates 10000 --bootstrap-seed 0
```

This resamples **whole paired seed blocks**, retaining all objects/episodes and using the same draw for both approaches. It produces percentile 95% intervals for each success rate and their difference. It is conditional on the fixed policies and object set and does not quantify uncertainty across training seeds or unseen objects. That assumption has not been accepted for the checked-in artifacts, whose `confidence_intervals` field is `null`.

## Calculation tests

```bash
python -m pytest -p no:cacheprovider tests/test_recovered_results_analysis.py -q
python -m pytest -p no:cacheprovider tests/test_recovered_provenance_audit.py -q
```

Tests use small synthetic arithmetic fixtures to check calculations, undefined values, duplicate/unpaired records, median conventions, paired cluster resampling, and export privacy. These fixtures validate the analysis software and make no claim about scientific model performance.

## Full experimental reproduction still requires

| Experiment | Available | Missing for a complete rerun |
| --- | --- | --- |
| Perception | Evaluator source, counts/timings, native settings, AMP/device metadata, validation selection, DINOv3 training config/overrides and source timing protocol | Exact invocation/warmup override; split manifest and dataset version; checkpoint content hashes; DINOv3 code/weight revision |
| Robotics | Runners, parser, all 200 logs/statuses, 400 Hydra files, environment snapshot, saved SAC configs, checkpoint references, runtime commit and source seed propagation | Checkpoint content hashes; final merged runtime-state capture; all-RNG/reset/state pairing validation; full environment rerun |
| Training | Saved DINOv3 Hydra config/overrides, 50 epochs, batch size 2, sample counts and checkpoint-selection log at epoch 28 | Explicit training-seed evidence; immutable dataset/environment identity; content-hash link to evaluated weights |

The templates in the [README](../README.md#level-2-full-experiments) show entry points, not a verified regeneration of these records. No new model training, inference, simulator rollout, or weight download was performed during consolidation.
