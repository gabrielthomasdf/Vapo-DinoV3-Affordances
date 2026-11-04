# Recalculating the recovered experiments

This workflow audits completed experiments and reproduces calculations from saved records. It does not run training, model inference, or robot simulation, and repeating those experiments is not a completion requirement. Source CSVs and full split lists remain private; aggregate outputs, scripts, tests, hashes and derived figures are public.

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
| `identity_audit.json` | Six recorded checkpoint identities, hashes of four split manifests, ordered-ID fingerprints and command evidence levels |
| `checkpoints.sha256` | Recorded weight digests with logical relative paths, excluding private account paths |

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

`vapo_val_iou_batch2` reconstructs the recovered summary by sorting on `index`, combining consecutive groups of two images, taking foreground intersection/union for each batch, and averaging **defined** batch IoUs. The recovered source confirms ordered validation loading, and the four split manifests now reproduce the evaluated CSV's sample ordering exactly.

The actual `compute_mIoU` in [losses.py](../vapo/affordance/utils/losses.py) computes **foreground-only** IoU over a batch and uses `max(union, 1e-6)`, giving zero when the union is empty. Therefore the analysis also exports `training_foreground_iou_batch_mean_empty_zero`. For DINOv3, the recovered convention gives 0.728521 after excluding 28 empty batches, while the training convention gives 0.722132 over all 3,193 batches. These are different from binary global mIoU 0.819045. The training logger also omits the final batch from its console mean; checkpoint, distributed reduction, and sample order remain needed to match a specific epoch exactly.

Latency means and medians are computed directly from each image's positive finite millisecond value. The conventional median averages the two central values when sample count is even. The JSON's two stored medians use the lower middle sample rounded to float32; the report retains the conventional median discrepancy and an explicit compatibility flag. The supplementary recovered evaluator confirms this implementation with `torch.tensor(latencies).median()`.

## Validation and interpretation

The validator rejects missing required columns, duplicate column names, malformed/empty records, duplicate sample IDs, missing/duplicate image indices, negative confusion counts, nonfinite timings, inconsistent derived metrics, invalid success flags, duplicate episode keys, and unpaired robotic keys across approaches. It checks `positive_pixels_gt = TP + FN` and verifies the summary's input sizes against CSV pixel totals.

Robotic keys use `(model, seed, object, episode_index)` because episode indices restart within each seed. Matching keys establish a paired **record structure**; they do not prove matching physical simulator states, identical policies, or effective use of the seed. Completeness is checked against internal indices and cross-model pairing; absent records in both files cannot be detected without the original evaluation plan.

The recovered setup compares native preprocessing: baseline 64 px / radius 10 versus DINOv3 128 px / radius 20. The foreground definitions are consequently related but not the exact same pixel labels. No resized-mask comparison or controlled encoder-only effect can be reconstructed from confusion counts alone.

The recorded device is NVIDIA RTX A5500 and AMP is enabled. Source review confirms batch size 1 and synchronized forward-only timing. The later command text specifies `--warmup 5 --amp`, overriding the source default of 20; actual execution with these flags is not independently attested by a contemporaneous process log. Latencies remain descriptive native-pipeline measurements, and no isolated encoder latency figure is presented.

## Checkpoint, split and command evidence

Three further recovered inputs are inspected without reading checkpoint binaries or invoking the evaluator:

| Private input | Check |
| --- | --- |
| `vapo_checkpoint_hashes_20261104.txt` | Six valid SHA-256 records; mapping by logical artifact suffix, including disambiguation of the VAPO and no-affordance SAC policies |
| `vapo_dataset_splits_20261104.tar.gz` | All four JSONs; loader-compatible gripper filtering/order; duplicate and training/validation overlap checks; exact CSV comparison |
| `vapo_comando_percepcao_20261104.txt` | Argument compatibility by static source inspection; warmup override and AMP agreement with recovered summary |

```bash
python scripts/audit_experiment_identity.py --audit-dir /path/to/private/audit-files --metrics-dir /path/to/private/recovered-files --output-dir /path/to/derived-results
```

The standard-library-only audit outputs [identity_audit.json](experiments/recovered/identity_audit.json) and [checkpoints.sha256](experiments/recovered/checkpoints.sha256). It preserves recorded digests while replacing internal account paths with public relative references. A supplied digest is a recorded artifact identity; this audit does not independently rehash weight files or certify historical load-time contents.

The merged manifest has 133,528 training gripper IDs and **6,386 validation IDs exactly matching the evaluated CSV in order**. Scene subsets are kitchen 44,733/2,108, miscellaneous 42,120/1,555 and tools 46,675/2,723 (training/validation). All three scene manifests reproduce the merged subsets and their union. There are no duplicate IDs or training/validation overlaps. Manifests therefore correspond to the evaluated partition at sample identity/order level; they do not attest raw image/label bytes, an immutable dataset release or the bytes of a JSON physically present during the historical process. Full lists stay private; byte hashes, counts and canonical ordered-ID fingerprints are exported.

The supplied command text records `python eval_affordance_dinov3_vs_baseline.py --warmup 5 --amp`. Its warmup override is valid and AMP matches `resumo.json`. This is a reported historical command consistent with source/metadata, not a contemporaneous execution trace. The identity report explicitly leaves `actual_runtime_warmup_independently_attested` false.

## Recovered provenance package

The private package `ic_vapo_proveniencia_20261104.tar.gz` contains the perception evaluator, robotic evaluator, runners for both approaches, CSV parser, DINOv3 configuration, metadata, run logs and a nested archive with **200 configs + 200 overrides**. Audit it without extracting or executing its contents:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/audit_recovered_provenance.py --archive /path/to/private/provenance.tar.gz --metrics-dir /path/to/private/recovered-files --output /path/to/derived-results/provenance_audit.json
```

When the second saved-policy/training package is available, add `--supplement /path/to/private/pendencias.tar.gz`. To attach the later identity evidence, also add `--identity-report /path/to/derived-results/identity_audit.json`. The verifier checks metric-source hashes and the evaluator hash before linking the reports. It streams the large console without extracting it. The checked-in report includes both packages and the later identity audit.

The [checked report](experiments/recovered/provenance_audit.json) verifies all 3,000 typed CSV rows against 200 logs, final success summaries, exit status zero for every run, and seed overrides matching every per-seed Hydra snapshot. It exports source hashes, config-difference field names, selected package versions and checkpoint filenames; it excludes private absolute paths and raw logs.

Batch metadata and all run logs report `4caeecf338c9a758607f1737adf222150806f9cb`; the archived evaluator matches that Git version. Earlier captures still record `bbded78`. The final code applies `test.eval_seed` to the environment before construction, explaining why the saved pre-execution Hydra `env.seed` remains null while the requested evaluation seed varies. The runner starts a new process for each seed. Neither this nor the repeated object IDs proves identical physical initial states or control of every global RNG.

Both approaches reference the same SAC path/name (`most_tasks_from_15.pth`), associated in the later manifest with SHA-256 `586ca6bfcc0a4cf4e134f28901341d351b3808097b4e6900a50a796c6e01dc3f`. Recovered gripper references are `gripper_tabletop.ckpt` and `gripper_seed0/trained_models/best_val_miou.ckpt`; their recorded digests and the backbone/static/no-affordance references are in `checkpoints.sha256`. The composed CLI configs differ in encoder, decoder widths, gripper checkpoint, preprocessing, resize and affordance-preservation settings; these differences must be retained when interpreting the comparison. The saved SAC config is `vapo/.hydra/config.yaml`, with affordance inputs, distance and reward shaping enabled and `env.use_egl=false`. A separate no-affordance config named `baseline` is not selected by the A/B references. The CLI files are not complete final runtime dumps, but the recovered saved environment resolves the earlier EGL-setting uncertainty.

The environment snapshot records Python 3.11.15, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Lightning 1.5.9, segmentation-models-pytorch 0.3.3, NumPy 1.26.4 and PyBullet 3.2.7. It was recovered rather than reinstalled or revalidated. The old upstream environment instructions remain historical; they are not the recovered run's environment specification.

The second package also supplies the DINOv3 training Hydra config/overrides: simulation/gripper, 128 px, radius 20, batch size 2, four workers, 50 epochs, one GPU, precision 16 and no distributed strategy. It uses historical environment variables `DINOV3_REPO` and `DINOV3_WEIGHTS`; current `_DIR`/`_PATH` names do not resolve those old interpolations automatically. The console identifies the best checkpoint save at epoch 28, step 1,936,155, with `val_miou=0.72213`. No training-seed override is recorded despite the directory label `gripper_seed0`.

### Recovered perception evaluator

The reviewed script is preserved verbatim at [scripts/eval_affordance_dinov3_vs_baseline.py](../scripts/eval_affordance_dinov3_vs_baseline.py), and its SHA-256 matches the private archive. The supplied historical command is documented here for provenance only:

```bash
python eval_affordance_dinov3_vs_baseline.py --warmup 5 --amp
```

The original script was at the repository root; its reviewed copy here is under `scripts/`. No evaluator is executed by this workflow. The source selects the validation split, checks paired sample ordering, uses `shuffle=False`, and evaluates each model at its native preprocessing settings. The command leaves checkpoint/config options at their recorded defaults; the historical training config's environment-variable requirements are noted above. The text record is not independent proof of execution with these flags. Models use eval/inference mode and optional FP16 autocast. CUDA synchronization brackets the forward pass measured by `perf_counter`; encoder, decoder, segmentation and direction heads are included, while loading, device transfer, argmax, Hough voting and SAC are excluded. Baseline always runs before DINOv3. cuDNN benchmarking is enabled; no general determinism guarantee follows from `torch.manual_seed(0)`.

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
python -m pytest -p no:cacheprovider tests/test_experiment_identity_audit.py -q
```

Tests use small synthetic arithmetic fixtures to check calculations, undefined values, duplicate/unpaired records, median conventions, paired cluster resampling, and export privacy. These fixtures validate the analysis software and make no claim about scientific model performance.

## Unresolved evidence boundaries

| Experiment | Available | Unresolved evidence |
| --- | --- | --- |
| Perception | Evaluator source, numerical records, exact ordered split IDs, manifest/weight digests, training config, AMP/device metadata, reported `--warmup 5 --amp` command and timing protocol | Raw data content/release identity; contemporaneous command trace; historical loaded bytes and DINOv3 code revision |
| Robotics | All 200 logs/statuses, 400 Hydra files, runners/parser, environment snapshot, SAC configs, recorded weight digests, runtime commit and source seed propagation | Historical weight load-time attestation; complete runtime state; all-RNG and physical-state pairing evidence |
| Training | Saved DINOv3 config/overrides, split manifests/cardinalities and checkpoint-selection log at epoch 28 | Explicit training seed; immutable data/environment identity; proof that historical loaded weights equal the later recorded artifact |

The existing evaluations were performed by the researcher. No new model training, inference, simulator rollout or weight download is performed or required to finish this documentation. These boundaries remain explicit rather than being filled by assumed facts.

According to the supplied search record, real comparative images were not found in `reproduction/`, the project under the searched filename patterns, or `/tmp/vapo_ab`. No inference images are generated to replace missing historical examples; existing quantitative figures derive only from the saved metric records.
