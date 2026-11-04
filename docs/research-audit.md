# Research and repository audit

Updated: 2026-11-04. Scope: the four metric files, both provenance packages, the recovered checkpoint-hash manifest, four split manifests, the supplied historical command, and repository history through `16ee17e`. The researcher already performed the experiments. This work audits their saved evidence and recalculates numerical records; no additional training, inference or robotic evaluation is executed or required to complete this documentation.

## Recovered-file audit

The four files were inspected in a private directory outside the repository. No credentials or personal contact information were observed. The segmentation CSV includes dataset/sample path identifiers; the robotic CSV contains object and seed identifiers. No license or redistribution grant accompanies the recovered files. They remain outside Git; only scripts, aggregates, source hashes and derived charts are prepared for publication. No datasets or weights were copied.

| File | Bytes | Schema and outcome |
| --- | ---: | --- |
| `metricas_por_imagem.csv` | 1,801,955 | 23 columns; 6,386 paired records; unique sample IDs; contiguous unique indices 0–6,385; no empty fields |
| `resumo.json` | 1,761 | Baseline, DINOv3 and comparison sections; counts, global metrics and means agree with recalculation; two median convention differences retained |
| `ab_results.csv` | 107,334 | 7 columns; 3,000 records; baseline and DINOv3 labels; 100 seed labels each; 15 objects per seed and approach |
| `git_commit.txt` | 41 | `bbded785430e5cded88ba57aa8ca62458ef038ce`; resolves in this history as “Add opt-in DINOv3 affordance evaluation” |

Exact source hashes and derived numbers are in [analysis_summary.json](experiments/recovered/analysis_summary.json). The standalone commit file records an ancestor that predates the evaluation seed override. Batch metadata and all 200 run logs record `4caeecf338c9a758607f1737adf222150806f9cb`, and the archived evaluator matches that Git version after newline normalization. Older `bbded78` captures remain in two other commit files; they do not describe the final repeated-evaluation code. The later checkpoint-hash manifest supplies six weight identities, documented below with their evidence limits.

### Supplementary provenance package

`ic_vapo_proveniencia_20261104.tar.gz` is 163,835 bytes, SHA-256 `b04e3eacfb99538b5535a4f658da82ecf1169e868ff9a593927e71c0d8567441`. It contains the perception evaluator, robotic evaluator, two shell runners, CSV parser, DINOv3 config, 200 individual run logs, status tables, two resolved-config captures, environment/GPU metadata, and a nested archive with 200 Hydra configs plus 200 overrides. Its copy of `ab_results.csv` is byte-identical to the previously audited original.

The verifier in `scripts/audit_recovered_provenance.py` reads the archive in memory and uses safe YAML parsing. It does not extract or execute archived code. All 3,000 parsed episode tuples, including return and step count, match the CSV in order. Every run has 15 episode records, a matching final success summary and exit status zero. Every requested seed 0–99 matches its Hydra `test.eval_seed` and override; excluding that field, composed configs are constant within each approach. [provenance_audit.json](experiments/recovered/provenance_audit.json) records hashes and checks without internal directory paths.

The reviewed perception script contains no private absolute paths and is copied verbatim to `scripts/eval_affordance_dinov3_vs_baseline.py` (SHA-256 `41bbd3bdbdb40c01fc20db0675d4c65d3480a677334867268c249e39bdc3de55`). Shell runners, logs and full configs contain internal account paths and remain private; only curated references, package versions and aggregate findings are exported.

Both resolved captures reference the same policy run ending in `trained_models/policy/tabletop/vapo` and `most_tasks_from_15.pth`. Baseline gripper affordance refers to `gripper_tabletop.ckpt`; DINOv3 refers to `reproduction/dinov3_affordance/gripper_seed0/trained_models/best_val_miou.ckpt`; the static reference is `static_tabletop.ckpt`. These identify intended inputs, not immutable file contents. The 400 Hydra files describe CLI composition before the saved-policy merge; they are not final runtime-config dumps. The first package lacked that saved policy config; the second package supplies it, as described below.

The recovered environment snapshot records Python 3.11.15, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, PyTorch Lightning 1.5.9, segmentation-models-pytorch 0.3.3, NumPy 1.26.4 and PyBullet 3.2.7. This is a recovered snapshot, not a newly installed/tested environment. The GPU inventory includes RTX A5500 devices; runners select visible device 3. The perception CSV's device name is independently recorded in its JSON.

### Saved-policy and training-config package

`ic_vapo_pendencias_20261104.tar.gz` is 19,889,473 bytes, SHA-256 `439b16ceb6ea8bac6704c0be24e18e1e9d6ddca2f79186fe569258d64cab849e`. Its nine files include saved `vapo` and `baseline` policy configs/overrides, DINOv3 training config/overrides and logs, and the same perception evaluator source. The 395,312,256-byte training console is streamed in bounded chunks and never copied into public output. Its SHA-256 is recorded in the supplementary JSON report. These two archives do not include weights or identity/split manifests; the later files described below supply those manifests. No 64-character hexadecimal hash or evaluation-command/warmup keyword was found in this training console.

The selected saved policy is **`vapo/.hydra/config.yaml`**, based on the common policy-run reference in both A/B captures. It enables gripper affordance inputs, distance and reward densification and uses 64-pixel observations, hidden dimension 256, latent dimension 16, four layers, `CNNPolicyDenseNet` and `CNNCriticDenseNet`. Its saved environment has `use_egl=false`, matching the requested flag and resolving the earlier configuration uncertainty about EGL. Its seed is null in the training snapshot; evaluation overrides it before environment construction. The separate policy config named `baseline` disables the three affordance features and is not selected by these A/B runners. Consequently the A/B label “baseline” means original ResNet18 VAPO, not the no-affordance ablation.

The recovered DINOv3 config uses the simulation dataset, gripper camera, 128-pixel inputs, radius 20, 50 epochs, batch size 2, four loader workers, one GPU, precision 16 and no distributed strategy. The backbone is frozen and input normalization is enabled. Training overrides confirm these values, but contain no explicit training-seed assignment. The folder name `gripper_seed0` is not sufficient evidence of a training seed.

The log records 133,528 training samples and 6,386 validation samples. At epoch 28, global step 1,936,155, the console records `val_miou=0.72213` and saves `trained_models/best_val_miou.ckpt`. This agrees to the displayed precision with the independently reconstructed empty-union-zero batch mean of 0.722132. It strengthens the metric-convention explanation and provides a checkpoint selection record. The later manifest supplies a weight identity, without independently attesting historical loaded bytes. The historical config uses `DINOV3_REPO` and `DINOV3_WEIGHTS`, whereas current integration configs use `DINOV3_REPO_DIR` and `DINOV3_WEIGHTS_PATH`.

### Checkpoint identities, validation partitions and historical command

[identity_audit.json](experiments/recovered/identity_audit.json) records source hashes for the new evidence and links them to the same metric CSVs. `scripts/audit_experiment_identity.py` performs only saved-artifact checks. Private absolute paths from the supplied checkpoint manifest are replaced by logical repository-relative references in [checkpoints.sha256](experiments/recovered/checkpoints.sha256).

| Weight artifact | Recorded SHA-256 |
| --- | --- |
| ResNet18 gripper affordance, `gripper_tabletop.ckpt` | `0784ac06b828c6ce1fc880cd018a1f5fb34921bcad0b7e21add28fc4a337e1fa` |
| Static affordance, `static_tabletop.ckpt` | `5fa2e6f32ab3bdecfc2a8ee6a89724f26dc365bc5c1a0b041db705006a899a5b` |
| DINOv3 gripper, `gripper_seed0/trained_models/best_val_miou.ckpt` | `f1ac08ebe5eff5df150746e028cf4d1e1981e1cda5f54cd5cead59d2f65ade45` |
| Shared SAC VAPO policy, `vapo/trained_models/most_tasks_from_15.pth` | `586ca6bfcc0a4cf4e134f28901341d351b3808097b4e6900a50a796c6e01dc3f` |
| Separate no-affordance policy, `baseline/trained_models/most_tasks_from_15.pth` | `a0c9bb6f15872034f22a618714eb702c2f18bf9f3d52178a738c53e5af8cc46e` |
| DINOv3 backbone, `dinov3_vits16plus_pretrain_lvd1689m.pth` | `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea` |

These are **identities recorded in the recovered SHA-256 manifest**. The logical references agree with the recovered code/configuration. The weight files were not rehashed locally in this consolidation, and no contemporaneous load-time hashes independently certify the exact bytes used historically. The separately named no-affordance policy is documented for disambiguation and was not selected by the A/B references. The static checkpoint is a retained setup reference; a hash alone does not demonstrate that it influenced a rollout.

The dataset archive has SHA-256 `4d1ef5f15a2cef576099619240f0e078fb5e87d6d72ecf84db5e898f9c4490b9`. All four `episodes_split.json` files were reconstructed using `VREnvData._get_split_data` semantics: preserve episode insertion order, sort filenames inside each episode, and filter the gripper camera.

| Manifest | Training gripper samples | Validation gripper samples | Verified relationship |
| --- | ---: | ---: | --- |
| Consolidated `episodes_split.json` | 133,528 | 6,386 | Exact sample IDs and ordering of the evaluated CSV |
| `tabletop_kitchen/episodes_split.json` | 44,733 | 2,108 | Exact ordered kitchen subset |
| `tabletop_misc/episodes_split.json` | 42,120 | 1,555 | Exact ordered miscellaneous subset |
| `tabletop_tools/episodes_split.json` | 46,675 | 2,723 | Exact ordered tools subset |

The scene manifests form the consolidated training and validation sets with no missing/extra gripper IDs, duplicate IDs or training/validation overlap. The ordered validation-list fingerprint is `865a92fe8f1e9a58c57c719844d573dcb6c8149110619c7d4ad215073abc4a54` (UTF-8 IDs joined by newline plus a final newline). All four manifest byte hashes are in the identity report. Thus **the manifests correspond to the evaluated partition at sample identity/order level**. This does not attest raw image/label contents, an immutable dataset release, or which physical JSON file was read during the historical process. Matching the training count to the training log does not independently attest every historical training input.

The supplied text records:

```bash
python eval_affordance_dinov3_vs_baseline.py --warmup 5 --amp
```

The recovered parser supports both arguments. `--warmup 5` overrides its default of 20; the warmup loop executes the requested number of iterations for both models. `--amp` agrees with `resumo.json` and enables autocast FP16. Checkpoint/config options remain at the script's recovered defaults, and no `--max-samples` restriction is present. This is a **reported historical command checked for compatibility with code and records**. The new text file is not a contemporaneous terminal history, process capture or evaluation log, so actual execution with five warmup iterations is not independently attested. No fresh latency or model evaluation was run to fill that gap.

### Segmentation schema and integrity

`index` orders the records and `sample` identifies a paired input. Each approach has `positive_pixels_gt`, TP/FP/FN/TN, foreground IoU, Dice/F1, precision, recall and latency in milliseconds; the final column records the foreground-IoU difference.

All confusion counts are nonnegative integers. Every baseline row totals 4,096 pixels (64²); every DINOv3 row totals 16,384 (128²), consistent with the summary's native input sizes. All rows satisfy `positive_pixels_gt = TP + FN`. Derived metric columns and defined deltas agree with their counts; timings are finite and positive. There are no duplicate samples, missing indices or malformed rows. This establishes internal completeness, not completeness against an unavailable evaluation manifest.

There are 707 numeric `NaN` cells, all explained by zero denominators:

| Undefined quantity | Baseline | DINOv3 |
| --- | ---: | ---: |
| Foreground IoU | 2 | 72 |
| Foreground Dice | 2 | 72 |
| Precision | 22 | 136 |
| Recall | 164 | 164 |

There are also 73 undefined per-image deltas where at least one foreground IoU is undefined. These values are reported explicitly and excluded only from the corresponding defined-value means; confusion counts from every image still contribute to global metrics.

Sample IDs identify 2,108 kitchen records, 2,723 tools records and 1,555 miscellaneous records in three scene/trajectory groups. These now match the four recovered manifests exactly as documented above. Dataset release identity, raw contents and independent frame sampling are not established by the lists alone.

### Recalculated perception results

| Metric | Baseline | DINOv3 |
| --- | ---: | ---: |
| Foreground global IoU | 0.498748 | 0.672048 |
| Background global IoU | 0.951221 | 0.966042 |
| Binary global mIoU | 0.724985 | 0.819045 |
| Foreground global Dice | 0.665553 | 0.803862 |
| Global precision | 0.815155 | 0.805920 |
| Global recall | 0.562348 | 0.801815 |
| Global accuracy | 0.953479 | 0.968251 |
| Mean defined per-image foreground IoU | 0.523913 | 0.733339 |
| Mean defined batch-of-two foreground IoU | 0.522834 | 0.728521 |
| Batch mean with empty union assigned zero | 0.522834 | 0.722132 |
| Recorded mean latency, ms | 3.682468 | 24.412255 |
| Conventional median latency, ms | 3.320898 | 23.378070 |

The previous ~22 ms DINOv3 estimate is superseded by the CSV mean of **24.412255 ms**. The JSON stores medians of 3.320842981 and 23.377534866 ms. The recovered evaluator explicitly uses `torch.tensor(latencies).median()`, explaining the lower-middle float32 convention. The exported report retains the two discrepancies against the conventional median instead of hiding them. Counts, global scores, mean latencies, defined-image/batch means and comparison deltas otherwise match at the documented tolerance.

Recovered metadata: NVIDIA RTX A5500, AMP enabled, `native_preprocessing`, baseline input/radius 64/10 and DINOv3 128/20. The native grid and label radius differences mean this is a comparison of two recorded setups, not an isolated encoder intervention. Source review establishes synchronized `perf_counter` timing around the forward pass, batch size 1, inference mode, optional FP16 autocast and fixed baseline-then-DINOv3 ordering. It includes encoder, decoder, segmentation and direction heads; it excludes loading, device transfer, argmax, Hough voting and SAC. The supplied historical command specifies warmup 5 and AMP, with the evidence limit described above. Timings remain descriptive and no controlled latency chart is generated.

### Why training val_miou differs

The recovered binary mIoU of 0.819045 averages foreground and background IoUs after summing all counts. The function named `compute_mIoU` in `vapo/affordance/utils/losses.py` computes **foreground-only** intersection/union over a batch. The recovered summary's batch-of-two foreground mean is 0.728521 after excluding 28 empty-union DINOv3 batches. Applying the training function's empty-union value of zero gives **0.722132**, close to the earlier logged validation values.

This resolves a substantive metric/aggregation/convention difference. It does not establish identical dataset releases or runtime-loaded bytes. The training logger excludes its final batch from the console average. The recovered evaluator confirms validation split selection, `shuffle=False`, explicit equality of paired dataset sample order, and reconstruction of IoU batches from that order while excluding undefined unions. The training log confirms 133,528 training samples, 6,386 validation samples and 50 epochs; the recorded DINOv3 checkpoint hash now identifies the supplied artifact reference without independently attesting the original load operation.

### Robotic schema and integrity

`model`, `seed`, `episode_index`, `object`, `success`, `return` and `steps` are present on every row. Success flags are 0/1, returns are finite and step counts are positive integers. There are no duplicated rows or duplicated composite keys `(model, seed, object, episode_index)`. Episode indices 0–4 restart inside each seed; `(model, seed, episode_index)` alone is not a unique episode key.

Both approaches have the same 1,500 seed/object/index keys: seed labels 0–99, each with the same 15 object/index pairs. No cross-approach missing keys or imbalanced object counts were found. A missing record in both approaches could not be detected without the experiment plan. No scene/task column or independent training-repetition ID is supplied.

| Approach | Successes | Episodes | Success rate |
| --- | ---: | ---: | ---: |
| Baseline | 1,293 | 1,500 | 86.20% |
| DINOv3 | 937 | 1,500 | 62.47% |

The difference is **−23.73 percentage points** for DINOv3. All objects have 100 episodes per approach:

| Object | Baseline successes | DINOv3 successes | Difference, pp |
| --- | ---: | ---: | ---: |
| bowl | 100 | 95 | −5 |
| drill_1 | 95 | 97 | +2 |
| drill_2 | 100 | 99 | −1 |
| frying_pan_1 | 100 | 8 | −92 |
| frying_pan_2 | 0 | 1 | +1 |
| frying_pan_3 | 100 | 32 | −68 |
| hammer_1 | 100 | 100 | 0 |
| hammer_2 | 87 | 3 | −84 |
| hammer_3 | 45 | 0 | −45 |
| knife | 100 | 100 | 0 |
| marker | 98 | 100 | +2 |
| small_tool_1 | 100 | 100 | 0 |
| small_tool_2 | 84 | 85 | +1 |
| small_tool_3 | 97 | 29 | −68 |
| spatula | 87 | 88 | +1 |

The repository's pickup evaluator reads `info["success"]`. In `PlayTableRL`, a successful pickup triggers moving/dropping the target into the box and checking its XY position against the box bounds (±0.12 m in X and ±0.2 m in Y; lower bounds exclusive, upper inclusive). The recovered parser copies the logged Boolean `Success` into CSV values 0/1, and the archived evaluator/logged commit trace back to this evaluation code. No saved simulator trajectories are available to independently rejudge each physical success.

No confidence interval is claimed from independent episodes. The runner launches a new process per seed, and the archived evaluator copies `test.eval_seed` to `run_cfg.env.seed` before construction. `PlayTableSimEnv` initializes its scene RNG using that seed. This supports intended environment seeding, but does not prove identical physical initial states or control of every RNG: the code also uses global NumPy/Python random sources. Shared seed/object/policy structure can create dependence, and seeds are not independently trained policies. [The analysis documentation](experimental-reproduction.md#statistical-uncertainty) provides an explicitly conditional paired seed-block bootstrap for use only after its assumptions are justified.

## Contributions visible in Git

The initial fork comparison was against `a0b9444`, the last upstream commit visible in the local history. No fetched upstream ref was present, so this is not a comparison against the upstream's latest online state. The original implementation changes comprise 34 paths through `4caeecf`; subsequent commits added portfolio documentation.

| Area | Evidence |
| --- | --- |
| Encoder integration | `vapo/affordance/Dinov3Adapter.py`, selectable encoders in `affordance_model.py` and compatibility tests |
| Configuration/preprocessing | DINOv3 gripper config, RGB transform YAML, opt-in tabletop config; 128-pixel gripper affordance input, 64-pixel SAC observation |
| Pipeline compatibility | Shape handling in `aff_wrapper_base.py`; preservation of selected affordance config in `load_eval_cfg` |
| Diagnostics | Observation capture, offline replay, same-tensor inference checks; simulation/real-world entry points |
| Evaluation repeatability | Explicit evaluation seed override and configuration tests |
| Portability | Hough build, path, transform and loss-device fixes |
| Artifact analysis | `scripts/analyze_recovered_results.py`, calculation tests, aggregate tables and derived figures |

Git records work on the fork; exact individual contribution credit should follow the actual collaboration record. VAPO's original MIT notice, author credits and original README are preserved. DINOv3 remains subject to its separate license; no weights are included.

## Centers of affordance: secondary investigation

No numerical center-error experiment was found among the four recovered files or the inspected results folders. The CSVs contain no predicted/ground-truth center coordinates, matching criterion, center distance or center-specific success threshold.

Existing code captures Hough center counts, 2D centers, projected depth pixels, world coordinates, finite-value/shape checks, successful world conversions, and runtime versus replay consistency. `simulation_diagnostics.py` also reports robot-to-target distance after positioning. These are diagnostic measurements; they are not an annotated center-localization benchmark.

Local historical output folders `19-33-21/static_centers` and `19-46-10/static_centers` contain visual center overlays. They were left private; there is no accompanying center accuracy table or ground-truth matching record. The code motivates hypotheses about coordinate conversion, mask resizing, target selection and policy input shift. The recovered records do not test those hypotheses or establish the cause of reduced robotic success.

The supplied search record reports no real comparative segmentation examples in `reproduction/`, the project under the searched filename patterns, or `/tmp/vapo_ab`. This is a report of the searches performed, not a claim that every possible location was searched. Existing center overlays do not establish paired baseline/DINOv3 inference comparisons. Only numerical charts derived from saved records and an illustrative architecture diagram are presented; no inference examples or demonstrations are fabricated.

## Verification

The metric analysis reads all four originals on CPU, producing three PNG/SVG numerical figures, four aggregate CSVs and a strict JSON report. The earlier numerical workflow produced byte-identical copies of all 11 outputs on repetition. Separate identity and provenance audits now link both packages, six recorded checkpoint hashes, four split manifests and the supplied command. Tests cover aggregation, empty unions, malformed/duplicate/unpaired records, paired block resampling, privacy, ordered split membership, checkpoint-role disambiguation, static command parsing and cross-report source hashes. There are 17 metric-analysis tests, ten provenance tests and nine identity tests; with two existing build tests the complete `tests` directory returned **38 passed, 7 skipped**. The skipped modules require unavailable research dependencies. No model code, training, inference or simulation is executed by these artifact audits.

The original README is retained at `docs/upstream-readme.md`. This audit does not change model behavior or experiment history. Simulator meshes/textures remain intact. The researcher's completed experiments are not a pending rerun task; this consolidation performs only artifact inspection, arithmetic recalculation and software tests.

## Repository metadata

The GitHub description was updated and verified on 2026-11-04: “Research fork of VAPO exploring DINOv3 visual affordances for robotic manipulation. UNICAMP undergraduate research with audited perception and policy-evaluation artifacts.” Topics are `affordance-learning`, `computer-vision`, `dinov3`, `machine-learning`, `pytorch`, `reinforcement-learning`, `reproducibility`, `research`, `robotic-manipulation`, `robotics` and `vapo`.

Before this attribute was published, the GitHub languages API reported C 621,377 bytes, Python 600,657, C++ 13,318, CUDA 11,909, Dockerfile 2,388 and Shell 702. The unchanged upstream GLAD bindings occupy nine files totaling 633,208 bytes. `.gitattributes` identifies only `VREnv/egl_check/glad/**` as third-party vendored code; local `git check-attr` confirms the rule and leaves the fork's model code unmarked. This is a language-statistics correction using [GitHub Linguist's documented vendored-code override](https://github.com/github-linguist/linguist/blob/main/docs/overrides.md#vendored-code). GitHub refreshes the statistics after publication; the final displayed breakdown depends on that asynchronous recalculation.

## Unresolved provenance limits

1. Recorded weight identities are available, but weight binaries were not locally rehashed and historical load-time contents are not independently attested.
2. The exact validation IDs/order match all four manifests; raw image/label content hashes and a dataset release identifier remain unavailable.
3. The supplied `--warmup 5 --amp` command agrees with code and metadata but lacks a contemporaneous execution trace.
4. Explicit training-seed assignment, all RNG streams and paired physical initial states are not established by the recovered evidence.
5. Comparative real images and quantitative center-localization results are unavailable in the supplied search record. No examples are invented.

These are documented evidence boundaries, not requirements to repeat training or evaluation.
