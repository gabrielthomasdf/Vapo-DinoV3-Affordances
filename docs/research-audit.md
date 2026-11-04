# Research and repository audit

Updated: 2026-11-04. Scope: the four recovered metric files, the supplementary provenance archive, repository history through `8396d6f`, and local historical logs. This supersedes the initial 2026-11-03 statement that the comparison CSVs were unavailable. Numerical results are now **checked against recovered artifacts**, including all robotic run logs; full training, inference and robotic evaluation have not been reproduced.

## Recovered-file audit

The four files were inspected in a private directory outside the repository. No credentials or personal contact information were observed. The segmentation CSV includes dataset/sample path identifiers; the robotic CSV contains object and seed identifiers. No license or redistribution grant accompanies the recovered files. They remain outside Git; only scripts, aggregates, source hashes and derived charts are prepared for publication. No datasets or weights were copied.

| File | Bytes | Schema and outcome |
| --- | ---: | --- |
| `metricas_por_imagem.csv` | 1,801,955 | 23 columns; 6,386 paired records; unique sample IDs; contiguous unique indices 0–6,385; no empty fields |
| `resumo.json` | 1,761 | Baseline, DINOv3 and comparison sections; counts, global metrics and means agree with recalculation; two median convention differences retained |
| `ab_results.csv` | 107,334 | 7 columns; 3,000 records; baseline and DINOv3 labels; 100 seed labels each; 15 objects per seed and approach |
| `git_commit.txt` | 41 | `bbded785430e5cded88ba57aa8ca62458ef038ce`; resolves in this history as “Add opt-in DINOv3 affordance evaluation” |

Exact SHA-256 hashes and all derived numbers are in [analysis_summary.json](experiments/recovered/analysis_summary.json). The standalone commit file records an ancestor that predates the evaluation seed override. The supplementary package now resolves the robotic code version: its batch metadata and all 200 run logs record `4caeecf338c9a758607f1737adf222150806f9cb`, and its archived evaluator matches that Git version after newline normalization. Older `bbded78` captures remain in two other commit files; they do not describe the final repeated-evaluation code. Checkpoint content hashes remain absent.

### Supplementary provenance package

`ic_vapo_proveniencia_20261104.tar.gz` is 163,835 bytes, SHA-256 `b04e3eacfb99538b5535a4f658da82ecf1169e868ff9a593927e71c0d8567441`. It contains the perception evaluator, robotic evaluator, two shell runners, CSV parser, DINOv3 config, 200 individual run logs, status tables, two resolved-config captures, environment/GPU metadata, and a nested archive with 200 Hydra configs plus 200 overrides. Its copy of `ab_results.csv` is byte-identical to the previously audited original.

The verifier in `scripts/audit_recovered_provenance.py` reads the archive in memory and uses safe YAML parsing. It does not extract or execute archived code. All 3,000 parsed episode tuples, including return and step count, match the CSV in order. Every run has 15 episode records, a matching final success summary and exit status zero. Every requested seed 0–99 matches its Hydra `test.eval_seed` and override; excluding that field, composed configs are constant within each approach. [provenance_audit.json](experiments/recovered/provenance_audit.json) records hashes and checks without internal directory paths.

The reviewed perception script contains no private absolute paths and is copied verbatim to `scripts/eval_affordance_dinov3_vs_baseline.py` (SHA-256 `41bbd3bdbdb40c01fc20db0675d4c65d3480a677334867268c249e39bdc3de55`). Shell runners, logs and full configs contain internal account paths and remain private; only curated references, package versions and aggregate findings are exported.

Both resolved captures reference the same policy run ending in `trained_models/policy/tabletop/vapo` and `most_tasks_from_15.pth`. Baseline gripper affordance refers to `gripper_tabletop.ckpt`; DINOv3 refers to `reproduction/dinov3_affordance/gripper_seed0/trained_models/best_val_miou.ckpt`; the static reference is `static_tabletop.ckpt`. These identify intended inputs, not immutable file contents. The 400 Hydra files describe CLI composition before the saved-policy merge; they are not final runtime-config dumps. The first package lacked that saved policy config; the second package supplies it, as described below.

The recovered environment snapshot records Python 3.11.15, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, PyTorch Lightning 1.5.9, segmentation-models-pytorch 0.3.3, NumPy 1.26.4 and PyBullet 3.2.7. This is a recovered snapshot, not a newly installed/tested environment. The GPU inventory includes RTX A5500 devices; runners select visible device 3. The perception CSV's device name is independently recorded in its JSON.

### Saved-policy and training-config package

`ic_vapo_pendencias_20261104.tar.gz` is 19,889,473 bytes, SHA-256 `439b16ceb6ea8bac6704c0be24e18e1e9d6ddca2f79186fe569258d64cab849e`. Its nine files include saved `vapo` and `baseline` policy configs/overrides, DINOv3 training config/overrides and logs, and the same perception evaluator source. The 395,312,256-byte training console is streamed in bounded chunks and is never copied into public output. Its SHA-256 is recorded in the supplementary JSON report. No checkpoint binaries, checkpoint-hash manifest or dataset/split manifest are present; no 64-character hexadecimal hash or evaluation-command/warmup keyword was found in the training console.

The selected saved policy is **`vapo/.hydra/config.yaml`**, based on the common policy-run reference in both A/B captures. It enables gripper affordance inputs, distance and reward densification and uses 64-pixel observations, hidden dimension 256, latent dimension 16, four layers, `CNNPolicyDenseNet` and `CNNCriticDenseNet`. Its saved environment has `use_egl=false`, matching the requested flag and resolving the earlier configuration uncertainty about EGL. Its seed is null in the training snapshot; evaluation overrides it before environment construction. The separate policy config named `baseline` disables the three affordance features and is not selected by these A/B runners. Consequently the A/B label “baseline” means original ResNet18 VAPO, not the no-affordance ablation.

The recovered DINOv3 config uses the simulation dataset, gripper camera, 128-pixel inputs, radius 20, 50 epochs, batch size 2, four loader workers, one GPU, precision 16 and no distributed strategy. The backbone is frozen and input normalization is enabled. Training overrides confirm these values, but contain no explicit training-seed assignment. The folder name `gripper_seed0` is not sufficient evidence of a training seed.

The log records 133,528 training samples and 6,386 validation samples. At epoch 28, global step 1,936,155, the console records `val_miou=0.72213` and saves `trained_models/best_val_miou.ckpt`. This agrees to the displayed precision with the independently reconstructed empty-union-zero batch mean of 0.722132. It strengthens the metric-convention explanation and provides a checkpoint selection record; without content hashes it does not prove binary identity of the evaluated weights. The historical config uses `DINOV3_REPO` and `DINOV3_WEIGHTS`, whereas current integration configs use `DINOV3_REPO_DIR` and `DINOV3_WEIGHTS_PATH`; recreation must supply the variables expected by the historical config.

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

Sample IDs identify 2,108 kitchen records, 2,723 tools records and 1,555 miscellaneous records, in three scene/trajectory groups. Counts and sample IDs alone do not establish the precise dataset release or independent frame sampling. The original split manifest is still required.

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

Recovered metadata: NVIDIA RTX A5500, AMP enabled, `native_preprocessing`, baseline input/radius 64/10 and DINOv3 128/20. The native grid and label radius differences mean this is a comparison of two recorded setups, not an isolated encoder intervention. Source review now establishes synchronized `perf_counter` timing around the forward pass, batch size 1, inference mode, optional FP16 autocast and fixed baseline-then-DINOv3 ordering. It includes encoder, decoder, segmentation and direction heads; it excludes loading, device transfer, argmax, Hough voting and SAC. Warmup defaults to 20 iterations; the exact historical invocation/overrides are absent. Timings remain descriptive and no controlled latency chart is generated.

### Why training val_miou differs

The recovered binary mIoU of 0.819045 averages foreground and background IoUs after summing all counts. The function named `compute_mIoU` in `vapo/affordance/utils/losses.py` computes **foreground-only** intersection/union over a batch. The recovered summary's batch-of-two foreground mean is 0.728521 after excluding 28 empty-union DINOv3 batches. Applying the training function's empty-union value of zero gives **0.722132**, close to the earlier logged validation values.

This resolves a substantive metric/aggregation/convention difference. It does not establish identical training/evaluation checkpoints or dataset releases. The original training logger excludes its final batch from the console average. The recovered evaluator confirms validation split selection, `shuffle=False`, explicit equality of paired dataset sample order, and reconstruction of IoU batches from that order while excluding undefined unions. The separate historical DINOv3 log confirms 133,528 training samples, 6,386 validation samples and 50 epochs, but its exact checkpoint cannot yet be mapped to this evaluation by content hash.

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

## Verification

The metric analysis was run against all four originals on CPU, generating three PNG/SVG figures, four aggregate CSVs and a strict JSON report. A second execution of that workflow produced byte-identical copies of all 11 outputs. The supplementary provenance verifier adds a separate JSON audit of both packages. The calculation tests cover global versus per-image aggregation, empty unions, malformed/duplicate/unpaired records, finite values, median conventions, conditional block resampling, source hashes and export privacy. There are 17 metric-analysis tests and eight provenance-audit tests; the complete `tests` directory returned **27 passed, 7 skipped**. The skipped modules require unavailable research dependencies; no full scientific evaluation or training was rerun.

The original README is retained at `docs/upstream-readme.md`. This audit does not change model behavior or experiment history. Simulator meshes/textures remain intact. The earlier hypothesis that vendored GLAD C sources explain GitHub's language classification remains unverified against GitHub Linguist.

## Remaining work

1. Recover checkpoint content hashes, the dataset/split manifest and exact perception invocation/warmup settings. Saved policy configs and the DINOv3 training config/overrides are now recovered, alongside evaluators, runners, parser, per-seed CLI configs and environment snapshot.
2. Verify all RNG streams, reset semantics and paired physical initial states. Requested seeds and environment-seed propagation are evidenced; the same SAC path/name is referenced, but checkpoint identity still needs hashes.
3. Confirm actual perception timing invocation/warmup settings before a controlled latency claim. Timing scope, CUDA synchronization, fixed ordering and batch size are now known from source.
4. Recover annotated center experiments and paired image/prediction examples, with permission and licenses for public reuse.
5. Reproduce model inference and robotic evaluation in the compatible Linux/CUDA environment when those inputs are available; compare matched settings before causal claims.
6. Complete GitHub description/topics and verify its language breakdown. No push is included in this consolidation.
