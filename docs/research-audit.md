# Research and repository audit

Audit date: 2026-11-03. Scope: local fork at `4caeecf`, compared with `a0b9444`, the last upstream commit visible in this repository's history. The `upstream` remote exists, but no fetched upstream ref is present; this is a comparison with the recorded fork point, not a claim about the upstream's latest state. The working tree was clean before documentation changes.

## Contributions visible in Git

| Area | Evidence in this fork |
| --- | --- |
| Encoder integration | `vapo/affordance/Dinov3Adapter.py`; changes to `vapo/affordance/affordance_model.py`; selectable encoder tests |
| Configuration and preprocessing | DINOv3 gripper model and RGB transform YAMLs; `cfg_tabletop_dinov3.yaml`; 128-pixel gripper affordance input and 64-pixel SAC observation |
| Pipeline compatibility | `vapo/wrappers/affordance/aff_wrapper_base.py`; affordance shape tests; evaluation config preservation in `vapo/utils/utils.py` |
| Diagnostics | Observation capture and replay modules, simulation and real-world validation scripts, related tests |
| Evaluation repeatability | `test.eval_seed` override in `scripts/eval_tabletop.py`; policy config loading tests |
| Portability | Hough build, loss device, path, and dataset transform fixes; associated tests |

This is 34 changed paths relative to `a0b9444`. The history identifies these as work on the fork, but Git authorship by itself does not prove that every line was written by one person; credit should follow the actual collaboration record.

## Evidence inventory

| Claim | Evidence found | Remaining gap |
| --- | --- | --- |
| 6,386 validation images | Local `Faculdade/IC/Resultados/Dino/train_affordancedino.log`, line with `val_data 6386` | Dataset version and split manifest |
| Binary mIoU 0.7250 vs 0.8190; Dice 0.6656 vs 0.8039; recall 0.5623 vs 0.8018 | Researcher handoff only. The local DINOv3 training log contains per-epoch `val_miou`, generally near 0.69–0.72, with no matching 0.8190 entry. | Evaluation script, metric definitions, prediction outputs, baseline/DINOv3 checkpoints and run configurations |
| ~3.68 vs ~22 ms inference | Researcher handoff only | Benchmark script, device, batch size, warmup and timing protocol |
| 1,293/1,500 vs 937/1,500 successes | Researcher handoff only. Local training logs are **not** the 1,500-episode evaluation record. | Per-episode CSV/log, scene list, evaluation commands, policy and affordance checkpoint hashes, seeds |
| DINOv3 training schedule | Local log records 50 epochs, 133,528 training samples, 6,386 validation samples, `batch_size=2`, and an older config using `DINOV3_REPO`/`DINOV3_WEIGHTS`. | Exact Git revision/config snapshot and resulting checkpoint mapping; current config uses `DINOV3_REPO_DIR`/`DINOV3_WEIGHTS_PATH` |

The `Resultados` and policy checkpoint directories are outside this repository. They were inspected read-only and were not copied into the public project. The supplied summary is the source of all comparison figures in the new README. The discrepancy in `val_miou` requires reconciliation before the perception table can be described as independently verified.

## Reproduction and environment gaps

- The repository has configuration and model tests, but no checked-in script or structured results for the full perception comparison or 1,500-episode robotic comparison.
- The DINOv3 model code and weights come from Meta under separate terms; their precise revision and weight hash are absent. The fork's `LICENSE` is the original VAPO MIT notice, not a license for those external assets.
- The upstream installation instructions target older Linux/CUDA/PyTorch versions. `environment.yml` specifies Python 3.9, PyTorch 1.10.0, CUDA 11.3 and torchvision 0.11.1. The original README says Python 3.8 / PyTorch 1.11. This inconsistency should be resolved against a working environment before claiming a reproducible setup.
- The checked-in full evaluation defaults to five episodes per object via `config/cfg_tabletop.yaml`; the total depends on the scene's objects. The supplied 1,500-episode result cannot be reconstructed from this default alone.
- `cfg_tabletop_dinov3.yaml` changes the gripper affordance encoder while retaining the saved SAC policy configuration. A direct causal encoder comparison still requires the exact same policy and matched other settings.
- On the current Windows Python 3.14 environment, `python -m pytest -p no:cacheprovider tests -q` reported 2 passed and 7 skipped in 1.35 seconds. The skipped modules require unavailable research dependencies; no GPU or simulator evaluation was attempted.

## Repository hygiene and presentation

The dominant C classification on GitHub is plausibly due to the large vendored `VREnv/egl_check/glad` C source and header files and other simulator assets; this is an inference, not a verified Linguist breakdown. Add `.gitattributes` only after confirming GitHub's breakdown if it materially improves navigation. The repository tracks 1,326 files, including simulator meshes/textures; the largest inspected tracked asset is about 46 MB. No model weights were found tracked. Keep simulator assets unless their role and redistribution terms are understood; do not delete them solely to change the language bar.

The original README has been preserved at `docs/upstream-readme.md`. Existing original docs and MIT copyright remain in place. DINOv3 license terms should be reviewed directly before redistributing any code or weights. No generated visual has been added because paired, verifiable comparison examples were not located.

## Incremental priorities

1. **P0:** recover the perception evaluation report/script and 1,500-episode run artifacts; record exact configs, checkpoints, hashes, seeds, dataset split, and hardware; reconcile the mIoU discrepancy.
2. **P0:** verify installation and commands in a compatible Linux/CUDA environment; document the tested dependency lock and test outcomes.
3. **P1:** add structured result CSV/JSON and a small number of real image comparisons with traceable source samples and checkpoint provenance.
4. **P1:** inspect GitHub's actual language breakdown and set repository description/topics after the evidence is complete.
5. **P2:** run controlled experiments to isolate encoder, target selection, policy compatibility, resolution, and training-budget effects; add lightweight CI for tests that truly run without GPU.

