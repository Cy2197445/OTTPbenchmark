# Reproduction

## Archived results

Run `python release_tools/verify_release.py` to check file hashes, run records, seed-table metrics, and shared preprocessing files. Add `--statistics` to recompute the main tables, F1-F5 comparisons, seed-stability summaries, and offline alarm tables. This check uses the archived evidence without downloading raw data.

The principal result directories are:

| Directory | Analysis |
| --- | --- |
| `results/main/` | Main Step results, history gain, Outcome summaries, and figure |
| `results/model_comparison/` | Three-model task/seed matrix, tuning checks, and complexity |
| `results/target_ablation/` | Target and cost-masking ablations |
| `results/split_audit/` | Episode-split and source-field audits |
| `results/seed_stability/` | Eight tasks, fixed split, five training seeds |
| `results/external/` | FISOR Point Robot results |
| `results/task_groups/` | Descriptive DSRL task-group comparisons |
| `results/offline_utility/` | Validation-selected alarm-budget replay |
| `results/diagnostics/` | Earlier fixed-model diagnostics and convergence curves |

## Full three-model matrix

Download the tasks and build their precursor files. The task-to-file mapping and source hashes are in `manifests/datasets.json`.

```sh
python ctsc/scripts/download_unified38_public_data.py --output-dir work/raw
python ctsc/scripts/build_benchmark_datasets.py --raw-dir work/raw --output-dir work/precursors
python ctsc/scripts/run_benchmark.py --dataset-dir work/precursors --output-dir work/main --device cuda
python ctsc/scripts/analyze_benchmark.py --experiment-root work/main --output-dir work/main_tables
```

The default matrix contains 342 preprocessing cells and 1,026 model runs: 38 tasks, three split seeds, three conditions, and three models. Training uses 12 epochs, batch size 256, weight decay 1e-5, and the model-specific learning rates in the protocol. Four explicit seeds are equal to the selected split seed. `--prepare-only` writes the selected plan; `--resume` continues matching completed cells. A Transformer CUDA-memory fallback uses batch size 128 with two-step gradient accumulation, as in the recorded implementation.

Task, seed, and condition selectors support smaller runs. `--smoke` changes only the training length and sample caps for an execution check; it does not reproduce the full-matrix numbers.

The analysis command checks model and metrics hashes, training histories, four-seed settings, and shared preprocessing before producing seed/task tables, the main Step and Outcome tables, history gains, and F1-F3 paired statistics. Full tables require all 38 tasks, three seeds, three conditions, and three models at the recorded training settings. For subsets and smoke runs, add `--partial` to export seed metrics only. Analysis outputs go to a separate empty directory. To regenerate these tables from the distributed records, use `--experiment-root runs/main` and another output directory.

`manifests/datasets.json` is the fixed source catalog (filenames, sizes, URLs, and hashes). The downloader writes a local `download_log.json` recording actual file paths. `manifests/experiment_registry.json` supplies experiment settings and task identities. Commands accepting `--raw-dir` use the fixed catalog to locate downloaded files.

## Additional experiments

### Target and cost-masking ablations

The target/cost ablations use a fixed learning rate of 1e-3 and the original legacy-seed reference protocol. Train that reference separately from the main, model-specific-learning-rate matrix:

```sh
python ctsc/scripts/run_ablation_reference.py --dataset-dir work/precursors --output-dir work/ablation_reference --device cuda
python ctsc/scripts/run_revision_stage4.py --precursor-source-dir work/precursors --g0-root work/ablation_reference --output-dir work/target_ablation --device cuda
```

### Diagnostic experiments

The following commands run the recorded target-definition, split-family, feature-profile, window-size, and anti-shortcut diagnostics. Each suite writes its own summaries and analyses. These retain their original diagnostic model sets and aggregation rules; the three-model main tables use `analyze_benchmark.py`.

```sh
python ctsc/scripts/run_Unified38_target_definition_ablation.py --precursor-source-dir work/precursors --output-dir work/target_definition --device cpu
python ctsc/scripts/run_Unified38_split_family_counterfactual.py --precursor-source-dir work/precursors --output-dir work/split_family --device cpu
python ctsc/scripts/run_Unified38_gru_profile_matrix.py --precursor-source-dir work/precursors --output-dir work/profile_matrix --device cpu
python ctsc/scripts/run_Unified38_gru_window_sweep.py --precursor-source-dir work/precursors --output-dir work/window_sweep --device cpu
python ctsc/scripts/run_Unified38_anti_shortcut_suite.py --raw-dir work/raw --output-dir work/anti_shortcut --device cpu
```

Select `--tasks BallRun --split-seeds 0` for a smaller diagnostic run; use `--epochs 1 --max-train-samples 1000 --max-val-samples 500 --max-test-samples 1000` for an execution check. The split-family suite can also build precursors from `--raw-dir work/raw` when no `--precursor-source-dir` is provided. The anti-shortcut suite accepts either the fixed source catalog with `--raw-dir`, or a local download log through `--download-manifest`.

### Learning rates, seed stability, and external data

`run_revision_stage5.py --phase tuning --precursor-source-dir work/precursors --output-dir work/lr_selection --device cuda` reruns the 36 validation-only learning-rate jobs. `run_additional.py --experiment seed_stability --dataset-dir work/precursors --output-dir work/seed_stability --device cuda` runs the fixed-split training-seed study.

```sh
python ctsc/scripts/fetch_revision_stage8a_point_robot.py --output work/fisor/point_robot-expert-random-100k.hdf5
python ctsc/scripts/build_revision_stage8a_point_robot.py --raw work/fisor/point_robot-expert-random-100k.hdf5 --output work/precursors/FISOR_PointRobot.pt --audit work/fisor/build_audit.json
python ctsc/scripts/run_additional.py --experiment external --dataset-dir work/precursors --output-dir work/external --device cpu
```

### Offline alarm utility

Replay full-training models from the main matrix with their matching precursor files:

```sh
python ctsc/scripts/run_revision_stage9_utility.py --experiment-root work/main --dataset-dir work/precursors --output-dir work/offline_utility --device cpu
python ctsc/scripts/analyze_revision_stage9_utility.py --input-dir work/offline_utility --output-dir work/offline_utility_tables
```

## Model-replay tests

The default suite skips six validation/test replay comparisons when the BallRun seed-0 GRU, FlatMLP, and Transformer weights or precursor file are absent. These weights are generated by `run_benchmark.py`. After training, set the following environment variables to run all six checks. A single-epoch smoke matrix also supports these implementation checks; the tests read its actual sample caps and compare replay AUPRC with its saved metrics.

PowerShell:

```powershell
$env:OTTP_REPLAY_EXPERIMENT_ROOT = (Resolve-Path work/smoke).Path
$env:OTTP_REPLAY_DATASET_DIR = (Resolve-Path work/precursors).Path
python -m pytest tests/test_stage9_utility.py -k test_vectorized_real_g5a_scores_match_standard_replay -q
```

POSIX shell:

```sh
OTTP_REPLAY_EXPERIMENT_ROOT="$PWD/work/smoke" OTTP_REPLAY_DATASET_DIR="$PWD/work/precursors" python -m pytest tests/test_stage9_utility.py -k test_vectorized_real_g5a_scores_match_standard_replay -q
```

Tests use `work/main` and `work/precursors` by default. Explicitly configured missing assets cause failures. The two environment variables can be unset to return to the default paths.

## Recorded preprocessing

`manifests/index_reuse.json` maps all 342 main-matrix cells to their shared arrays and condition-specific thresholds and episode assignments. Array files are stored once per task/split combination. Their hashes identify the original index and normalization contents.

Historical precursor and parent-artifact hashes identify the original experiment files. Public path fields have been made portable; `provenance/SHA256SUMS.txt` identifies the distributed files. Current precursor reconstruction preserves the underlying arrays and includes an episode-group compatibility alias. Serialized hashes can differ when metadata or source paths differ; compare the raw-data hash and array contents when checking reconstruction.
