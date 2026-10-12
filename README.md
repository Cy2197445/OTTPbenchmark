# OTTP benchmark

OTTP evaluates prediction of a hindsight-defined discounted cumulative-cost label from trajectory prefixes in offline safe reinforcement learning. The benchmark covers 38 DSRL tasks, with GRU, FlatMLP, and causal Transformer baselines.

## Installation

Use Python 3.10. The following commands install the CPU build used for the public-package tests:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q
python release_tools/verify_release.py
```

For GPU training, install the corresponding PyTorch 2.11.0 CUDA build before installing the remaining requirements. The Dockerfile defines a CPU environment with the same dependencies:

```sh
docker build -t ottp:1.1.0 .
docker run --rm ottp:1.1.0
```

Six model-replay tests require locally trained BallRun weights and a precursor dataset. They are skipped until those assets are available. [Reproduction](docs/reproduction.md#model-replay-tests) shows how to supply them.

## Quick start

Download one task, build its trajectory-prefix dataset, and run a small training check:

```sh
python ctsc/scripts/download_unified38_public_data.py --tasks BallRun --output-dir work/raw
python ctsc/scripts/build_public_precursor_dataset.py --dataset-path work/raw/SafetyBallRun-v0-80-940.hdf5 --task-name BallRun --output work/precursors/BallRun.pt
python ctsc/scripts/run_benchmark.py --dataset-dir work/precursors --output-dir work/smoke --tasks BallRun --split-seeds 0 --conditions canonical_full --smoke --device cpu
python ctsc/scripts/analyze_benchmark.py --experiment-root work/smoke --output-dir work/smoke_tables --partial
```

The smoke setting uses one epoch and reduced sample caps. Full experiments use the settings in [Reproduction](docs/reproduction.md).

## Computing resources

Archive verification uses the approximately 421 MB unpacked package and the Python environment. The single-task quick start runs on CPU and downloads approximately 5.9 MB of raw BallRun data. The complete 38-task raw collection is 25,910,326,255 bytes (25.91 GB / 24.13 GiB); allow additional space for precursor files, model weights, outputs, and the environment. Full training comprises 1,026 twelve-epoch model runs. CPU and CUDA devices are supported; runtime and memory requirements depend on the task and device. Training commands check available disk space before starting jobs.

## Protocol and results

The canonical target uses current-inclusive discounted cost-to-go with gamma 0.99 and training-set q75/q90 thresholds. Inputs contain up to 32 steps. Episodes are split 60/20/20 with split seeds 0, 1, and 2; thresholds and normalization are fitted on training data. Each task contributes one seed-averaged observation to the main statistical comparisons. See [Protocol](docs/protocol.md).

Mean Step AUPRC across the 38 tasks:

| Model | Step AUPRC | Full-prefix minus current-only |
| --- | ---: | ---: |
| GRU | 0.7390 | +0.0313 |
| FlatMLP | 0.7144 | -0.0031 |
| Causal Transformer | 0.7354 | +0.0140 |

The matched empirical Step prevalence is 0.1009. See the [task-level tables](results/main/) and [paired statistical comparisons](results/model_comparison/f1_f2_f3_statistics.csv). Target ablations use their separately recorded fixed learning rate of 1e-3. The FISOR experiment covers one additional task, and alarm utility is evaluated by offline replay.

## Contents

| Directory | Contents |
| --- | --- |
| `ctsc/` | Dataset construction, models, training, analyses, and diagnostics |
| `results/` | Task and seed tables, statistics, split audits, and figures |
| `runs/` | Recorded metrics, training histories, and preprocessing artifacts |
| `manifests/` | Dataset sources and hashes, experiment settings, and shared-index mapping |
| `tests/` | Scientific unit tests and public-release checks |
| `release_tools/` | Archive and numerical verification |

Raw datasets are downloaded from their publishers, and trained weights are generated locally. Access information is in [Data sources](docs/data_sources.md).

## Citation and license

Yi Chai, Dandan Liu, and Zixiao Luo. *OTTP: A Benchmark Protocol for Trajectory-Prefix Risk Prediction in Offline Safe Reinforcement Learning.* Software citation metadata is provided in [CITATION.cff](CITATION.cff).

Project code and documentation are available under the [MIT license](LICENSE). Third-party datasets retain their upstream terms.
