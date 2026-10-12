'Run the Unified38 split-family counterfactual study.'

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.data import SPLIT_FAMILY_REGISTRY
from ctsc.precursor.public_paths import raw_dataset_path
from ctsc.scripts.run_benchmark import output_directory


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "manifests" / "experiment_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=str, default=str(REGISTRY_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--split-families", type=str, nargs="+", choices=list(SPLIT_FAMILY_REGISTRY), default=list(SPLIT_FAMILY_REGISTRY))
    parser.add_argument("--output-dir", type=str, default="work/split_family")
    parser.add_argument("--precursor-source-dir", type=str, default=None)
    parser.add_argument("--raw-dir", type=Path, default=Path("work/raw"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-profile", type=str, default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def _run_suite_job(
    command: list[str],
    *,
    cwd: Path,
    task_name: str,
    slice_name: str,
    split_seed: int,
    split_family: str,
) -> dict[str, Any]:
    _run(command, cwd=cwd)
    return {
        "task_name": str(task_name),
        "slice_name": str(slice_name),
        "split_seed": int(split_seed),
        "split_family": str(split_family),
    }


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path
    registry = _load_json(registry_path)
    output_dir = output_directory(REPO_ROOT / args.output_dir)
    precursor_dir = output_dir / "precursor_datasets"
    precursor_source_dir = None
    if args.precursor_source_dir is not None:
        precursor_source_dir = Path(args.precursor_source_dir)
        if not precursor_source_dir.is_absolute():
            precursor_source_dir = REPO_ROOT / precursor_source_dir
    task_run_dir = output_dir / "task_runs"
    aggregate_dir = output_dir / "aggregates"
    analysis_dir = output_dir / "analysis"
    if precursor_source_dir is None:
        precursor_dir.mkdir(parents=True, exist_ok=True)
    task_run_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    tasks = (
        list(dict.fromkeys(str(task) for task in args.tasks))
        if args.tasks
        else [str(task["task_name"]) for task in registry["tasks"]]
    )
    split_seeds = list(args.split_seeds) if args.split_seeds is not None else list(registry["mainline_fresh_rerun"]["split_seeds"])
    split_families = list(dict.fromkeys(str(family) for family in args.split_families))
    slices = list(registry["mainline_fresh_rerun"]["slices"])
    defaults = dict(registry["mainline_fresh_rerun"]["shared_defaults"])
    feature_profile = str(args.feature_profile) if args.feature_profile is not None else str(defaults["feature_profile"])
    suite_models = list(dict.fromkeys(str(model) for model in args.models)) if args.models else None
    epochs = int(args.epochs) if args.epochs is not None else int(defaults["epochs"])
    batch_size = int(args.batch_size) if args.batch_size is not None else int(defaults["batch_size"])
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else defaults.get("max_train_samples")
    max_val_samples = args.max_val_samples if args.max_val_samples is not None else defaults.get("max_val_samples")
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else defaults.get("max_test_samples")
    task_map = {str(task["task_name"]): task for task in registry["tasks"]}
    unknown_tasks = [task_name for task_name in tasks if task_name not in task_map]
    if unknown_tasks:
        raise ValueError(f"Unknown counterfactual task(s): {unknown_tasks}")

    suite_jobs: list[dict[str, Any]] = []
    for task_name in tasks:
        task = task_map[task_name]
        if precursor_source_dir is not None:
            precursor_path = precursor_source_dir / f"{task_name}.pt"
            if not precursor_path.exists():
                raise FileNotFoundError(f"Precursor source missing for task `{task_name}`: {precursor_path}")
        else:
            precursor_path = precursor_dir / f"{task_name}.pt"
        summary_path = precursor_dir / f"{task_name}.summary.json"
        if precursor_source_dir is None and not precursor_path.exists():
            configured = Path(str(task["dataset_path"]))
            if configured.is_absolute():
                dataset_path = configured
                if not dataset_path.is_file():
                    raise FileNotFoundError(dataset_path)
            else:
                dataset_path = raw_dataset_path(
                    task_name, _load_json(REPO_ROOT / "manifests/datasets.json"),
                    REPO_ROOT / args.raw_dir, root=REPO_ROOT,
                )
            _run(
                [
                    sys.executable,
                    "ctsc/scripts/build_public_precursor_dataset.py",
                    "--dataset-path",
                    str(dataset_path),
                    "--task-name",
                    task_name,
                    "--output",
                    str(precursor_path),
                    "--summary-json",
                    str(summary_path),
                ],
                cwd=REPO_ROOT,
            )
        for slice_spec in slices:
            slice_name = str(slice_spec["name"])
            for split_seed in split_seeds:
                for split_family in split_families:
                    suite_dir = task_run_dir / task_name / slice_name / f"seed_{int(split_seed)}" / str(split_family)
                    if args.skip_existing and (suite_dir / "suite_summary.json").exists():
                        continue
                    command = [
                        sys.executable,
                        "ctsc/scripts/run_precursor_baseline_suite.py",
                        "--dataset",
                        str(precursor_path),
                        "--output-dir",
                        str(suite_dir),
                        "--slice-name",
                        slice_name,
                        "--seed",
                        str(int(split_seed)),
                        "--split-family",
                        str(split_family),
                        "--device",
                        str(args.device),
                        "--feature-profile",
                        feature_profile,
                        "--window-size",
                        str(int(defaults["window_size"])),
                        "--hidden-dim",
                        str(int(defaults["hidden_dim"])),
                        "--num-layers",
                        str(int(defaults["num_layers"])),
                        "--dropout",
                        str(float(defaults["dropout"])),
                        "--cnn-kernel-size",
                        str(int(defaults["cnn_kernel_size"])),
                        "--mlp-depth",
                        str(int(defaults["mlp_depth"])),
                        "--batch-size",
                        str(int(batch_size)),
                        "--epochs",
                        str(int(epochs)),
                        "--lr",
                        str(float(defaults["lr"])),
                        "--weight-decay",
                        str(float(defaults["weight_decay"])),
                        "--step-q-low",
                        str(float(slice_spec["step_q_low"])),
                        "--step-q-high",
                        str(float(slice_spec["step_q_high"])),
                        "--outcome-q-low",
                        str(float(slice_spec["outcome_q_low"])),
                        "--outcome-q-high",
                        str(float(slice_spec["outcome_q_high"])),
                    ]
                    if suite_models:
                        command.extend(["--models", *suite_models])
                    if max_train_samples is not None:
                        command.extend(["--max-train-samples", str(int(max_train_samples))])
                    if max_val_samples is not None:
                        command.extend(["--max-val-samples", str(int(max_val_samples))])
                    if max_test_samples is not None:
                        command.extend(["--max-test-samples", str(int(max_test_samples))])
                    suite_jobs.append(
                        {
                            "command": command,
                            "task_name": str(task_name),
                            "slice_name": str(slice_name),
                            "split_seed": int(split_seed),
                            "split_family": str(split_family),
                        }
                    )

    if int(args.concurrency) <= 1:
        for job in suite_jobs:
            _run_suite_job(
                job["command"],
                cwd=REPO_ROOT,
                task_name=str(job["task_name"]),
                slice_name=str(job["slice_name"]),
                split_seed=int(job["split_seed"]),
                split_family=str(job["split_family"]),
            )
    else:
        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as executor:
            futures = [
                executor.submit(
                    _run_suite_job,
                    job["command"],
                    cwd=REPO_ROOT,
                    task_name=str(job["task_name"]),
                    slice_name=str(job["slice_name"]),
                    split_seed=int(job["split_seed"]),
                    split_family=str(job["split_family"]),
                )
                for job in suite_jobs
            ]
            for future in as_completed(futures):
                future.result()

    _run(
        [
            sys.executable,
            "ctsc/scripts/aggregate_Unified38_split_family_counterfactual.py",
            "--input-root",
            str(task_run_dir),
            "--output-dir",
            str(aggregate_dir),
            "--task-names",
            *tasks,
            "--split-seeds",
            *[str(int(seed)) for seed in split_seeds],
            "--slices",
            *[str(slice_spec["name"]) for slice_spec in slices],
            "--split-families",
            *split_families,
        ],
        cwd=REPO_ROOT,
    )
    _run(
        [
            sys.executable,
            "ctsc/scripts/analyze_Unified38_split_family_counterfactual.py",
            "--aggregate-dir",
            str(aggregate_dir),
            "--output-dir",
            str(analysis_dir),
        ],
        cwd=REPO_ROOT,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "task_count": len(tasks),
                "split_seed_count": len(split_seeds),
                "split_families": split_families,
                "suite_job_count": len(suite_jobs),
                "concurrency": int(args.concurrency),
                "feature_profile": feature_profile,
                "models": suite_models if suite_models is not None else "suite_default",
                "aggregate_dir": str(aggregate_dir),
                "analysis_dir": str(analysis_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
