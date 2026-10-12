'Run canonical Unified38 GRU window-size sweeps on existing precursor datasets.'

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

import torch

from ctsc.precursor.public_source_catalog import dsrl_catalog_records
from ctsc.scripts.run_benchmark import output_directory


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work" / "window_sweep"
DEFAULT_PRECURSOR_SOURCE_DIR = REPO_ROOT / "work" / "precursors"
DEFAULT_WINDOW_SIZES = [1, 4, 8, 16, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--window-sizes", type=int, nargs="+", default=list(DEFAULT_WINDOW_SIZES))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument(
        "--precursor-source-dir",
        type=str,
        default=str(DEFAULT_PRECURSOR_SOURCE_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--fallback-concurrency", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--step-q-low", type=float, default=0.75)
    parser.add_argument("--step-q-high", type=float, default=0.90)
    parser.add_argument("--outcome-q-low", type=float, default=0.75)
    parser.add_argument("--outcome-q-high", type=float, default=0.90)
    parser.add_argument("--max-train-samples", type=int, default=100000)
    parser.add_argument("--max-val-samples", type=int, default=50000)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--critic-baseline-mode", type=str, choices=["dataset", "disabled"], default="dataset")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _metrics_summary(metrics_path: Path, bundle_path: Path, *, window_size: int) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    label_step = metrics["label_step"]["test"]["overall"]
    label_outcome = metrics["label_outcome"]["test"]["overall"]
    return {
        "window_size": int(window_size),
        "bundle_path": str(bundle_path),
        "metrics_path": str(metrics_path),
        "model_name": str(metrics["model_name"]),
        "model_family": str(metrics["model_family"]),
        "input_view": str(metrics["input_view"]),
        "feature_profile": str(metrics["feature_profile"]),
        "val_step_catastrophic_auprc": float(metrics["label_step"]["val"]["overall"]["catastrophic_auprc"]),
        "test_step_catastrophic_auprc": float(label_step["catastrophic_auprc"]),
        "val_outcome_catastrophic_auprc": float(metrics["label_outcome"]["val"]["overall"]["catastrophic_auprc"]),
        "test_outcome_catastrophic_auprc": float(label_outcome["catastrophic_auprc"]),
        "test_catastrophic_ece": float(label_step["catastrophic_ece"]),
        "test_lead_time_median_at_0.5": float(label_step["lead_time_median_at_0.5"]),
        "training": dict(metrics["training"]),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_logged(command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
        stdout_handle.write(f"$ {' '.join(command)}\n")
        stderr_handle.write(f"$ {' '.join(command)}\n")
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")


def _task_job_key(job: dict[str, Any]) -> tuple[str, int]:
    return (str(job["task_name"]), int(job["split_seed"]))


def _run_task(job: dict[str, Any]) -> dict[str, Any]:
    task_name = str(job["task_name"])
    split_seed = int(job["split_seed"])
    task_dir = Path(job["task_dir"])
    summary_path = task_dir / "window_sweep_summary.json"
    if bool(job["skip_existing"]) and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    precursor_path = Path(job["precursor_path"])
    if not precursor_path.exists():
        raise FileNotFoundError(f"Missing precursor for task `{task_name}`: {precursor_path}")

    stdout_path = Path(job["stdout_path"])
    stderr_path = Path(job["stderr_path"])
    window_sizes = [int(value) for value in job["window_sizes"]]
    window_summaries: dict[str, Any] = {}
    for window_size in window_sizes:
        window_dir = task_dir / f"window_{int(window_size)}"
        metrics_path = window_dir / "metrics.json"
        bundle_path = window_dir / "precursor_model.pt"
        if not (bool(job["skip_existing"]) and metrics_path.exists() and bundle_path.exists()):
            command = [
                sys.executable,
                "ctsc/scripts/train_precursor_model.py",
                "--dataset",
                str(precursor_path),
                "--output-dir",
                str(window_dir),
                "--model",
                "gru",
                "--seed",
                str(split_seed),
                "--device",
                str(job["device"]),
                "--feature-profile",
                "full",
                "--window-size",
                str(int(window_size)),
                "--hidden-dim",
                str(int(job["hidden_dim"])),
                "--num-layers",
                str(int(job["num_layers"])),
                "--dropout",
                str(float(job["dropout"])),
                "--batch-size",
                str(int(job["batch_size"])),
                "--epochs",
                str(int(job["epochs"])),
                "--lr",
                str(float(job["lr"])),
                "--weight-decay",
                str(float(job["weight_decay"])),
                "--num-workers",
                str(int(job["num_workers"])),
                "--step-q-low",
                str(float(job["step_q_low"])),
                "--step-q-high",
                str(float(job["step_q_high"])),
                "--outcome-q-low",
                str(float(job["outcome_q_low"])),
                "--outcome-q-high",
                str(float(job["outcome_q_high"])),
                "--critic-baseline-mode",
                str(job["critic_baseline_mode"]),
            ]
            if job["max_train_samples"] is not None:
                command.extend(["--max-train-samples", str(int(job["max_train_samples"]))])
            if job["max_val_samples"] is not None:
                command.extend(["--max-val-samples", str(int(job["max_val_samples"]))])
            if job["max_test_samples"] is not None:
                command.extend(["--max-test-samples", str(int(job["max_test_samples"]))])
            _run_logged(command, cwd=REPO_ROOT, stdout_path=stdout_path, stderr_path=stderr_path)
        window_summaries[str(int(window_size))] = _metrics_summary(metrics_path, bundle_path, window_size=int(window_size))

    payload: dict[str, Any] = {
        "phase": "Unified38_gru_window_sweep",
        "status": "success",
        "slice_name": "canonical",
        "task_name": task_name,
        "source_platform": str(job["source_platform"]),
        "split_seed": split_seed,
        "window_sizes": window_sizes,
        "windows": window_summaries,
        "shared_defaults": {
            "hidden_dim": int(job["hidden_dim"]),
            "num_layers": int(job["num_layers"]),
            "dropout": float(job["dropout"]),
            "batch_size": int(job["batch_size"]),
            "epochs": int(job["epochs"]),
            "lr": float(job["lr"]),
            "weight_decay": float(job["weight_decay"]),
            "step_q_low": float(job["step_q_low"]),
            "step_q_high": float(job["step_q_high"]),
            "outcome_q_low": float(job["outcome_q_low"]),
            "outcome_q_high": float(job["outcome_q_high"]),
            "max_train_samples": None if job["max_train_samples"] is None else int(job["max_train_samples"]),
            "max_val_samples": None if job["max_val_samples"] is None else int(job["max_val_samples"]),
            "max_test_samples": None if job["max_test_samples"] is None else int(job["max_test_samples"]),
        },
    }
    if "1" in window_summaries and "32" in window_summaries:
        payload["window_contrasts"] = {
            "window_32_minus_window_1_step_auprc": float(
                window_summaries["32"]["test_step_catastrophic_auprc"] - window_summaries["1"]["test_step_catastrophic_auprc"]
            )
        }
    _write_json(summary_path, payload)
    return payload


def _execute_jobs(jobs: list[dict[str, Any]], *, concurrency: int, fallback_concurrency: int) -> tuple[int, bool]:
    completed_keys: set[tuple[str, int]] = set()
    fallback_used = False

    def run_one(job: dict[str, Any]) -> dict[str, Any]:
        payload = _run_task(job)
        completed_keys.add(_task_job_key(job))
        return payload

    if int(concurrency) <= 1:
        for job in jobs:
            run_one(job)
        return len(completed_keys), fallback_used

    try:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
            future_to_job = {executor.submit(_run_task, job): job for job in jobs}
            for future in as_completed(future_to_job):
                payload = future.result()
                completed_keys.add((str(payload["task_name"]), int(payload["split_seed"])))
    except Exception:
        if int(fallback_concurrency) < 1:
            raise
        fallback_used = True
        remaining_jobs = [job for job in jobs if _task_job_key(job) not in completed_keys]
        for job in remaining_jobs:
            run_one(job)
    return len(completed_keys), fallback_used


def main() -> None:
    args = parse_args()
    output_dir = output_directory(_resolve(args.output_dir))
    precursor_source_dir = _resolve(args.precursor_source_dir)
    records = dsrl_catalog_records(task_names=args.tasks)
    tasks = [str(row["task_name"]) for row in records]
    task_records = {str(row["task_name"]): row for row in records}
    split_seeds = [int(seed) for seed in args.split_seeds]
    window_sizes = list(dict.fromkeys(int(value) for value in args.window_sizes))

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    launch_manifest = {
        "phase": "Unified38_gru_window_sweep",
        "tasks": tasks,
        "split_seeds": split_seeds,
        "window_sizes": window_sizes,
        "output_dir": str(output_dir),
        "precursor_source_dir": str(precursor_source_dir),
        "device": str(args.device),
        "concurrency": int(args.concurrency),
        "fallback_concurrency": int(args.fallback_concurrency),
    }
    _write_json(output_dir / "launch_manifest.json", launch_manifest)

    jobs: list[dict[str, Any]] = []
    for task_name in tasks:
        source_platform = str(task_records[task_name]["source_platform"])
        precursor_path = precursor_source_dir / f"{task_name}.pt"
        for split_seed in split_seeds:
            task_dir = output_dir / "tasks" / task_name / f"seed_{int(split_seed)}"
            task_dir.mkdir(parents=True, exist_ok=True)
            jobs.append(
                {
                    "task_name": task_name,
                    "split_seed": int(split_seed),
                    "source_platform": source_platform,
                    "precursor_path": str(precursor_path),
                    "task_dir": str(task_dir),
                    "stdout_path": str(logs_dir / f"{task_name}.seed_{int(split_seed)}.out.log"),
                    "stderr_path": str(logs_dir / f"{task_name}.seed_{int(split_seed)}.err.log"),
                    "window_sizes": window_sizes,
                    "skip_existing": bool(args.skip_existing),
                    "device": str(args.device),
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "dropout": float(args.dropout),
                    "batch_size": int(args.batch_size),
                    "epochs": int(args.epochs),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "num_workers": int(args.num_workers),
                    "step_q_low": float(args.step_q_low),
                    "step_q_high": float(args.step_q_high),
                    "outcome_q_low": float(args.outcome_q_low),
                    "outcome_q_high": float(args.outcome_q_high),
                    "max_train_samples": args.max_train_samples,
                    "max_val_samples": args.max_val_samples,
                    "max_test_samples": args.max_test_samples,
                    "critic_baseline_mode": str(args.critic_baseline_mode),
                }
            )

    completed_job_count, fallback_used = _execute_jobs(
        jobs,
        concurrency=int(args.concurrency),
        fallback_concurrency=int(args.fallback_concurrency),
    )
    finish_status = {
        "phase": "Unified38_gru_window_sweep",
        "status": "success",
        "task_count": len(tasks),
        "split_seed_count": len(split_seeds),
        "job_count": len(jobs),
        "completed_job_count": int(completed_job_count),
        "window_sizes": window_sizes,
        "output_dir": str(output_dir),
        "fallback_used": bool(fallback_used),
    }
    _write_json(output_dir / "finish_status.json", finish_status)
    print(json.dumps(finish_status, indent=2))


if __name__ == "__main__":
    main()
