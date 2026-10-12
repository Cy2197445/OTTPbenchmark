'Run the Unified38 current-facing target-definition ablation study.'

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.ablation import ABLATION_GROUP_MODE_REGISTRY
from ctsc.scripts.run_benchmark import output_directory


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "manifests" / "experiment_registry.json"
DEFAULT_PRECURSOR_SOURCE_DIR = REPO_ROOT / "work" / "precursors"
DEFAULT_MODELS = ["gru", "flat-mlp", "prevalence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=str, default=str(REGISTRY_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="work/target_definition")
    parser.add_argument(
        "--precursor-source-dir",
        type=str,
        default=str(DEFAULT_PRECURSOR_SOURCE_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--initial-concurrency", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--min-concurrency", type=int, default=2)
    parser.add_argument("--warmup-jobs", type=int, default=6)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _canonical_slice(registry: dict[str, Any]) -> dict[str, Any]:
    for slice_spec in registry["mainline_fresh_rerun"]["slices"]:
        if str(slice_spec["name"]) == "canonical":
            return dict(slice_spec)
    raise ValueError("Registry does not contain a canonical slice.")


def _gpu_snapshot() -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    payload = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(payload) != 6:
        return None
    return {
        "name": payload[0],
        "memory_total_mb": int(payload[1]),
        "memory_used_mb": int(payload[2]),
        "memory_free_mb": int(payload[3]),
        "utilization_gpu": int(payload[4]),
        "temperature_gpu": int(payload[5]),
    }


def _memory_free_gb() -> float | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
            return None
        return float(status.ullAvailPhys / (1024**3))
    page_size = os.sysconf("SC_PAGE_SIZE")
    avail_pages = os.sysconf("SC_AVPHYS_PAGES")
    return float(page_size * avail_pages / (1024**3))


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path.anchor or path))
    return float(usage.free / (1024**3))


def _resource_snapshot(output_dir: Path) -> dict[str, Any]:
    return {
        "gpu": _gpu_snapshot(),
        "memory_free_gb": _memory_free_gb(),
        "disk_free_gb": _disk_free_gb(output_dir),
    }


def _contains_oom(text: str) -> bool:
    lowered = text.lower()
    return "cuda out of memory" in lowered or "out of memory" in lowered


def _run_suite_job(job: dict[str, Any], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        job["command"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "task_name": str(job["task_name"]),
        "slice_name": str(job["slice_name"]),
        "split_seed": int(job["split_seed"]),
        "ablation_group": str(job["ablation_group"]),
        "ablation_mode": str(job["ablation_mode"]),
        "suite_dir": str(job["suite_dir"]),
        "returncode": int(completed.returncode),
        "success": bool(completed.returncode == 0),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "oom_detected": _contains_oom((completed.stdout or "") + "\n" + (completed.stderr or "")),
    }


def _execute_jobs(jobs: list[dict[str, Any]], *, concurrency: int, cwd: Path) -> list[dict[str, Any]]:
    if not jobs:
        return []
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
        futures = [executor.submit(_run_suite_job, job, cwd=cwd) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "task_name": result["task_name"],
                        "split_seed": result["split_seed"],
                        "ablation_group": result["ablation_group"],
                        "ablation_mode": result["ablation_mode"],
                        "success": result["success"],
                        "oom_detected": result["oom_detected"],
                    },
                    ensure_ascii=False,
                )
            )
    return results


def _choose_followup_concurrency(
    *,
    warmup_results: list[dict[str, Any]],
    snapshot: dict[str, Any],
    initial: int,
    minimum: int,
    maximum: int,
) -> tuple[int, dict[str, Any]]:
    gpu = snapshot.get("gpu")
    gpu_free_mb = None if gpu is None else float(gpu["memory_free_mb"])
    memory_free_gb = None if snapshot.get("memory_free_gb") is None else float(snapshot["memory_free_gb"])
    disk_free_gb = float(snapshot["disk_free_gb"])
    any_failure = any(not row["success"] for row in warmup_results)
    any_oom = any(bool(row["oom_detected"]) for row in warmup_results)
    resource_low = (
        any_oom
        or (gpu_free_mb is not None and gpu_free_mb < 1200.0)
        or (memory_free_gb is not None and memory_free_gb < 8.0)
        or disk_free_gb < 25.0
    )
    scale_up_ok = (
        not any_failure
        and not any_oom
        and (gpu_free_mb is None or gpu_free_mb >= 2000.0)
        and (memory_free_gb is None or memory_free_gb >= 12.0)
        and disk_free_gb >= 28.0
    )
    if any_failure or resource_low:
        next_concurrency = int(minimum)
        decision = "degrade_to_min"
    elif scale_up_ok:
        next_concurrency = int(maximum)
        decision = "scale_up"
    else:
        next_concurrency = int(initial)
        decision = "hold_initial"
    return next_concurrency, {
        "decision": decision,
        "any_failure": any_failure,
        "any_oom": any_oom,
        "snapshot": snapshot,
    }


def _failed_jobs(results: list[dict[str, Any]], all_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed_keys = {
        (
            row["task_name"],
            row["slice_name"],
            row["split_seed"],
            row["ablation_group"],
            row["ablation_mode"],
        )
        for row in results
        if not row["success"]
    }
    return [
        job
        for job in all_jobs
        if (
            str(job["task_name"]),
            str(job["slice_name"]),
            int(job["split_seed"]),
            str(job["ablation_group"]),
            str(job["ablation_mode"]),
        )
        in failed_keys
    ]


def _persistent_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for row in results:
        key = (
            str(row["task_name"]),
            str(row["slice_name"]),
            int(row["split_seed"]),
            str(row["ablation_group"]),
            str(row["ablation_mode"]),
        )
        latest_by_key[key] = row
    return [row for row in latest_by_key.values() if not row["success"]]


def main() -> None:
    args = parse_args()
    registry = _load_json(_resolve(args.registry))
    output_dir = output_directory(_resolve(args.output_dir))
    task_run_dir = output_dir / "task_runs"
    aggregate_dir = output_dir / "aggregates"
    analysis_dir = output_dir / "analysis"
    task_run_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    precursor_source_dir = _resolve(args.precursor_source_dir)
    if not precursor_source_dir.exists():
        raise FileNotFoundError(f"Precursor source directory not found: {precursor_source_dir}")

    tasks = (
        list(dict.fromkeys(str(task) for task in args.tasks))
        if args.tasks
        else [str(task["task_name"]) for task in registry["tasks"]]
    )
    split_seeds = list(args.split_seeds) if args.split_seeds is not None else list(registry["mainline_fresh_rerun"]["split_seeds"])
    slice_spec = _canonical_slice(registry)
    defaults = dict(registry["mainline_fresh_rerun"]["shared_defaults"])
    task_map = {str(task["task_name"]): task for task in registry["tasks"]}
    unknown_tasks = [task_name for task_name in tasks if task_name not in task_map]
    if unknown_tasks:
        raise ValueError(f"Unknown target-definition ablation task(s): {unknown_tasks}")

    epochs = int(args.epochs) if args.epochs is not None else int(defaults["epochs"])
    batch_size = int(args.batch_size) if args.batch_size is not None else int(defaults["batch_size"])
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else defaults.get("max_train_samples")
    max_val_samples = args.max_val_samples if args.max_val_samples is not None else defaults.get("max_val_samples")
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else defaults.get("max_test_samples")

    suite_jobs: list[dict[str, Any]] = []
    for task_name in tasks:
        precursor_path = precursor_source_dir / f"{task_name}.pt"
        if not precursor_path.exists():
            raise FileNotFoundError(f"Precursor source missing for task `{task_name}`: {precursor_path}")
        for split_seed in split_seeds:
            for ablation_group, modes in ABLATION_GROUP_MODE_REGISTRY.items():
                for ablation_mode in modes:
                    suite_dir = task_run_dir / task_name / "canonical" / f"seed_{int(split_seed)}" / ablation_group / ablation_mode
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
                        "canonical",
                        "--seed",
                        str(int(split_seed)),
                        "--split-family",
                        "episode_group",
                        "--ablation-group",
                        ablation_group,
                        "--ablation-mode",
                        ablation_mode,
                        "--device",
                        str(args.device),
                        "--feature-profile",
                        str(defaults["feature_profile"]),
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
                        "--models",
                        *[str(model) for model in args.models],
                    ]
                    if max_train_samples is not None:
                        command.extend(["--max-train-samples", str(int(max_train_samples))])
                    if max_val_samples is not None:
                        command.extend(["--max-val-samples", str(int(max_val_samples))])
                    if max_test_samples is not None:
                        command.extend(["--max-test-samples", str(int(max_test_samples))])
                    suite_jobs.append(
                        {
                            "command": command,
                            "task_name": task_name,
                            "slice_name": "canonical",
                            "split_seed": int(split_seed),
                            "ablation_group": ablation_group,
                            "ablation_mode": ablation_mode,
                            "suite_dir": suite_dir,
                        }
                    )

    initial_concurrency = max(int(args.min_concurrency), min(int(args.initial_concurrency), int(args.max_concurrency)))
    warmup_count = min(int(args.warmup_jobs), len(suite_jobs))
    warmup_jobs = suite_jobs[:warmup_count]
    remaining_jobs = suite_jobs[warmup_count:]

    warmup_results = _execute_jobs(warmup_jobs, concurrency=initial_concurrency, cwd=REPO_ROOT)
    warmup_snapshot = _resource_snapshot(output_dir)
    steady_concurrency, warmup_policy = _choose_followup_concurrency(
        warmup_results=warmup_results,
        snapshot=warmup_snapshot,
        initial=initial_concurrency,
        minimum=int(args.min_concurrency),
        maximum=int(args.max_concurrency),
    )

    all_results = list(warmup_results)
    failed_after_warmup = _failed_jobs(warmup_results, warmup_jobs)
    retry_results: list[dict[str, Any]] = []
    if failed_after_warmup and steady_concurrency != int(args.min_concurrency):
        steady_concurrency = int(args.min_concurrency)
    if failed_after_warmup:
        retry_results = _execute_jobs(failed_after_warmup, concurrency=int(args.min_concurrency), cwd=REPO_ROOT)
        all_results.extend(retry_results)

    remaining_results = _execute_jobs(remaining_jobs, concurrency=steady_concurrency, cwd=REPO_ROOT)
    all_results.extend(remaining_results)
    failed_after_remaining = _failed_jobs(remaining_results, remaining_jobs)
    retry_remaining_results: list[dict[str, Any]] = []
    if failed_after_remaining and steady_concurrency > int(args.min_concurrency):
        retry_remaining_results = _execute_jobs(failed_after_remaining, concurrency=int(args.min_concurrency), cwd=REPO_ROOT)
        all_results.extend(retry_remaining_results)

    persistent_failures = _persistent_failures(all_results)
    if persistent_failures:
        failure_summary = [
            {
                "task_name": row["task_name"],
                "split_seed": row["split_seed"],
                "ablation_group": row["ablation_group"],
                "ablation_mode": row["ablation_mode"],
                "oom_detected": row["oom_detected"],
                "stderr_tail": (row["stderr"] or "")[-400:],
            }
            for row in persistent_failures
        ]
        raise RuntimeError(f"Target-definition ablation jobs failed: {json.dumps(failure_summary, ensure_ascii=False)}")

    subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/aggregate_Unified38_target_definition_ablation.py",
            "--input-root",
            str(task_run_dir),
            "--output-dir",
            str(aggregate_dir),
            "--task-names",
            *tasks,
            "--split-seeds",
            *[str(int(seed)) for seed in split_seeds],
        ],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/analyze_Unified38_target_definition_ablation.py",
            "--aggregate-dir",
            str(aggregate_dir),
            "--output-dir",
            str(analysis_dir),
        ],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "task_count": len(tasks),
                "split_seed_count": len(split_seeds),
                "suite_job_count": len(suite_jobs),
                "warmup_job_count": warmup_count,
                "initial_concurrency": initial_concurrency,
                "steady_concurrency": steady_concurrency,
                "warmup_policy": warmup_policy,
                "resource_snapshot_final": _resource_snapshot(output_dir),
                "precursor_source_dir": _repo_relative_or_absolute(precursor_source_dir),
                "aggregate_dir": str(aggregate_dir),
                "analysis_dir": str(analysis_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
