"""Run validation-only learning-rate selection on the recorded tuning panel."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.revision import file_sha256, require_free_space
from ctsc.precursor.stage5 import (
    CONDITIONS_BY_ID,
    FINAL_CONDITIONS,
    LEARNING_RATES,
    SPLIT_SEEDS,
    TUNING_MODELS,
    TUNING_TASKS,
    final_jobs,
    frozen_matrix_manifest,
    learning_rate_compatibility_gate,
    select_learning_rates,
    tuning_jobs,
    write_csv,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "work/lr_selection"
DEFAULT_PRECURSORS = REPO_ROOT / "work/precursors"
REGISTRY = REPO_ROOT / "manifests/experiment_registry.json"
MODEL_CLI = {"gru": "gru", "flat_mlp": "flat-mlp", "transformer": "transformer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)))
    parser.add_argument("--precursor-source-dir", default=str(DEFAULT_PRECURSORS.relative_to(REPO_ROOT)))
    parser.add_argument("--phase", choices=["tuning"], default="tuning")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retry-concurrency", type=int, default=2)
    parser.add_argument("--minimum-free-space-gib", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry() -> tuple[list[str], dict[str, Any]]:
    payload = _load_json(REGISTRY)
    tasks = [str(row["task_name"]) for row in payload["tasks"]]
    return tasks, dict(payload["mainline_fresh_rerun"]["shared_defaults"])


def _guard_root(output: Path, *, resume: bool, matrix: Mapping[str, Any]) -> None:
    from ctsc.scripts.run_benchmark import output_directory
    output_directory(output)
    matrix_path = output / "stage5_frozen_matrix.json"
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        if not resume:
            raise FileExistsError(f"Stage 5 output is nonempty; pass --resume to continue: {output}")
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Resume root lacks frozen matrix: {matrix_path}")
        existing = _load_json(matrix_path)
        if str(existing.get("matrix_sha256")) != str(matrix["matrix_sha256"]):
            raise ValueError("Existing Stage 5 matrix does not match the frozen matrix")
    output.mkdir(parents=True, exist_ok=True)
    write_json(matrix_path, matrix)


def _lr_token(value: float) -> str:
    return format(float(value), ".0e").replace("-", "m").replace("+", "p")


def _canonical_model(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _attempt_base(output: Path, job: Mapping[str, Any], attempt: int) -> Path:
    if str(job["job_id"]).startswith("tuning__"):
        return (
            output
            / "tuning_runs"
            / str(job["task_name"])
            / str(job["model_name"])
            / f"lr_{_lr_token(float(job['learning_rate']))}"
            / f"attempt-{attempt:03d}"
        )
    return (
        output
        / "final_runs"
        / str(job["condition_id"])
        / str(job["task_name"])
        / f"attempt-{attempt:03d}"
    )


def _run_dir(output: Path, job: Mapping[str, Any], attempt: int) -> Path:
    seed = int(job["split_seed"])
    return _attempt_base(output, job, attempt) / f"split_seed_{seed}" / f"train_seed_{seed}"


def _candidate_metrics(output: Path, job: Mapping[str, Any]) -> list[Path]:
    if str(job["job_id"]).startswith("tuning__"):
        base = (
            output
            / "tuning_runs"
            / str(job["task_name"])
            / str(job["model_name"])
            / f"lr_{_lr_token(float(job['learning_rate']))}"
        )
    else:
        base = output / "final_runs" / str(job["condition_id"]) / str(job["task_name"])
    seed = int(job["split_seed"])
    return sorted(base.glob(f"attempt-*/split_seed_{seed}/train_seed_{seed}/metrics.json"))


def _valid_success(path: Path, job: Mapping[str, Any]) -> bool:
    try:
        metrics = _load_json(path)
        if _canonical_model(metrics["model_name"]) != str(job["model_name"]):
            return False
        protocol = dict(metrics.get("label_protocol") or {})
        expected_q = (0.75, 0.90)
        expected_profile = "full"
        if not str(job["job_id"]).startswith("tuning__"):
            condition = CONDITIONS_BY_ID[str(job["condition_id"])]
            expected_q = (condition.q_low, condition.q_high)
            expected_profile = condition.feature_profile
        observed_q = (float(protocol["step_q_low"]), float(protocol["step_q_high"]))
        if observed_q != expected_q or str(metrics.get("feature_profile")) != expected_profile:
            return False
        if int(metrics.get("split_seed", protocol.get("split_seed", job["split_seed"]))) != int(job["split_seed"]):
            return False
        if str(job["job_id"]).startswith("tuning__"):
            run_manifest = _load_json(path.with_name("run_manifest.json"))
            observed_lr = float(run_manifest["training"]["learning_rate"])
            if not math.isclose(observed_lr, float(job["learning_rate"]), rel_tol=0.0, abs_tol=1e-15):
                return False
        history = list(metrics["training"]["history"])
        return len(history) == 12
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _existing_success(output: Path, job: Mapping[str, Any]) -> Path | None:
    valid = [path for path in _candidate_metrics(output, job) if _valid_success(path, job)]
    return valid[-1] if valid else None


def _next_attempt(output: Path, job: Mapping[str, Any]) -> int:
    for attempt in range(1, 1000):
        if not _attempt_base(output, job, attempt).exists():
            return attempt
    raise RuntimeError(f"Exhausted attempt identifiers for {job['job_id']}")


def _job_command(
    *,
    output: Path,
    precursor_root: Path,
    job: Mapping[str, Any],
    attempt: int,
    defaults: Mapping[str, Any],
    device: str,
    minimum_free_space_gib: float,
    oom_fallback: bool,
    selected_transformer_lr: float | None,
) -> list[str]:
    is_tuning = str(job["job_id"]).startswith("tuning__")
    seed = int(job["split_seed"])
    model = str(job["model_name"])
    if is_tuning:
        learning_rate = float(job["learning_rate"])
        q_low, q_high, feature_profile = 0.75, 0.90, "full"
        work_package = "stage5-tuning"
    else:
        if selected_transformer_lr is None:
            raise ValueError("Final Stage 5 command requires a locked Transformer learning rate")
        learning_rate = float(selected_transformer_lr)
        condition = CONDITIONS_BY_ID[str(job["condition_id"])]
        q_low, q_high, feature_profile = condition.q_low, condition.q_high, condition.feature_profile
        work_package = "stage5-final"
    if oom_fallback and model != "transformer":
        raise ValueError("OOM fallback is restricted to Transformer")
    batch_size = 128 if oom_fallback else int(defaults["batch_size"])
    accumulation = 2 if oom_fallback else 1
    run_dir = _run_dir(output, job, attempt)
    return [
        sys.executable,
        "ctsc/scripts/train_precursor_model.py",
        "--dataset",
        str(precursor_root / f"{job['task_name']}.pt"),
        "--output-dir",
        str(run_dir),
        "--model",
        MODEL_CLI[model],
        "--seed",
        str(seed),
        "--split-seed",
        str(seed),
        "--sampling-seed",
        str(seed),
        "--train-seed",
        str(seed),
        "--loader-seed",
        str(seed),
        "--split-family",
        "episode_group",
        "--device",
        str(device),
        "--feature-profile",
        feature_profile,
        "--window-size",
        "32",
        "--hidden-dim",
        "128",
        "--num-layers",
        "1",
        "--dropout",
        "0",
        "--cnn-kernel-size",
        str(int(defaults["cnn_kernel_size"])),
        "--mlp-depth",
        str(int(defaults["mlp_depth"])),
        "--transformer-num-layers",
        "2",
        "--transformer-num-heads",
        "4",
        "--transformer-ffn-multiplier",
        "4",
        "--transformer-max-length",
        "32",
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(accumulation),
        "--epochs",
        "12",
        "--lr",
        str(learning_rate),
        "--weight-decay",
        "1e-5",
        "--step-q-low",
        str(q_low),
        "--step-q-high",
        str(q_high),
        "--outcome-q-low",
        str(q_low),
        "--outcome-q-high",
        str(q_high),
        "--max-train-samples",
        str(int(defaults["max_train_samples"])),
        "--max-val-samples",
        str(int(defaults["max_val_samples"])),
        "--target-variant",
        "current_inclusive",
        "--gamma",
        "0.99",
        "--experiment-id",
        str(job["job_id"]),
        "--attempt-id",
        f"attempt-{attempt:03d}",
        "--work-package",
        work_package,
        "--minimum-free-space-gib",
        str(float(minimum_free_space_gib)),
    ]


def _execute_one(job_record: Mapping[str, Any], output: Path) -> dict[str, Any]:
    job = dict(job_record["job"])
    attempt = int(job_record["attempt"])
    command = list(job_record["command"])
    stem = f"{job['job_id']}__attempt-{attempt:03d}".replace("/", "_")
    stdout_path = output / "logs" / f"{stem}.out.log"
    stderr_path = output / "logs" / f"{stem}.err.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
    metrics_path = _run_dir(output, job, attempt) / "metrics.json"
    success = completed.returncode == 0 and metrics_path.is_file() and _valid_success(metrics_path, job)
    stderr_lower = completed.stderr.lower()
    return {
        "job_id": job["job_id"],
        "task_name": job["task_name"],
        "model_name": job["model_name"],
        "condition_id": job.get("condition_id", "tuning"),
        "learning_rate": job.get("learning_rate"),
        "split_seed": int(job["split_seed"]),
        "attempt_id": f"attempt-{attempt:03d}",
        "returncode": int(completed.returncode),
        "success": bool(success),
        "oom_detected": "out of memory" in stderr_lower or "cuda error: out of memory" in stderr_lower,
        "oom_fallback": bool(job_record["oom_fallback"]),
        "batch_size": 128 if job_record["oom_fallback"] else 256,
        "gradient_accumulation_steps": 2 if job_record["oom_fallback"] else 1,
        "metrics_path": str(metrics_path) if metrics_path.is_file() else "",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stderr_tail": completed.stderr[-1000:].replace("\r", " ").replace("\n", " ").strip(),
    }


def _execute(records: Sequence[Mapping[str, Any]], *, concurrency: int, output: Path) -> list[dict[str, Any]]:
    if not records:
        return []
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = {executor.submit(_execute_one, record, output): record for record in records}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({key: result[key] for key in ("job_id", "attempt_id", "success", "oom_detected")}), flush=True)
    return results


def _prepare_records(
    jobs: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    precursor_root: Path,
    defaults: Mapping[str, Any],
    device: str,
    minimum_free_space_gib: float,
    selected_transformer_lr: float | None,
    previous_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for job in jobs:
        existing = _existing_success(output, job)
        if existing is not None:
            skipped.append({"job_id": job["job_id"], "metrics_path": str(existing), "reason": "existing_valid_success"})
            continue
        prior = None if previous_results is None else previous_results.get(str(job["job_id"]))
        oom_fallback = bool(prior and prior.get("oom_detected") and str(job["model_name"]) == "transformer")
        attempt = _next_attempt(output, job)
        records.append(
            {
                "job": dict(job),
                "attempt": attempt,
                "oom_fallback": oom_fallback,
                "command": _job_command(
                    output=output,
                    precursor_root=precursor_root,
                    job=job,
                    attempt=attempt,
                    defaults=defaults,
                    device=device,
                    minimum_free_space_gib=minimum_free_space_gib,
                    oom_fallback=oom_fallback,
                    selected_transformer_lr=selected_transformer_lr,
                ),
            }
        )
    return records, skipped


def _run_with_retries(
    jobs: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    precursor_root: Path,
    defaults: Mapping[str, Any],
    device: str,
    minimum_free_space_gib: float,
    concurrency: int,
    retry_concurrency: int,
    selected_transformer_lr: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_results: list[dict[str, Any]] = []
    records, skipped = _prepare_records(
        jobs,
        output=output,
        precursor_root=precursor_root,
        defaults=defaults,
        device=device,
        minimum_free_space_gib=minimum_free_space_gib,
        selected_transformer_lr=selected_transformer_lr,
    )
    wave = _execute(records, concurrency=concurrency, output=output)
    all_results.extend(wave)
    failures = [row for row in wave if not row["success"]]
    for next_concurrency in (retry_concurrency, 1):
        if not failures:
            break
        failed_ids = {str(row["job_id"]) for row in failures}
        failed_jobs = [job for job in jobs if str(job["job_id"]) in failed_ids]
        prior = {str(row["job_id"]): row for row in failures}
        records, _ = _prepare_records(
            failed_jobs,
            output=output,
            precursor_root=precursor_root,
            defaults=defaults,
            device=device,
            minimum_free_space_gib=minimum_free_space_gib,
            selected_transformer_lr=selected_transformer_lr,
            previous_results=prior,
        )
        wave = _execute(records, concurrency=next_concurrency, output=output)
        all_results.extend(wave)
        failures = [row for row in wave if not row["success"]]
    return all_results, skipped, failures


def _collect_tuning_validation_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in tuning_jobs():
        metrics_path = _existing_success(output, job)
        if metrics_path is None:
            raise FileNotFoundError(f"Missing frozen tuning result: {job['job_id']}")
        metrics = _load_json(metrics_path)
        # Do not access the test object in this pre-lock function.
        validation = metrics["label_step"]["val"]["overall"]
        rows.append(
            {
                "task_name": job["task_name"],
                "model_name": job["model_name"],
                "learning_rate": float(job["learning_rate"]),
                "validation_step_catastrophic_auprc": float(validation["catastrophic_auprc"]),
                "metrics_path": str(metrics_path),
                "metrics_sha256": file_sha256(metrics_path),
            }
        )
    return rows







def main() -> None:
    args = parse_args()
    tasks, defaults = _registry()
    output = _resolve(args.output_dir)
    precursor_root = _resolve(args.precursor_source_dir)
    matrix = frozen_matrix_manifest(tasks)
    _guard_root(output, resume=bool(args.resume), matrix=matrix)
    jobs = tuning_jobs()
    if args.prepare_only:
        records, skipped = _prepare_records(jobs, output=output, precursor_root=precursor_root,
                                           defaults=defaults, device=args.device,
                                           minimum_free_space_gib=args.minimum_free_space_gib,
                                           selected_transformer_lr=None)
        write_json(output / "tuning_plan.json", {"jobs": records, "existing_successes": skipped})
        print(json.dumps({"jobs": len(jobs), "status": "prepared"}))
        return
    if args.smoke:
        jobs = jobs[:1]
        defaults.update(epochs=1, max_train_samples=1000, max_val_samples=500)
    results, skipped, failures = _run_with_retries(
        jobs, output=output, precursor_root=precursor_root, defaults=defaults,
        device=args.device, minimum_free_space_gib=args.minimum_free_space_gib,
        concurrency=args.concurrency, retry_concurrency=args.retry_concurrency,
        selected_transformer_lr=None)
    write_json(output / "tuning_run_status.json", {"attempts": results,
                                                 "existing_successes": skipped, "failures": failures})
    if failures:
        raise RuntimeError(f"Failed tuning jobs: {len(failures)}")
    if not args.smoke:
        validation_rows = _collect_tuning_validation_rows(output)
        scores, selection = select_learning_rates(validation_rows)
        write_csv(output / "tuning_validation_scores.csv", scores)
        selection_path = output / "learning_rate_selection.json"
        write_json(selection_path, selection)
        write_json(output / "learning_rate_selection.lock.json", {
            "selection_file_sha256": file_sha256(selection_path),
            "learning_rates": {m: v["selected_learning_rate"] for m, v in selection["models"].items()}})
    print(json.dumps({"jobs": len(jobs), "status": "completed", "smoke": args.smoke}))


if __name__ == "__main__":
    main()
