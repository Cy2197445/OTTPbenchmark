"""Run the target-robustness and cost-masking ablation matrix."""

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

import numpy as np

from ctsc.precursor.data import (
    FeatureMatrixSequenceDataset,
    build_feature_profile_matrix,
    build_splits,
    compute_profile_normalization_stats,
    feature_profile_current_cost_index,
    load_precursor_dataset,
)
from ctsc.precursor.revision import (
    arrays_sha256,
    build_label_sidecar,
    file_sha256,
    require_free_space,
)
from ctsc.precursor.stage4 import (
    CONDITIONS_BY_ID,
    FREEZE_ID,
    MODELS,
    NEW_CONDITIONS,
    PANEL_TASKS,
    SPLIT_SEEDS,
    FrozenCondition,
    build_current_reuse_parity_audit,
    condition_tasks,
    frozen_matrix_manifest,
    sidecar_name,
    write_json,
)
from ctsc.scripts.train_precursor_model import _cap_indices


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "work/target_ablation"
DEFAULT_PRECURSORS = REPO_ROOT / "work/precursors"
DEFAULT_G0 = REPO_ROOT / "work/ablation_reference"
REGISTRY = REPO_ROOT / "manifests/experiment_registry.json"
BASE_ARRAY_FIELDS = ("obs", "actions", "costs", "episode_uid", "episode_position", "env_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)))
    parser.add_argument("--precursor-source-dir", default=str(DEFAULT_PRECURSORS.relative_to(REPO_ROOT)))
    parser.add_argument("--g0-root", default=str(DEFAULT_G0.relative_to(REPO_ROOT)))
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(SPLIT_SEEDS))
    parser.add_argument("--condition-ids", nargs="*", default=None)
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


def _registry() -> tuple[list[str], dict[str, str], dict[str, Any]]:
    payload = _load_json(REGISTRY)
    tasks = [str(row["task_name"]) for row in payload["tasks"]]
    platforms = {str(row["task_name"]): str(row["source_platform"]) for row in payload["tasks"]}
    defaults = dict(payload["mainline_fresh_rerun"]["shared_defaults"])
    return tasks, platforms, defaults


def _guard_root(output: Path, *, resume: bool, matrix: Mapping[str, Any]) -> None:
    from ctsc.scripts.run_benchmark import output_directory
    output_directory(output)
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        matrix_path = output / "stage4_frozen_matrix.json"
        if not resume:
            raise FileExistsError(f"Stage 4 output is nonempty; pass --resume to continue: {output}")
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Resume root lacks frozen matrix manifest: {matrix_path}")
        existing = _load_json(matrix_path)
        if str(existing.get("matrix_sha256")) != str(matrix["matrix_sha256"]):
            raise ValueError("Existing Stage 4 matrix does not match the frozen matrix")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "stage4_frozen_matrix.json", matrix)


def _train_indices(dataset: dict[str, Any], *, split_seed: int, max_train_samples: int) -> np.ndarray:
    split = build_splits(dataset, seed=int(split_seed), split_family="episode_group")
    return _cap_indices(np.asarray(split["train"], dtype=np.int64), int(max_train_samples), seed=int(split_seed) + 11)


def _sidecar_math_audit(
    *,
    dataset: dict[str, Any],
    sidecar_path: Path,
    condition: FrozenCondition,
) -> dict[str, Any]:
    with np.load(sidecar_path, allow_pickle=False) as payload:
        values = np.asarray(payload["target_values"], dtype=np.float32)
    costs = np.asarray(dataset["costs"], dtype=np.float32)
    episodes = np.asarray(dataset["episode_uid"], dtype=np.int64)
    same_episode = episodes[:-1] == episodes[1:]
    terminal = np.concatenate([~same_episode, np.asarray([True])])
    if condition.target_variant == "future_only":
        recurrence = np.abs(values[:-1][same_episode] - (costs[1:][same_episode] + condition.gamma * values[1:][same_episode]))
        terminal_error = np.abs(values[terminal])
        base_current = np.asarray(dataset["cost_to_go_disc"], dtype=np.float32)
        base_alignment = np.abs(values[:-1][same_episode] - base_current[1:][same_episode])
    else:
        recurrence = np.abs(values[:-1][same_episode] - (costs[:-1][same_episode] + condition.gamma * values[1:][same_episode]))
        terminal_error = np.abs(values[terminal] - costs[terminal])
        base_alignment = np.asarray([], dtype=np.float32)
    max_recurrence = float(recurrence.max()) if recurrence.size else 0.0
    max_terminal = float(terminal_error.max()) if terminal_error.size else 0.0
    max_base_alignment = float(base_alignment.max()) if base_alignment.size else None
    tolerance = 2e-4
    status = max_recurrence <= tolerance and max_terminal <= tolerance
    if max_base_alignment is not None:
        status = status and max_base_alignment <= tolerance
    return {
        "target_variant": condition.target_variant,
        "gamma": condition.gamma,
        "q_low": condition.q_low,
        "q_high": condition.q_high,
        "transition_count": int(values.size),
        "episode_count": int(np.unique(episodes).size),
        "max_recurrence_abs_error": max_recurrence,
        "max_terminal_abs_error": max_terminal,
        "max_future_vs_base_next_current_abs_error": max_base_alignment,
        "tolerance": tolerance,
        "status": "PASS" if status else "FAIL",
    }


def _prepare_sidecars(
    *,
    output: Path,
    precursor_root: Path,
    tasks: Sequence[str],
    conditions: Sequence[FrozenCondition],
    max_train_samples: int,
    minimum_free_space_gib: float,
) -> tuple[dict[tuple[str, str], Path], list[dict[str, Any]]]:
    required: dict[tuple[str, str], FrozenCondition] = {}
    for condition in conditions:
        name = sidecar_name(condition)
        if name is None:
            continue
        for task_name in condition_tasks(condition, tasks):
            if task_name in tasks:
                required[(task_name, name)] = condition
    by_task: dict[str, list[tuple[str, FrozenCondition]]] = {}
    for (task_name, name), condition in required.items():
        by_task.setdefault(task_name, []).append((name, condition))

    paths: dict[tuple[str, str], Path] = {}
    audits: list[dict[str, Any]] = []
    for task_name in sorted(by_task):
        require_free_space(output, minimum_gib=minimum_free_space_gib)
        dataset_path = precursor_root / f"{task_name}.pt"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Missing precursor dataset: {dataset_path}")
        dataset = load_precursor_dataset(dataset_path)
        dataset_digest = file_sha256(dataset_path)
        arrays_digest = arrays_sha256(dataset, BASE_ARRAY_FIELDS)
        train_indices = _train_indices(dataset, split_seed=0, max_train_samples=max_train_samples)
        for name, condition in sorted(by_task[task_name], key=lambda item: item[0]):
            path = output / "sidecars" / task_name / name
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            if path.is_file() and manifest_path.is_file():
                manifest = _load_json(manifest_path)
                if (
                    str(manifest.get("base_dataset_sha256")) != dataset_digest
                    or str(manifest.get("base_arrays_sha256")) != arrays_digest
                    or str(manifest.get("target_variant")) != condition.target_variant
                    or not math.isclose(float(manifest.get("gamma")), condition.gamma, abs_tol=1e-12)
                ):
                    raise ValueError(f"Existing sidecar does not match frozen condition: {path}")
            else:
                if path.exists() or manifest_path.exists():
                    raise FileExistsError(f"Partial sidecar collision: {path}")
                manifest = build_label_sidecar(
                    dataset=dataset,
                    base_dataset_path=dataset_path,
                    train_indices=train_indices,
                    output_path=path,
                    target_variant=condition.target_variant,
                    gamma=condition.gamma,
                    quantile_pair=(condition.q_low, condition.q_high),
                    precomputed_base_dataset_sha256=dataset_digest,
                    precomputed_base_arrays_sha256=arrays_digest,
                )
            math_audit = _sidecar_math_audit(dataset=dataset, sidecar_path=path, condition=condition)
            if math_audit["status"] != "PASS":
                raise ValueError(f"Sidecar mathematical audit failed: {path}: {math_audit}")
            paths[(task_name, condition.condition_id)] = path
            audits.append(
                {
                    "task_name": task_name,
                    "condition_id": condition.condition_id,
                    "sidecar_path": str(path),
                    "sidecar_sha256": file_sha256(path),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": file_sha256(manifest_path),
                    "base_dataset_sha256": manifest["base_dataset_sha256"],
                    "base_arrays_sha256": manifest["base_arrays_sha256"],
                    "contains_base_observation_or_action_arrays": manifest[
                        "contains_base_observation_or_action_arrays"
                    ],
                    **math_audit,
                }
            )
        del dataset
    for condition in conditions:
        name = sidecar_name(condition)
        if name is None:
            continue
        for task_name in condition_tasks(condition, tasks):
            if task_name in tasks:
                path = output / "sidecars" / task_name / name
                if not path.is_file():
                    raise FileNotFoundError(f"Prepared sidecar alias target is missing: {path}")
                paths[(task_name, condition.condition_id)] = path
    write_json(
        output / "sidecar_build_audit.json",
        {
            "schema_version": "ottp_stage4_sidecar_build_audit_v1",
            "status": "PASS",
            "sidecar_count": len(audits),
            "rows": audits,
        },
    )
    return paths, audits


def _masked_feature_audit(precursor_path: Path, output: Path) -> dict[str, Any]:
    dataset = load_precursor_dataset(precursor_path)
    features = build_feature_profile_matrix(dataset, profile_name="full")
    split = build_splits(dataset, seed=0, split_family="episode_group")
    stats = compute_profile_normalization_stats(dataset, np.asarray(split["train"]), "full", features)
    normalized = ((features - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)
    cost_index = feature_profile_current_cost_index(dataset, "full")
    positions = np.asarray(dataset["episode_position"], dtype=np.int64)
    candidates = np.flatnonzero((positions >= 1) & (np.abs(normalized[:, int(cost_index)]) > 1e-7))
    if candidates.size == 0:
        raise ValueError("Could not find a nonzero current-cost audit sample")
    sample_index = int(candidates[0])
    labels = np.zeros(normalized.shape[0], dtype=np.int64)
    kwargs = {
        "features": normalized,
        "dataset": dataset,
        "indices": np.asarray([sample_index], dtype=np.int64),
        "labels": labels,
        "window_size": 32,
        "shuffle_seed": 0,
        "current_cost_index": cost_index,
    }
    full = FeatureMatrixSequenceDataset(sequence_view="full", **kwargs)[0]["sequence"].numpy()
    masked = FeatureMatrixSequenceDataset(sequence_view="history_without_current_cost", **kwargs)[0]["sequence"].numpy()
    changed = np.argwhere(full != masked)
    expected = np.asarray([[31, int(cost_index)]], dtype=np.int64)
    status = np.array_equal(changed, expected) and float(masked[31, int(cost_index)]) == 0.0
    payload = {
        "schema_version": "ottp_stage4_masked_feature_audit_v1",
        "task_name": str(dataset["metadata"]["task_name"]),
        "sample_index": sample_index,
        "window_size": 32,
        "current_cost_feature_index": int(cost_index),
        "changed_cell_count": int(changed.shape[0]),
        "changed_cells": changed.tolist(),
        "all_non_current_cost_features_byte_identical": bool(
            np.array_equal(np.delete(full, int(cost_index), axis=1), np.delete(masked, int(cost_index), axis=1))
        ),
        "masked_current_cost_value": float(masked[31, int(cost_index)]),
        "status": "PASS" if status else "FAIL",
    }
    if not status:
        raise ValueError(f"Masked feature audit failed: {payload}")
    write_json(output / "masked_feature_audit.json", payload)
    return payload


def _attempt_root(output: Path, condition: FrozenCondition, task_name: str, split_seed: int, attempt: int) -> Path:
    return (
        output
        / "task_runs"
        / condition.matrix
        / task_name
        / condition.condition_id
        / f"attempt-{attempt:03d}"
        / f"split_seed_{split_seed}"
        / f"train_seed_{split_seed}"
    )


def _existing_success(output: Path, condition: FrozenCondition, task_name: str, split_seed: int) -> Path | None:
    base = output / "task_runs" / condition.matrix / task_name / condition.condition_id
    candidates = sorted(base.glob(f"attempt-*/split_seed_{split_seed}/train_seed_{split_seed}/suite_summary.json"))
    return candidates[-1] if candidates else None


def _next_attempt(output: Path, condition: FrozenCondition, task_name: str, split_seed: int) -> tuple[int, Path]:
    for attempt in range(1, 1000):
        path = _attempt_root(output, condition, task_name, split_seed, attempt)
        if not path.exists():
            return attempt, path
    raise RuntimeError("Exhausted Stage 4 attempt identifiers")


def _job_command(
    *,
    output: Path,
    precursor_root: Path,
    sidecars: Mapping[tuple[str, str], Path],
    condition: FrozenCondition,
    task_name: str,
    split_seed: int,
    attempt: int,
    suite_dir: Path,
    defaults: Mapping[str, Any],
    device: str,
    minimum_free_space_gib: float,
) -> list[str]:
    command = [
        sys.executable,
        "ctsc/scripts/run_precursor_baseline_suite.py",
        "--dataset",
        str(precursor_root / f"{task_name}.pt"),
        "--output-dir",
        str(suite_dir),
        "--slice-name",
        "canonical" if condition.q_low == 0.75 and condition.q_high == 0.90 else condition.condition_id,
        "--seed",
        str(split_seed),
        "--split-seed",
        str(split_seed),
        "--sampling-seed",
        str(split_seed),
        "--train-seed",
        str(split_seed),
        "--loader-seed",
        str(split_seed),
        "--split-family",
        "episode_group",
        "--device",
        device,
        "--models",
        "gru",
        "flat-mlp",
        "--feature-profile",
        condition.feature_profile,
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
        str(int(defaults["batch_size"])),
        "--epochs",
        str(int(defaults["epochs"])),
        "--lr",
        str(float(defaults["lr"])),
        "--weight-decay",
        str(float(defaults["weight_decay"])),
        "--step-q-low",
        str(condition.q_low),
        "--step-q-high",
        str(condition.q_high),
        "--outcome-q-low",
        str(condition.q_low),
        "--outcome-q-high",
        str(condition.q_high),
        "--max-train-samples",
        str(int(defaults["max_train_samples"])),
        "--max-val-samples",
        str(int(defaults["max_val_samples"])),
        "--target-variant",
        condition.target_variant,
        "--gamma",
        str(condition.gamma),
        "--attempt-id",
        f"attempt-{attempt:03d}",
        "--work-package",
        f"stage4-{condition.matrix}",
        "--minimum-free-space-gib",
        str(float(minimum_free_space_gib)),
    ]
    sidecar = sidecars.get((task_name, condition.condition_id))
    if sidecar is not None:
        command.extend(["--label-sidecar", str(sidecar)])
    return command


def _run_job(job: Mapping[str, Any], *, output: Path) -> dict[str, Any]:
    completed = subprocess.run(job["command"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['job_id']}__{job['attempt_id']}"
    (log_dir / f"{stem}.out.log").write_text(completed.stdout or "", encoding="utf-8")
    (log_dir / f"{stem}.err.log").write_text(completed.stderr or "", encoding="utf-8")
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    return {
        "job_id": job["job_id"],
        "condition_id": job["condition_id"],
        "task_name": job["task_name"],
        "split_seed": job["split_seed"],
        "attempt_id": job["attempt_id"],
        "suite_dir": str(job["suite_dir"]),
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "oom_detected": "out of memory" in combined.lower(),
        "stderr_tail": (completed.stderr or "")[-1000:],
    }


def _execute(jobs: Sequence[Mapping[str, Any]], *, concurrency: int, output: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not jobs:
        return results
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = {executor.submit(_run_job, job, output=output): job for job in jobs}
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed_count += 1
            print(
                json.dumps(
                    {
                        "progress": f"{completed_count}/{len(jobs)}",
                        "job_id": result["job_id"],
                        "attempt": result["attempt_id"],
                        "success": result["success"],
                        "oom": result["oom_detected"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return results


def _make_jobs(
    *,
    output: Path,
    precursor_root: Path,
    sidecars: Mapping[tuple[str, str], Path],
    tasks: Sequence[str],
    split_seeds: Sequence[int],
    conditions: Sequence[FrozenCondition],
    defaults: Mapping[str, Any],
    device: str,
    minimum_free_space_gib: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for condition in conditions:
        for task_name in condition_tasks(condition, tasks):
            if task_name not in tasks:
                continue
            for split_seed in split_seeds:
                job_id = f"{condition.condition_id}__{task_name}__ss{split_seed}"
                existing = _existing_success(output, condition, task_name, split_seed)
                if existing is not None:
                    skipped.append({"job_id": job_id, "suite_summary": str(existing), "status": "existing_success"})
                    continue
                attempt, suite_dir = _next_attempt(output, condition, task_name, split_seed)
                jobs.append(
                    {
                        "job_id": job_id,
                        "condition_id": condition.condition_id,
                        "task_name": task_name,
                        "split_seed": int(split_seed),
                        "attempt_id": f"attempt-{attempt:03d}",
                        "attempt_number": attempt,
                        "suite_dir": suite_dir,
                        "command": _job_command(
                            output=output,
                            precursor_root=precursor_root,
                            sidecars=sidecars,
                            condition=condition,
                            task_name=task_name,
                            split_seed=int(split_seed),
                            attempt=attempt,
                            suite_dir=suite_dir,
                            defaults=defaults,
                            device=device,
                            minimum_free_space_gib=minimum_free_space_gib,
                        ),
                    }
                )
    return jobs, skipped


def _retry_jobs(
    failed: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    precursor_root: Path,
    sidecars: Mapping[tuple[str, str], Path],
    defaults: Mapping[str, Any],
    device: str,
    minimum_free_space_gib: float,
) -> list[dict[str, Any]]:
    retries: list[dict[str, Any]] = []
    for row in failed:
        condition = CONDITIONS_BY_ID[str(row["condition_id"])]
        task_name = str(row["task_name"])
        split_seed = int(row["split_seed"])
        attempt, suite_dir = _next_attempt(output, condition, task_name, split_seed)
        retries.append(
            {
                "job_id": row["job_id"],
                "condition_id": condition.condition_id,
                "task_name": task_name,
                "split_seed": split_seed,
                "attempt_id": f"attempt-{attempt:03d}",
                "attempt_number": attempt,
                "suite_dir": suite_dir,
                "command": _job_command(
                    output=output,
                    precursor_root=precursor_root,
                    sidecars=sidecars,
                    condition=condition,
                    task_name=task_name,
                    split_seed=split_seed,
                    attempt=attempt,
                    suite_dir=suite_dir,
                    defaults=defaults,
                    device=device,
                    minimum_free_space_gib=minimum_free_space_gib,
                ),
            }
        )
    return retries


def main() -> None:
    args = parse_args()
    all_tasks, _, defaults = _registry()
    matrix = frozen_matrix_manifest(all_tasks)
    output = _resolve(args.output_dir)
    precursor_root = _resolve(args.precursor_source_dir)
    g0_root = _resolve(args.g0_root)
    _guard_root(output, resume=bool(args.resume), matrix=matrix)
    require_free_space(output, minimum_gib=float(args.minimum_free_space_gib))

    if args.smoke:
        selected_tasks = ["SwimmerVelocity"]
        split_seeds = [0]
        conditions = [condition for condition in NEW_CONDITIONS if condition.matrix == "core"]
    else:
        selected_tasks = list(dict.fromkeys(args.tasks or all_tasks))
        unknown = sorted(set(selected_tasks) - set(all_tasks))
        if unknown:
            raise ValueError(f"Unknown Stage 4 tasks: {unknown}")
        split_seeds = list(dict.fromkeys(int(seed) for seed in args.split_seeds))
        if not set(split_seeds).issubset(SPLIT_SEEDS):
            raise ValueError(f"Split seeds must be a subset of {SPLIT_SEEDS}")
        condition_ids = set(args.condition_ids or [condition.condition_id for condition in NEW_CONDITIONS])
        unknown_conditions = sorted(condition_ids - set(CONDITIONS_BY_ID))
        if unknown_conditions:
            raise ValueError(f"Unknown Stage 4 conditions: {unknown_conditions}")
        conditions = [condition for condition in NEW_CONDITIONS if condition.condition_id in condition_ids]

    sidecars, sidecar_audits = _prepare_sidecars(
        output=output,
        precursor_root=precursor_root,
        tasks=selected_tasks,
        conditions=conditions,
        max_train_samples=int(defaults["max_train_samples"]),
        minimum_free_space_gib=float(args.minimum_free_space_gib),
    )
    mask_task = "SwimmerVelocity" if "SwimmerVelocity" in selected_tasks else selected_tasks[0]
    mask_audit = _masked_feature_audit(precursor_root / f"{mask_task}.pt", output)

    jobs, skipped = _make_jobs(
        output=output,
        precursor_root=precursor_root,
        sidecars=sidecars,
        tasks=selected_tasks,
        split_seeds=split_seeds,
        conditions=conditions,
        defaults=defaults,
        device=str(args.device),
        minimum_free_space_gib=float(args.minimum_free_space_gib),
    )
    if args.prepare_only:
        write_json(
            output / "stage4_prepare_status.json",
            {
                "status": "PASS",
                "selected_tasks": selected_tasks,
                "conditions": [condition.condition_id for condition in conditions],
                "sidecar_count": len(sidecar_audits),
                "masked_feature_audit": mask_audit,
                "pending_job_count": len(jobs),
                "existing_success_count": len(skipped),
            },
        )
        print(json.dumps({"status": "PREPARED", "pending_job_count": len(jobs)}, indent=2))
        return

    all_results = _execute(jobs, concurrency=int(args.concurrency), output=output)
    failures = [row for row in all_results if not row["success"]]
    if failures:
        retry_jobs = _retry_jobs(
            failures,
            output=output,
            precursor_root=precursor_root,
            sidecars=sidecars,
            defaults=defaults,
            device=str(args.device),
            minimum_free_space_gib=float(args.minimum_free_space_gib),
        )
        retry_results = _execute(retry_jobs, concurrency=int(args.retry_concurrency), output=output)
        all_results.extend(retry_results)
        failures = [row for row in retry_results if not row["success"]]
    if failures:
        final_jobs = _retry_jobs(
            failures,
            output=output,
            precursor_root=precursor_root,
            sidecars=sidecars,
            defaults=defaults,
            device=str(args.device),
            minimum_free_space_gib=float(args.minimum_free_space_gib),
        )
        final_results = _execute(final_jobs, concurrency=1, output=output)
        all_results.extend(final_results)
        failures = [row for row in final_results if not row["success"]]

    selected_expected = len(conditions) * len(split_seeds)
    selected_expected = sum(len([task for task in condition_tasks(condition, all_tasks) if task in selected_tasks]) for condition in conditions) * len(split_seeds)
    completed = len(skipped) + sum(bool(row["success"]) for row in all_results if row["attempt_id"] == max(
        [candidate["attempt_id"] for candidate in all_results if candidate["job_id"] == row["job_id"]]
    ))
    status_payload = {
        "schema_version": "ottp_stage4_run_status_v1",
        "freeze_id": FREEZE_ID,
        "status": "FAIL" if failures else "PASS",
        "smoke": bool(args.smoke),
        "selected_tasks": selected_tasks,
        "split_seeds": split_seeds,
        "conditions": [condition.condition_id for condition in conditions],
        "expected_suite_jobs": selected_expected,
        "newly_attempted_suite_jobs": len(jobs),
        "existing_success_suite_jobs": len(skipped),
        "persistent_failure_count": len(failures),
        "results": all_results,
        "skipped": skipped,
        "masked_feature_audit": mask_audit,
        "sidecar_audit_count": len(sidecar_audits),
    }
    status_name = "stage4_smoke_run_status.json" if args.smoke else "stage4_full_run_status.json"
    write_json(output / status_name, status_payload)
    if failures:
        raise RuntimeError(f"Persistent Stage 4 failures: {json.dumps(failures, ensure_ascii=False)}")

    if args.smoke:
        parity = build_current_reuse_parity_audit(
            experiment_root=output,
            g0_root=g0_root,
            tasks=["SwimmerVelocity"],
            split_seeds=[0],
        )
        smoke_payload = {
            "schema_version": "ottp_stage4_smoke_gate_v1",
            "status": "PASS",
            "future_only_sidecar_math": "PASS",
            "masked_feature_isolation": mask_audit["status"],
            "current_inclusive_canonical_preprocessing_parity": parity[0][
                "current_inclusive_canonical_reuse_status"
            ],
            "suite_job_count": 3,
            "model_result_count": 6,
            "task": "SwimmerVelocity",
            "split_seed": 0,
        }
        write_json(output / "stage4_smoke_gate.json", smoke_payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "smoke": bool(args.smoke),
                "selected_expected_suite_jobs": selected_expected,
                "new_jobs": len(jobs),
                "reused_completed_jobs": len(skipped),
                "output_dir": str(output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
