'Audit the current-branch Unified38 fresh-rerun result layer.'

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "work" / "reference_diagnostics"
DEFAULT_REGISTRY = REPO_ROOT / "manifests" / "experiment_registry.json"
EXPECTED_MODELS = ["gru", "temporal-cnn", "flat-mlp", "linear", "prevalence"]
SEQUENCE_MODELS = ["gru", "temporal-cnn"]
NONSEQUENTIAL_MODELS = ["flat-mlp", "linear"]
REQUIRED_METRIC_KEYS = [
    "model_name",
    "model_family",
    "input_view",
    "feature_profile",
    "selection_mode",
    "split_sizes",
    "label_protocol",
    "label_step",
    "label_outcome",
    "critic_baseline",
    "training",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=str, default=str(DEFAULT_RUN_ROOT.relative_to(REPO_ROOT)))
    parser.add_argument("--registry", type=str, default=str(DEFAULT_REGISTRY.relative_to(REPO_ROOT)))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--precursor-dir", type=str, default=None)
    parser.add_argument("--task-names", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--slices", type=str, nargs="+", default=None)
    parser.add_argument("--feature-profile", type=str, default="full")
    parser.add_argument("--expected-models", type=str, nargs="+", default=None)
    parser.add_argument(
        "--hash-policy",
        type=str,
        choices=["small", "all-json", "none"],
        default="small",
        help="Hash JSON/CSV/MD audit-critical files by default; model bundles are size-recorded only.",
    )
    return parser.parse_args()


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_hash(path: Path, policy: str) -> bool:
    if policy == "none":
        return False
    if policy == "all-json":
        return path.suffix.lower() in {".json", ".csv", ".md", ".txt", ".log"}
    if path.suffix.lower() in {".json", ".csv", ".md", ".txt", ".log"}:
        return path.stat().st_size <= 20 * 1024 * 1024
    return False


def _file_record(path: Path, *, hash_policy: str, role: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": _repo_rel(path),
        "exists": path.exists(),
    }
    if path.exists():
        record["size_bytes"] = int(path.stat().st_size)
        record["sha256"] = _sha256(path) if _should_hash(path, hash_policy) else None
    return record


def _path_from_payload(value: str | None, *, base: Path) -> Path:
    if not value:
        return base
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _isclose(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def _expected_from_registry(registry: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str], list[int], int]:
    task_names = (
        list(dict.fromkeys(str(name) for name in args.task_names))
        if args.task_names
        else [str(task["task_name"]) for task in registry["tasks"]]
    )
    rerun = registry["mainline_fresh_rerun"]
    slices = list(args.slices) if args.slices is not None else [str(row["name"]) for row in rerun["slices"]]
    split_seeds = list(args.split_seeds) if args.split_seeds is not None else [int(seed) for seed in rerun["split_seeds"]]
    expected_epochs = int(rerun["shared_defaults"]["epochs"])
    return task_names, slices, split_seeds, expected_epochs


def _validate_suite(
    *,
    suite_path: Path,
    task_name: str,
    slice_name: str,
    split_seed: int,
    expected_epochs: int,
    expected_feature_profile: str,
    expected_models: list[str],
    hash_policy: str,
    artifacts: list[dict[str, Any]],
) -> tuple[int, int, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics_count = 0
    bundle_count = 0
    artifacts.append(_file_record(suite_path, hash_policy=hash_policy, role="suite_summary"))
    if not suite_path.exists():
        return metrics_count, bundle_count, [f"missing suite_summary: {_repo_rel(suite_path)}"], warnings

    suite = _load_json(suite_path)
    if str(suite.get("slice_name")) != str(slice_name):
        errors.append(f"{task_name}/{slice_name}/seed_{split_seed}: suite slice mismatch")
    if int(suite.get("seed", -1)) != int(split_seed):
        errors.append(f"{task_name}/{slice_name}/seed_{split_seed}: suite seed mismatch")
    if str(suite.get("feature_profile")) != str(expected_feature_profile):
        errors.append(
            f"{task_name}/{slice_name}/seed_{split_seed}: feature_profile is not {expected_feature_profile}"
        )
    models = suite.get("models", {})
    if sorted(models.keys()) != sorted(expected_models):
        errors.append(f"{task_name}/{slice_name}/seed_{split_seed}: model set mismatch {sorted(models.keys())}")

    test_step: dict[str, float] = {}
    val_step: dict[str, float] = {}
    for model_name in expected_models:
        model_payload = models.get(model_name)
        if not isinstance(model_payload, dict):
            continue
        metric_path = _path_from_payload(str(model_payload.get("metrics_path", "")), base=suite_path.parent)
        bundle_path = _path_from_payload(str(model_payload.get("bundle_path", "")), base=suite_path.parent)
        artifacts.append(_file_record(metric_path, hash_policy=hash_policy, role="metrics_json"))
        artifacts.append(_file_record(bundle_path, hash_policy="none", role="model_bundle"))
        if not metric_path.exists():
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: missing metrics.json")
            continue
        if not bundle_path.exists():
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: missing precursor_model.pt")
        else:
            bundle_count += 1
        metrics_count += 1
        metrics = _load_json(metric_path)
        for key in REQUIRED_METRIC_KEYS:
            if key not in metrics:
                errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: missing metrics key `{key}`")
        if str(metrics.get("model_name")) != model_name:
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: model_name mismatch")
        if str(metrics.get("feature_profile")) != str(expected_feature_profile):
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: metrics feature_profile mismatch")
        try:
            metric_step = float(metrics["label_step"]["test"]["overall"]["catastrophic_auprc"])
            metric_val_step = float(metrics["label_step"]["val"]["overall"]["catastrophic_auprc"])
        except KeyError:
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: incomplete label_step metrics")
            continue
        summary_step = float(model_payload.get("test_step_catastrophic_auprc", float("nan")))
        summary_val_step = float(model_payload.get("val_step_catastrophic_auprc", float("nan")))
        if not _finite(metric_step) or not _finite(summary_step):
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: non-finite test AUPRC")
        elif not _isclose(metric_step, summary_step):
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: suite/metrics test AUPRC mismatch")
        if not _finite(metric_val_step) or not _finite(summary_val_step):
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: non-finite val AUPRC")
        elif not _isclose(metric_val_step, summary_val_step):
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: suite/metrics val AUPRC mismatch")
        training = metrics.get("training", {})
        if model_name == "prevalence":
            if int(training.get("best_epoch", -999)) != -1 or training.get("history", []) != []:
                errors.append(f"{task_name}/{slice_name}/seed_{split_seed}/prevalence: reference training contract mismatch")
        else:
            history = training.get("history", [])
            if not isinstance(history, list) or len(history) != expected_epochs:
                warnings.append(
                    f"{task_name}/{slice_name}/seed_{split_seed}/{model_name}: history length {len(history) if isinstance(history, list) else 'invalid'} != {expected_epochs}"
                )
        test_step[model_name] = metric_step
        val_step[model_name] = metric_val_step

    if all(name in val_step for name in SEQUENCE_MODELS):
        best_by_val = max(SEQUENCE_MODELS, key=lambda name: val_step[name])
        if str(suite.get("best_sequence_baseline")) != best_by_val:
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}: best_sequence_baseline mismatch")
    if all(name in test_step for name in ["flat-mlp", "linear", str(suite.get("best_sequence_baseline"))]):
        best_name = str(suite["best_sequence_baseline"])
        expected_gap_flat = test_step[best_name] - test_step["flat-mlp"]
        expected_gap_linear = test_step[best_name] - test_step["linear"]
        if not _isclose(float(suite.get("gap_vs_flat")), expected_gap_flat):
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}: gap_vs_flat mismatch")
        if not _isclose(float(suite.get("gap_vs_linear")), expected_gap_linear):
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}: gap_vs_linear mismatch")
        strict = all(test_step[seq] > test_step[nonseq] for seq in SEQUENCE_MODELS for nonseq in NONSEQUENTIAL_MODELS)
        if bool(suite.get("strict_task_level_sequence_support")) != bool(strict):
            errors.append(f"{task_name}/{slice_name}/seed_{split_seed}: strict support flag mismatch")

    return metrics_count, bundle_count, errors, warnings


def _write_markdown_report(path: Path, summary: dict[str, Any], task_rows: list[dict[str, Any]]) -> None:
    rows = [
        "# Unified38 Current Rerun Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- completed tasks: `{summary['completed_task_count']}/{summary['expected_task_count']}`",
        f"- suite summaries: `{summary['observed_suite_count']}/{summary['expected_suite_count']}`",
        f"- metrics files: `{summary['observed_metrics_count']}/{summary['expected_metrics_count']}`",
        f"- model bundles: `{summary['observed_bundle_count']}/{summary['expected_metrics_count']}`",
        f"- aggregate row check: `{summary['aggregate_status']}`",
        f"- issue counts: errors `{summary['error_count']}`, warnings `{summary['warning_count']}`",
        "",
        "| Task | Suites | Metrics | Bundles | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in task_rows:
        rows.append(
            f"| {row['task_name']} | {row['suite_count']} | {row['metrics_count']} | {row['bundle_count']} | {row['status']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = _resolve_repo_path(args.run_root)
    registry_path = _resolve_repo_path(args.registry)
    output_dir = _resolve_repo_path(args.output_dir) if args.output_dir else run_root / "audit"
    registry = _load_json(registry_path)
    task_names, slices, split_seeds, expected_epochs = _expected_from_registry(registry, args)
    expected_models = (
        list(dict.fromkeys(str(model) for model in args.expected_models))
        if args.expected_models
        else list(EXPECTED_MODELS)
    )
    expected_feature_profile = str(args.feature_profile)

    task_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = [
        _file_record(registry_path, hash_policy=args.hash_policy, role="fresh_rerun_registry"),
    ]
    errors: list[str] = []
    warnings: list[str] = []
    observed_suite_count = 0
    observed_metrics_count = 0
    observed_bundle_count = 0

    precursor_dir = _resolve_repo_path(args.precursor_dir) if args.precursor_dir else run_root / "precursor_datasets"
    task_run_dir = run_root / "task_runs"
    aggregate_dir = run_root / "aggregates"
    aggregate_summary = aggregate_dir / "Unified38_current_run_summary.json"
    task_registry = aggregate_dir / "unified_task_registry_current.csv"
    seed_model_table = aggregate_dir / "unified_seed_model_table_current.csv"
    for aggregate_path, role in [
        (aggregate_summary, "aggregate_summary"),
        (task_registry, "current_task_registry"),
        (seed_model_table, "current_seed_model_table"),
    ]:
        artifacts.append(_file_record(aggregate_path, hash_policy=args.hash_policy, role=role))

    aggregate_status = "pass"
    if not aggregate_summary.exists() or not task_registry.exists() or not seed_model_table.exists():
        aggregate_status = "missing"
        errors.append("missing aggregate summary/table output")
    else:
        summary_payload = _load_json(aggregate_summary)
        if int(summary_payload.get("completed_task_count", -1)) != len(task_names):
            aggregate_status = "fail"
            errors.append("aggregate completed_task_count does not match expected task count")

    for task_name in task_names:
        task_errors: list[str] = []
        task_warnings: list[str] = []
        precursor_summary = precursor_dir / f"{task_name}.summary.json"
        precursor_pt = precursor_dir / f"{task_name}.pt"
        artifacts.append(_file_record(precursor_summary, hash_policy=args.hash_policy, role="precursor_summary"))
        artifacts.append(_file_record(precursor_pt, hash_policy="none", role="precursor_tensor"))
        if not precursor_summary.exists():
            task_errors.append(f"{task_name}: missing precursor summary")
        if not precursor_pt.exists():
            task_errors.append(f"{task_name}: missing precursor tensor")
        suite_count = 0
        metrics_count = 0
        bundle_count = 0
        for slice_name in slices:
            for split_seed in split_seeds:
                suite_path = task_run_dir / task_name / slice_name / f"seed_{int(split_seed)}" / "suite_summary.json"
                if suite_path.exists():
                    suite_count += 1
                suite_metrics, suite_bundles, suite_errors, suite_warnings = _validate_suite(
                    suite_path=suite_path,
                    task_name=task_name,
                    slice_name=slice_name,
                    split_seed=int(split_seed),
                    expected_epochs=expected_epochs,
                    expected_feature_profile=expected_feature_profile,
                    expected_models=expected_models,
                    hash_policy=args.hash_policy,
                    artifacts=artifacts,
                )
                metrics_count += suite_metrics
                bundle_count += suite_bundles
                task_errors.extend(suite_errors)
                task_warnings.extend(suite_warnings)
        observed_suite_count += suite_count
        observed_metrics_count += metrics_count
        observed_bundle_count += bundle_count
        status = "pass" if not task_errors else "fail"
        task_rows.append(
            {
                "task_name": task_name,
                "suite_count": suite_count,
                "expected_suite_count": len(slices) * len(split_seeds),
                "metrics_count": metrics_count,
                "expected_metrics_count": len(slices) * len(split_seeds) * len(expected_models),
                "bundle_count": bundle_count,
                "status": status,
                "warning_count": len(task_warnings),
            }
        )
        errors.extend(task_errors)
        warnings.extend(task_warnings)

    expected_suite_count = len(task_names) * len(slices) * len(split_seeds)
    expected_metrics_count = expected_suite_count * len(expected_models)
    if observed_suite_count != expected_suite_count:
        errors.append(f"suite count mismatch: {observed_suite_count}/{expected_suite_count}")
    if observed_metrics_count != expected_metrics_count:
        errors.append(f"metrics count mismatch: {observed_metrics_count}/{expected_metrics_count}")
    if observed_bundle_count != expected_metrics_count:
        errors.append(f"bundle count mismatch: {observed_bundle_count}/{expected_metrics_count}")

    summary = {
        "title": "Unified38 Current Rerun Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": _repo_rel(run_root),
        "precursor_dir": _repo_rel(precursor_dir),
        "registry": _repo_rel(registry_path),
        "status": "pass" if not errors else "fail",
        "expected_task_count": len(task_names),
        "completed_task_count": sum(1 for row in task_rows if row["status"] == "pass"),
        "expected_suite_count": expected_suite_count,
        "observed_suite_count": observed_suite_count,
        "expected_metrics_count": expected_metrics_count,
        "observed_metrics_count": observed_metrics_count,
        "observed_bundle_count": observed_bundle_count,
        "aggregate_status": aggregate_status,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "hash_policy": args.hash_policy,
        "models": list(expected_models),
        "feature_profile": expected_feature_profile,
        "slices": list(slices),
        "split_seeds": list(split_seeds),
        "audit_level": "current_full_rerun_contract_verified" if not errors else "current_full_rerun_contract_failed",
        "notes": [
            "Model bundles and precursor tensors are presence/size recorded by default; JSON/CSV/MD evidence files are hashed under the default small hash policy.",
            "This audit verifies the current fresh-rerun result contract and aggregation coverage; it is not a statistical rerun match against archived numbers.",
        ],
        "errors": errors[:200],
        "warnings": warnings[:200],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "Unified38_current_rerun_audit_summary.json", summary)
    _write_csv(output_dir / "Unified38_current_rerun_audit_table.csv", task_rows)
    _write_json(output_dir / "Unified38_current_rerun_artifact_manifest.json", {"artifacts": artifacts})
    _write_markdown_report(output_dir / "Unified38_current_rerun_audit_report.md", summary, task_rows)
    print(json.dumps({"audit_summary": str(output_dir / "Unified38_current_rerun_audit_summary.json"), "status": summary["status"]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
