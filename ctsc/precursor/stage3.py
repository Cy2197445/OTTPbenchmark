'Read-only Stage 3 reanalysis over frozen Unified38 result assets.'

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FULL_MODELS = ("gru", "temporal_cnn", "flat_mlp", "linear", "prevalence")
TRAINABLE_MODELS = ("gru", "temporal_cnn", "flat_mlp", "linear")
FIXED_PRIMARY_MODELS = ("gru", "flat_mlp")
SEQUENCE_MODELS = ("gru", "temporal_cnn")
SLICES = ("canonical", "sensitivity")
SPLIT_SEEDS = (0, 1, 2)
EXPECTED_TASK_COUNT = 38
EXPECTED_EPOCHS = 12
AUTHORITATIVE_ROOT_NAMES = (
    "pilot_artifacts/Unified38_fresh_rerun_active",
    "pilot_artifacts/Unified38_current_only_matrix",
)
MODEL_LABELS = {
    "gru": "GRU",
    "temporal_cnn": "TemporalCNN",
    "flat_mlp": "FlatMLP",
    "linear": "Linear",
    "prevalence": "Prevalence",
}
COLORS = {
    "gru": "#0072B2",
    "temporal_cnn": "#E69F00",
    "flat_mlp": "#009E73",
    "linear": "#CC79A7",
}


def canonical_model_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass
class RunRecord:
    condition: str
    task_name: str
    slice_name: str
    split_seed: int
    model_name: str
    source_platform: str
    step_auprc: float
    outcome_auprc: float
    step_ece: float
    step_lead: float
    step_prevalence: float
    outcome_prevalence: float
    training: dict[str, Any]
    protocol_fields: dict[str, Any]
    metrics_path: Path
    metrics_sha256: str
    bundle_path: Path
    bundle_sha256: str = ""
    parameter_count: int | None = None
    model_config: dict[str, Any] | None = None
    state_shape_signature: str = ""

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.task_name, self.slice_name, self.split_seed, self.model_name)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fields: list[str] = list(fieldnames or [])
    if not resolved_fields:
        for row in rows:
            for key in row:
                if key not in resolved_fields:
                    resolved_fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")


def guard_stage3_output(output_dir: str | Path, *, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    for relative in AUTHORITATIVE_ROOT_NAMES:
        authoritative = (root / relative).resolve()
        try:
            output.relative_to(authoritative)
        except ValueError:
            continue
        raise PermissionError(f"Stage 3 output may not be written under authoritative results: {authoritative}")
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise FileExistsError(f"Stage 3 output must be absent or empty: {output}")
    return output


def _parse_metrics_path(path: Path, run_root: Path) -> tuple[str, str, int, str]:
    parts = path.relative_to(run_root).parts
    if len(parts) != 5 or parts[-1] != "metrics.json" or not parts[2].startswith("seed_"):
        raise ValueError(f"Unexpected metrics path layout: {path}")
    task_name, slice_name, seed_token, model_dir, _ = parts
    return str(task_name), str(slice_name), int(seed_token.removeprefix("seed_")), canonical_model_name(model_dir)


def collect_run_records(
    run_root: str | Path,
    *,
    condition: str,
    repo_root: str | Path,
    hash_metrics: bool = True,
) -> list[RunRecord]:
    root = Path(run_root).resolve()
    repo = Path(repo_root).resolve()
    records: list[RunRecord] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        task_name, slice_name, split_seed, model_dir = _parse_metrics_path(metrics_path, root)
        payload = _read_json(metrics_path)
        model_name = canonical_model_name(payload["model_name"])
        if model_name != model_dir:
            raise ValueError(f"Model path/payload mismatch: {metrics_path}: {model_dir} != {model_name}")
        step = payload["label_step"]["test"]["overall"]
        outcome = payload["label_outcome"]["test"]["overall"]
        record = RunRecord(
            condition=str(condition),
            task_name=task_name,
            slice_name=slice_name,
            split_seed=int(split_seed),
            model_name=model_name,
            source_platform="",
            step_auprc=float(step["catastrophic_auprc"]),
            outcome_auprc=float(outcome["catastrophic_auprc"]),
            step_ece=float(step["catastrophic_ece"]),
            step_lead=float(step["lead_time_median_at_0.5"]),
            step_prevalence=float(step["catastrophic_prevalence"]),
            outcome_prevalence=float(outcome["catastrophic_prevalence"]),
            training=dict(payload["training"]),
            protocol_fields={
                "model_family": payload.get("model_family"),
                "input_view": payload.get("input_view"),
                "feature_profile": payload.get("feature_profile"),
                "selection_mode": payload.get("selection_mode"),
                "split_sizes": payload.get("split_sizes"),
                "label_protocol": payload.get("label_protocol"),
            },
            metrics_path=metrics_path,
            metrics_sha256=file_sha256(metrics_path) if hash_metrics else "",
            bundle_path=metrics_path.with_name("precursor_model.pt"),
        )
        if not record.bundle_path.is_file():
            raise FileNotFoundError(f"Missing model bundle for {metrics_path}")
        if not str(record.metrics_path.resolve()).startswith(str(repo)):
            raise ValueError(f"Metrics path escapes repository: {metrics_path}")
        records.append(record)
    return records


def validate_grid(full_records: Sequence[RunRecord], current_records: Sequence[RunRecord]) -> dict[str, Any]:
    full_tasks = sorted({record.task_name for record in full_records})
    current_tasks = sorted({record.task_name for record in current_records})
    errors: list[str] = []
    if len(full_tasks) != EXPECTED_TASK_COUNT:
        errors.append(f"full task count={len(full_tasks)}, expected={EXPECTED_TASK_COUNT}")
    if current_tasks != full_tasks:
        errors.append("current-only task set differs from full-prefix task set")

    def expected_keys(tasks: Sequence[str], models: Sequence[str]) -> set[tuple[str, str, int, str]]:
        return {
            (task, slice_name, seed, model)
            for task in tasks
            for slice_name in SLICES
            for seed in SPLIT_SEEDS
            for model in models
        }

    full_keys = [record.key for record in full_records]
    current_keys = [record.key for record in current_records]
    if len(set(full_keys)) != len(full_keys):
        errors.append("duplicate full-prefix run keys")
    if len(set(current_keys)) != len(current_keys):
        errors.append("duplicate current-only run keys")
    expected_full = expected_keys(full_tasks, FULL_MODELS)
    expected_current = expected_keys(full_tasks, TRAINABLE_MODELS)
    missing_full = sorted(expected_full - set(full_keys))
    extra_full = sorted(set(full_keys) - expected_full)
    missing_current = sorted(expected_current - set(current_keys))
    extra_current = sorted(set(current_keys) - expected_current)
    if missing_full or extra_full:
        errors.append(f"full grid mismatch: missing={len(missing_full)}, extra={len(extra_full)}")
    if missing_current or extra_current:
        errors.append(f"current-only grid mismatch: missing={len(missing_current)}, extra={len(extra_current)}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "task_count": len(full_tasks),
        "tasks": full_tasks,
        "full_metrics_count": len(full_records),
        "full_trainable_count": sum(record.model_name in TRAINABLE_MODELS for record in full_records),
        "current_only_metrics_count": len(current_records),
        "current_only_trainable_count": sum(record.model_name in TRAINABLE_MODELS for record in current_records),
    }


def cross_check_aggregate(
    records: Sequence[RunRecord],
    aggregate_path: str | Path,
    *,
    tolerance: float = 1e-12,
) -> dict[str, str]:
    rows = _read_csv(Path(aggregate_path))
    row_map = {
        (row["task_name"], row["slice_name"], int(row["split_seed"]), canonical_model_name(row["model_name"])): row
        for row in rows
    }
    if len(row_map) != len(rows) or len(rows) != len(records):
        raise ValueError(f"Aggregate coverage mismatch for {aggregate_path}: rows={len(rows)}, records={len(records)}")
    task_platform: dict[str, str] = {}
    for record in records:
        row = row_map.get(record.key)
        if row is None:
            raise KeyError(f"Run missing from aggregate: {record.key}")
        comparisons = {
            "step_auprc": record.step_auprc,
            "outcome_auprc": record.outcome_auprc,
            "ece": record.step_ece,
            "raw_step_lead_time": record.step_lead,
        }
        for field, observed in comparisons.items():
            if abs(float(row[field]) - float(observed)) > tolerance:
                raise ValueError(f"Aggregate mismatch {record.key} field={field}")
        platform = str(row["source_platform"])
        previous = task_platform.setdefault(record.task_name, platform)
        if previous != platform:
            raise ValueError(f"Task platform mismatch for {record.task_name}")
        record.source_platform = platform
    return task_platform


def attach_bundle_metadata(records: Sequence[RunRecord], *, hash_bundles: bool = True) -> None:
    for record in records:
        bundle = torch.load(record.bundle_path, map_location="cpu", weights_only=False)
        state = bundle.get("model_state_dict")
        config = dict(bundle.get("model_config") or {})
        if record.model_name == "prevalence":
            if state not in (None, {}):
                raise ValueError(f"Prevalence bundle unexpectedly has trainable state: {record.bundle_path}")
            parameter_count = 0
            signature_payload: list[dict[str, Any]] = []
        else:
            if not isinstance(state, Mapping) or not state:
                raise ValueError(f"Trainable bundle missing state_dict: {record.bundle_path}")
            parameter_count = int(sum(int(tensor.numel()) for tensor in state.values()))
            signature_payload = [
                {"name": str(name), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in state.items()
            ]
        record.parameter_count = parameter_count
        record.model_config = config
        record.state_shape_signature = canonical_json_sha256(signature_payload)
        record.bundle_sha256 = file_sha256(record.bundle_path) if hash_bundles else ""


def validate_matched_prevalence(
    full_records: Sequence[RunRecord],
    current_records: Sequence[RunRecord],
    *,
    tolerance: float = 1e-12,
) -> dict[tuple[str, str, int], tuple[float, float]]:
    grouped: dict[tuple[str, str, int], list[RunRecord]] = defaultdict(list)
    for record in [*full_records, *current_records]:
        grouped[(record.task_name, record.slice_name, record.split_seed)].append(record)
    result: dict[tuple[str, str, int], tuple[float, float]] = {}
    for key, rows in grouped.items():
        step_values = [row.step_prevalence for row in rows]
        outcome_values = [row.outcome_prevalence for row in rows]
        if max(step_values) - min(step_values) > tolerance:
            raise ValueError(f"Step prevalence is not model/condition invariant for {key}")
        if max(outcome_values) - min(outcome_values) > tolerance:
            raise ValueError(f"Outcome prevalence is not model/condition invariant for {key}")
        result[key] = (float(step_values[0]), float(outcome_values[0]))
    return result


def validate_endpoint_pair(metric_endpoint: str, reference_endpoint: str) -> None:
    'Reject any AUPRC/reference pairing that crosses Step and Outcome endpoints.'

    valid = {"step", "outcome"}
    metric = str(metric_endpoint).strip().lower()
    reference = str(reference_endpoint).strip().lower()
    if metric not in valid or reference not in valid:
        raise ValueError(f"Unknown endpoint pair: metric={metric_endpoint}, reference={reference_endpoint}")
    if metric != reference:
        raise ValueError(f"Cross-endpoint comparison is forbidden: metric={metric}, reference={reference}")


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("Cannot average an empty sequence")
    return float(sum(materialized) / len(materialized))


def task_model_means(records: Sequence[RunRecord]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.task_name, record.slice_name, record.model_name)].append(record)
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        if sorted(row.split_seed for row in rows) != list(SPLIT_SEEDS):
            raise ValueError(f"Expected split seeds {SPLIT_SEEDS} for {key}")
        result[key] = {
            "task_name": key[0],
            "slice_name": key[1],
            "model_name": key[2],
            "source_platform": rows[0].source_platform,
            "step_auprc": _mean(row.step_auprc for row in rows),
            "outcome_auprc": _mean(row.outcome_auprc for row in rows),
            "step_ece": _mean(row.step_ece for row in rows),
            "step_lead": _mean(row.step_lead for row in rows),
            "step_prevalence": _mean(row.step_prevalence for row in rows),
            "outcome_prevalence": _mean(row.outcome_prevalence for row in rows),
            "seed_count": len(rows),
        }
    return result


def build_primary_rows(full_means: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    validate_endpoint_pair("step", "step")
    tasks = sorted({key[0] for key in full_means})
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for slice_name in SLICES:
            gru = full_means[(task, slice_name, "gru")]
            flat = full_means[(task, slice_name, "flat_mlp")]
            prevalence = float(gru["step_prevalence"])
            if abs(prevalence - float(flat["step_prevalence"])) > 1e-12:
                raise ValueError(f"Fixed-model Step prevalence mismatch: {task}/{slice_name}")
            rows.append(
                {
                    "task_name": task,
                    "source_platform": gru["source_platform"],
                    "slice_name": slice_name,
                    "seed_aggregation": "arithmetic_mean_split_seeds_0_1_2",
                    "matched_step_prevalence": prevalence,
                    "gru_step_auprc": float(gru["step_auprc"]),
                    "gru_step_auprc_minus_prevalence": float(gru["step_auprc"]) - prevalence,
                    "flat_mlp_step_auprc": float(flat["step_auprc"]),
                    "flat_mlp_step_auprc_minus_prevalence": float(flat["step_auprc"]) - prevalence,
                }
            )
    return rows


def build_appendix_rows(full_means: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    validate_endpoint_pair("step", "step")
    tasks = sorted({key[0] for key in full_means})
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for slice_name in SLICES:
            base = full_means[(task, slice_name, "gru")]
            prevalence = float(base["step_prevalence"])
            row: dict[str, Any] = {
                "task_name": task,
                "source_platform": base["source_platform"],
                "slice_name": slice_name,
                "seed_aggregation": "arithmetic_mean_split_seeds_0_1_2",
                "matched_step_prevalence": prevalence,
            }
            for model in TRAINABLE_MODELS:
                value = float(full_means[(task, slice_name, model)]["step_auprc"])
                row[f"{model}_step_auprc"] = value
                row[f"{model}_step_auprc_minus_prevalence"] = value - prevalence
            rows.append(row)
    return rows


def build_endpoint_alignment_rows(
    full_means: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_endpoint_pair("step", "step")
    validate_endpoint_pair("outcome", "outcome")
    rows: list[dict[str, Any]] = []
    tasks = sorted({key[0] for key in full_means})
    for task in tasks:
        for slice_name in SLICES:
            legacy_reference = full_means[(task, slice_name, "prevalence")]
            for model in TRAINABLE_MODELS:
                value = full_means[(task, slice_name, model)]
                rows.append(
                    {
                        "task_name": task,
                        "source_platform": value["source_platform"],
                        "slice_name": slice_name,
                        "model_name": model,
                        "step_auprc": float(value["step_auprc"]),
                        "matched_step_prevalence": float(value["step_prevalence"]),
                        "step_auprc_minus_matched_step_prevalence": float(value["step_auprc"])
                        - float(value["step_prevalence"]),
                        "outcome_auprc": float(value["outcome_auprc"]),
                        "matched_outcome_prevalence": float(value["outcome_prevalence"]),
                        "outcome_auprc_minus_matched_outcome_prevalence": float(value["outcome_auprc"])
                        - float(value["outcome_prevalence"]),
                        "legacy_constant_score_step_auprc_not_used_as_prevalence": float(
                            legacy_reference["step_auprc"]
                        ),
                        "legacy_constant_score_outcome_auprc_not_used_as_prevalence": float(
                            legacy_reference["outcome_auprc"]
                        ),
                        "endpoint_alignment_status": "PASS",
                    }
                )
    return rows


def build_table_c1_rows(full_means: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    validate_endpoint_pair("outcome", "outcome")
    rows: list[dict[str, Any]] = []
    tasks = sorted({key[0] for key in full_means})
    for task in tasks:
        for slice_name in SLICES:
            sequence_values = {model: float(full_means[(task, slice_name, model)]["step_auprc"]) for model in SEQUENCE_MODELS}
            selected_model = max(SEQUENCE_MODELS, key=lambda model: (sequence_values[model], -SEQUENCE_MODELS.index(model)))
            selected = full_means[(task, slice_name, selected_model)]
            rows.append(
                {
                    "task_name": task,
                    "slice_name": slice_name,
                    "sequence_model_selected_by_mean_step_auprc": selected_model,
                    "outcome_auprc": float(selected["outcome_auprc"]),
                    "matched_outcome_prevalence": float(selected["outcome_prevalence"]),
                    "outcome_auprc_minus_matched_outcome_prevalence": float(selected["outcome_auprc"])
                    - float(selected["outcome_prevalence"]),
                    "step_ece": float(selected["step_ece"]),
                    "step_lead_median_steps": float(selected["step_lead"]),
                    "seed_aggregation": "arithmetic_mean_split_seeds_0_1_2",
                    "table_role": "auxiliary_not_fixed_model_main_table",
                }
            )
    return rows


def quantile_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("No values supplied")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_count": int(np.sum(array > 0.0)),
        "positive_proportion": float(np.mean(array > 0.0)),
    }


def bootstrap_mean_ci(values: Sequence[float], *, rounds: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap empty values")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(int(rounds), array.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_sign_flip_pvalue(values: Sequence[float], *, rounds: int, seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot test empty values")
    observed = abs(float(array.mean()))
    tolerance = 1e-15
    if array.size <= 20:
        total = 1 << int(array.size)
        extreme = 0
        offsets = np.arange(array.size, dtype=np.uint64)
        for start in range(0, total, 65536):
            indices = np.arange(start, min(total, start + 65536), dtype=np.uint64)
            bits = ((indices[:, None] >> offsets[None, :]) & 1).astype(np.int8)
            signs = bits * 2 - 1
            means = np.abs((signs * array[None, :]).mean(axis=1))
            extreme += int(np.sum(means >= observed - tolerance))
        return float(extreme / total)
    rng = np.random.default_rng(int(seed))
    extreme = 0
    completed = 0
    while completed < int(rounds):
        batch = min(20000, int(rounds) - completed)
        signs = rng.integers(0, 2, size=(batch, array.size), dtype=np.int8) * 2 - 1
        means = np.abs((signs * array[None, :]).mean(axis=1))
        extreme += int(np.sum(means >= observed - tolerance))
        completed += batch
    return float((extreme + 1) / (int(rounds) + 1))


def paired_dz(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float("nan")
    standard_deviation = float(array.std(ddof=1))
    mean = float(array.mean())
    if standard_deviation == 0.0:
        if mean == 0.0:
            return 0.0
        return math.copysign(float("inf"), mean)
    return mean / standard_deviation


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(((str(name), float(value)) for name, value in pvalues.items()), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        running = max(running, (total - rank) * pvalue)
        adjusted[name] = min(1.0, running)
    return adjusted


def _comparison_stats(
    values: Sequence[float],
    *,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    permutation_rounds: int,
    permutation_seed: int,
) -> dict[str, Any]:
    summary = quantile_summary(values)
    ci_low, ci_high = bootstrap_mean_ci(values, rounds=bootstrap_rounds, seed=bootstrap_seed)
    return {
        **summary,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "paired_sign_flip_p_raw": paired_sign_flip_pvalue(
            values, rounds=permutation_rounds, seed=permutation_seed
        ),
        "paired_dz": paired_dz(values),
    }


def build_statistics(
    full_means: Mapping[tuple[str, str, str], Mapping[str, Any]],
    current_means: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    permutation_rounds: int,
    permutation_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = sorted({key[0] for key in full_means})
    formal_definitions = [
        ("F1_canonical_learnability", "gru_vs_matched_step_prevalence", "gru"),
        ("F1_canonical_learnability", "flat_mlp_vs_matched_step_prevalence", "flat_mlp"),
        ("F2_canonical_history_increment", "gru_full_minus_current_only", "gru"),
        ("F2_canonical_history_increment", "flat_mlp_full_minus_current_only", "flat_mlp"),
    ]
    paired_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for family, comparison, model in formal_definitions:
        values: list[float] = []
        platform_values: dict[str, list[float]] = defaultdict(list)
        for task in tasks:
            full = full_means[(task, "canonical", model)]
            if family.startswith("F1"):
                left = float(full["step_auprc"])
                right = float(full["step_prevalence"])
                right_label = "matched_step_prevalence"
            else:
                left = float(full["step_auprc"])
                right = float(current_means[(task, "canonical", model)]["step_auprc"])
                right_label = "current_only_step_auprc"
            difference = left - right
            values.append(difference)
            platform_values[str(full["source_platform"])].append(difference)
        stats = _comparison_stats(
            values,
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_seed=bootstrap_seed,
            permutation_rounds=permutation_rounds,
            permutation_seed=permutation_seed,
        )
        platform_means = [_mean(group) for _, group in sorted(platform_values.items())]
        result_rows.append(
            {
                "family": family,
                "family_status": "interim_existing_models_m2_pending_transformer",
                "comparison": comparison,
                **stats,
                "platform_count": len(platform_means),
                "platform_equal_weighted_mean": _mean(platform_means),
                "bootstrap_rounds": int(bootstrap_rounds),
                "bootstrap_seed": int(bootstrap_seed),
                "permutation_rounds": int(permutation_rounds),
                "permutation_seed": int(permutation_seed),
            }
        )
    for family in sorted({row["family"] for row in result_rows}):
        family_rows = [row for row in result_rows if row["family"] == family]
        adjusted = holm_adjust({str(row["comparison"]): float(row["paired_sign_flip_p_raw"]) for row in family_rows})
        for row in family_rows:
            row["holm_adjusted_p_existing_m2"] = adjusted[str(row["comparison"])]
            row["reject_holm_0_05_existing_m2"] = bool(adjusted[str(row["comparison"])] <= 0.05)

    formal_lookup = {
        ("canonical", "gru", "step_auprc_minus_prevalence"): (
            "F1_canonical_learnability",
            "gru_vs_matched_step_prevalence",
        ),
        ("canonical", "flat_mlp", "step_auprc_minus_prevalence"): (
            "F1_canonical_learnability",
            "flat_mlp_vs_matched_step_prevalence",
        ),
        ("canonical", "gru", "full_minus_current_only"): (
            "F2_canonical_history_increment",
            "gru_full_minus_current_only",
        ),
        ("canonical", "flat_mlp", "full_minus_current_only"): (
            "F2_canonical_history_increment",
            "flat_mlp_full_minus_current_only",
        ),
    }
    descriptive_rows: list[dict[str, Any]] = []
    for slice_name in SLICES:
        for model in TRAINABLE_MODELS:
            learnability = [
                float(full_means[(task, slice_name, model)]["step_auprc"])
                - float(full_means[(task, slice_name, model)]["step_prevalence"])
                for task in tasks
            ]
            history = [
                float(full_means[(task, slice_name, model)]["step_auprc"])
                - float(current_means[(task, slice_name, model)]["step_auprc"])
                for task in tasks
            ]
            for contrast, values in (("step_auprc_minus_prevalence", learnability), ("full_minus_current_only", history)):
                formal = formal_lookup.get((slice_name, model, contrast))
                for task, difference in zip(tasks, values):
                    full = full_means[(task, slice_name, model)]
                    if contrast == "step_auprc_minus_prevalence":
                        right = float(full["step_prevalence"])
                        right_label = "matched_step_prevalence"
                    else:
                        right = float(current_means[(task, slice_name, model)]["step_auprc"])
                        right_label = "current_only_step_auprc"
                    paired_rows.append(
                        {
                            "family": formal[0] if formal else "descriptive_supplement",
                            "family_status": (
                                "interim_existing_models_m2_pending_transformer"
                                if formal
                                else "descriptive_only_not_new_significance_family"
                            ),
                            "comparison": formal[1] if formal else f"{slice_name}_{model}_{contrast}",
                            "slice_name": slice_name,
                            "model_name": model,
                            "contrast": contrast,
                            "task_name": task,
                            "source_platform": full["source_platform"],
                            "left_endpoint": "full_prefix_step_auprc",
                            "right_endpoint": right_label,
                            "left_value": float(full["step_auprc"]),
                            "right_value": right,
                            "paired_difference": float(difference),
                        }
                    )
                descriptive_rows.append(
                    {
                        "slice_name": slice_name,
                        "model_name": model,
                        "contrast": contrast,
                        "inference_status": "descriptive_only_not_new_significance_family",
                        **quantile_summary(values),
                        "paired_dz": paired_dz(values),
                    }
                )
    return paired_rows, result_rows, descriptive_rows


def audit_histories(
    records: Sequence[RunRecord],
    *,
    bootstrap_rounds: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    best_counts: dict[tuple[str, str, str, int], int] = defaultdict(int)
    trainable = [record for record in records if record.model_name in TRAINABLE_MODELS]
    for record in trainable:
        history = list(record.training.get("history") or [])
        reasons: list[str] = []
        if len(history) != EXPECTED_EPOCHS:
            reasons.append(f"history_length={len(history)}")
        epochs = [int(item.get("epoch", -1)) for item in history]
        if epochs != list(range(len(history))):
            reasons.append("epoch_sequence_invalid")
        numeric_fields = [
            float(item[field])
            for item in history
            for field in ("train_loss", "val_catastrophic_auprc", "temperature")
            if field in item
        ]
        if len(numeric_fields) != len(history) * 3 or not all(math.isfinite(value) for value in numeric_fields):
            reasons.append("nonfinite_or_missing_history_value")
        best_epoch = int(record.training.get("best_epoch", -1))
        if history:
            val_values = [float(item["val_catastrophic_auprc"]) for item in history]
            expected_best = int(np.argmax(np.asarray(val_values, dtype=np.float64)))
            if best_epoch != expected_best:
                reasons.append(f"best_epoch={best_epoch},expected={expected_best}")
            reported_best = float(record.training.get("best_val_catastrophic_auprc", float("nan")))
            if not math.isfinite(reported_best) or abs(reported_best - val_values[expected_best]) > 1e-12:
                reasons.append("best_val_metric_mismatch")
        if reasons:
            failures.append(
                {
                    "condition": record.condition,
                    "task_name": record.task_name,
                    "slice_name": record.slice_name,
                    "split_seed": record.split_seed,
                    "model_name": record.model_name,
                    "metrics_path": str(record.metrics_path),
                    "failure_reasons": ";".join(reasons),
                }
            )
        best_counts[(record.condition, record.slice_name, record.model_name, best_epoch)] += 1
        for item in history:
            long_rows.append(
                {
                    "condition": record.condition,
                    "task_name": record.task_name,
                    "source_platform": record.source_platform,
                    "slice_name": record.slice_name,
                    "split_seed": record.split_seed,
                    "model_name": record.model_name,
                    "epoch": int(item["epoch"]),
                    "train_loss": float(item["train_loss"]),
                    "validation_step_auprc": float(item["val_catastrophic_auprc"]),
                    "temperature": float(item["temperature"]),
                    "is_selected_best_epoch": int(int(item["epoch"]) == best_epoch),
                    "metrics_sha256": record.metrics_sha256,
                }
            )
    best_rows: list[dict[str, Any]] = []
    for condition in ("full_prefix", "current_only"):
        for slice_name in SLICES:
            for model in TRAINABLE_MODELS:
                total = sum(best_counts[(condition, slice_name, model, epoch)] for epoch in range(EXPECTED_EPOCHS))
                for epoch in range(EXPECTED_EPOCHS):
                    count = best_counts[(condition, slice_name, model, epoch)]
                    best_rows.append(
                        {
                            "condition": condition,
                            "slice_name": slice_name,
                            "model_name": model,
                            "best_epoch": epoch,
                            "run_count": count,
                            "run_proportion": float(count / total) if total else 0.0,
                            "total_runs": total,
                        }
                    )

    grouped: dict[tuple[str, str, str, int, str], list[float]] = defaultdict(list)
    for row in long_rows:
        grouped[(row["condition"], row["slice_name"], row["model_name"], row["epoch"], row["task_name"])].append(
            float(row["validation_step_auprc"])
        )
    curve_rows: list[dict[str, Any]] = []
    for condition in ("full_prefix", "current_only"):
        for slice_name in SLICES:
            for model in TRAINABLE_MODELS:
                for epoch in range(EXPECTED_EPOCHS):
                    task_values = [
                        _mean(grouped[(condition, slice_name, model, epoch, task)])
                        for task in sorted({key[4] for key in grouped if key[:4] == (condition, slice_name, model, epoch)})
                    ]
                    if len(task_values) != EXPECTED_TASK_COUNT:
                        raise ValueError(
                            f"Convergence task coverage mismatch: {(condition, slice_name, model, epoch)}={len(task_values)}"
                        )
                    stable_offset = int(sha256(f"{condition}/{slice_name}/{model}/{epoch}".encode()).hexdigest()[:8], 16)
                    ci_low, ci_high = bootstrap_mean_ci(
                        task_values,
                        rounds=bootstrap_rounds,
                        seed=int(bootstrap_seed) + stable_offset,
                    )
                    curve_rows.append(
                        {
                            "condition": condition,
                            "slice_name": slice_name,
                            "model_name": model,
                            "epoch": epoch,
                            "task_count": len(task_values),
                            "task_mean_validation_step_auprc": _mean(task_values),
                            "task_bootstrap_ci95_low": ci_low,
                            "task_bootstrap_ci95_high": ci_high,
                        }
                    )
    return long_rows, best_rows, failures, curve_rows


def _approximate_macs(model: str, config: Mapping[str, Any], feature_dim: int, window_size: int) -> int:
    classes = int(config.get("num_classes", 3))
    hidden = int(config.get("hidden_dim", 128))
    if model == "gru":
        return int(window_size * 3 * (feature_dim * hidden + hidden * hidden) + hidden * hidden + hidden * classes)
    if model == "temporal_cnn":
        kernel = int(config.get("cnn_kernel_size", 3))
        layers = int(config.get("num_layers", 1))
        total = window_size * kernel * feature_dim * hidden
        if layers > 1:
            total += (layers - 1) * window_size * kernel * hidden * hidden
        return int(total + hidden * hidden + hidden * classes)
    if model == "flat_mlp":
        depth = int(config.get("mlp_depth", 2))
        flat_dim = int(window_size * feature_dim)
        total = flat_dim * hidden
        if depth > 1:
            total += (depth - 1) * hidden * hidden
        return int(total + hidden * classes)
    if model == "linear":
        return int(window_size * feature_dim * classes)
    return 0


def build_complexity_rows(records: Sequence[RunRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.task_name, record.model_name)].append(record)
    task_rows: list[dict[str, Any]] = []
    formulas = {
        "gru": "W*3*(D*H + H*H) + H*H + H*C",
        "temporal_cnn": "W*K*D*H + (L-1)*W*K*H*H + H*H + H*C",
        "flat_mlp": "W*D*H + (L-1)*H*H + H*C",
        "linear": "W*D*C",
        "prevalence": "O(C) constant probability broadcast; 0 trainable parameters",
    }
    for (task, model), rows in sorted(grouped.items()):
        parameter_counts = {int(row.parameter_count or 0) for row in rows}
        signatures = {row.state_shape_signature for row in rows}
        if len(parameter_counts) != 1 or len(signatures) != 1:
            raise ValueError(f"Parameter/state shape inconsistency for {task}/{model}")
        representative = sorted(rows, key=lambda row: (row.condition, row.slice_name, row.split_seed))[0]
        config = dict(representative.model_config or {})
        feature_dim = int(config.get("input_dim", 0))
        window_size = int(config.get("window_size", 32))
        if model in {"flat_mlp", "linear"} and window_size > 0:
            feature_dim = int(feature_dim // window_size)
        task_rows.append(
            {
                "task_name": task,
                "model_name": model,
                "feature_dim": feature_dim,
                "window_size": window_size,
                "trainable_parameter_count": next(iter(parameter_counts)),
                "approximate_macs_per_sample": _approximate_macs(model, config, feature_dim, window_size),
                "dominant_complexity_formula": formulas[model],
                "state_shape_signature": next(iter(signatures)),
                "status": "provisional_existing_models_only_supersede_at_G5_G10",
            }
        )
    summary_rows: list[dict[str, Any]] = []
    for model in FULL_MODELS:
        rows = [row for row in task_rows if row["model_name"] == model]
        params = [int(row["trainable_parameter_count"]) for row in rows]
        macs = [int(row["approximate_macs_per_sample"]) for row in rows]
        summary_rows.append(
            {
                "model_name": model,
                "task_count": len(rows),
                "parameter_count_min": min(params),
                "parameter_count_median": float(statistics.median(params)),
                "parameter_count_max": max(params),
                "approximate_macs_min": min(macs),
                "approximate_macs_median": float(statistics.median(macs)),
                "approximate_macs_max": max(macs),
                "dominant_complexity_formula": rows[0]["dominant_complexity_formula"],
                "status": "provisional_existing_models_only_supersede_at_G5_G10",
            }
        )
    return task_rows, summary_rows


def _apply_plot_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.6,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, base_path: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix in (".pdf", ".svg", ".png"):
        path = base_path.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_convergence(curve_rows: Sequence[Mapping[str, Any]], *, condition: str, output_dir: Path) -> list[Path]:
    _apply_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    for axis, slice_name in zip(axes, SLICES):
        for model in TRAINABLE_MODELS:
            rows = sorted(
                [row for row in curve_rows if row["condition"] == condition and row["slice_name"] == slice_name and row["model_name"] == model],
                key=lambda row: int(row["epoch"]),
            )
            x = np.asarray([int(row["epoch"]) + 1 for row in rows])
            mean = np.asarray([float(row["task_mean_validation_step_auprc"]) for row in rows])
            low = np.asarray([float(row["task_bootstrap_ci95_low"]) for row in rows])
            high = np.asarray([float(row["task_bootstrap_ci95_high"]) for row in rows])
            axis.plot(x, mean, color=COLORS[model], label=MODEL_LABELS[model])
            axis.fill_between(x, low, high, color=COLORS[model], alpha=0.12, linewidth=0)
        axis.set_title(f"{slice_name.capitalize()} slice")
        axis.set_xlabel("Epoch")
        axis.set_xticks(range(1, EXPECTED_EPOCHS + 1, 2))
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Task-mean validation Step AUPRC")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"{condition.replace('_', ' ').title()} convergence (38 tasks; seed-mean per task)", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, output_dir / f"{condition}_convergence")


def plot_best_epoch(best_rows: Sequence[Mapping[str, Any]], *, condition: str, output_dir: Path) -> list[Path]:
    _apply_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    image = None
    for axis, slice_name in zip(axes, SLICES):
        matrix = np.zeros((len(TRAINABLE_MODELS), EXPECTED_EPOCHS), dtype=np.float64)
        for row in best_rows:
            if row["condition"] == condition and row["slice_name"] == slice_name:
                model_index = TRAINABLE_MODELS.index(str(row["model_name"]))
                matrix[model_index, int(row["best_epoch"])] = float(row["run_proportion"])
        image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=max(0.35, float(matrix.max())), cmap="Blues")
        axis.set_title(f"{slice_name.capitalize()} slice")
        axis.set_xlabel("Selected best epoch")
        axis.set_xticks(range(EXPECTED_EPOCHS))
        axis.set_xticklabels(range(1, EXPECTED_EPOCHS + 1))
        axis.set_yticks(range(len(TRAINABLE_MODELS)))
        axis.set_yticklabels([MODEL_LABELS[model] for model in TRAINABLE_MODELS])
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if value >= 0.08:
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color="white" if value >= 0.24 else "black",
                    )
    if image is not None:
        colorbar_axis = fig.add_axes([0.915, 0.18, 0.018, 0.64])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Run proportion")
    fig.suptitle(f"{condition.replace('_', ' ').title()} best-epoch distribution (114 runs/model/slice)", y=1.01)
    fig.subplots_adjust(left=0.12, right=0.885, bottom=0.17, top=0.83, wspace=0.14)
    return _save_figure(fig, output_dir / f"{condition}_best_epoch")


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[tuple[str, str]], digits: int = 4) -> str:
    lines = ["| " + " | ".join(label for _, label in fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values: list[str] = []
        for key, _ in fields:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)




def build_carcircle_trace(
    full_records: Sequence[RunRecord],
    full_means: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    rows = sorted(
        [
            record
            for record in full_records
            if record.task_name == "CarCircle" and record.slice_name == "canonical" and record.model_name == "gru"
        ],
        key=lambda record: record.split_seed,
    )
    old_median = float(statistics.median(record.outcome_auprc for record in rows))
    repo = Path(repo_root).resolve() if repo_root is not None else None
    payload = {
        "schema_version": "ottp_stage3_carcircle_trace_v1",
        "task_name": "CarCircle",
        "slice_name": "canonical",
        "model_name": "gru",
        "old_table_value": old_median,
        "old_table_value_semantics": "median split-seed Outcome AUPRC, ambiguously labeled AUPRC",
        "historical_paper_source": "OTTP.pdf page 22, Table C.1",
        "historical_task_card_source": "pilot_artifacts/Unified38_current_paper_source/Unified38_task_cards.json",
        "corrected_aggregation": "arithmetic_mean_split_seeds_0_1_2",
        "corrected_outcome_auprc": float(full_means[("CarCircle", "canonical", "gru")]["outcome_auprc"]),
        "matched_outcome_prevalence": float(full_means[("CarCircle", "canonical", "gru")]["outcome_prevalence"]),
        "corrected_step_auprc": float(full_means[("CarCircle", "canonical", "gru")]["step_auprc"]),
        "matched_step_prevalence": float(full_means[("CarCircle", "canonical", "gru")]["step_prevalence"]),
        "seed_rows": [
            {
                "split_seed": record.split_seed,
                "step_auprc": record.step_auprc,
                "matched_step_prevalence": record.step_prevalence,
                "outcome_auprc": record.outcome_auprc,
                "matched_outcome_prevalence": record.outcome_prevalence,
                "metrics_path": _relative(record.metrics_path, repo) if repo is not None else str(record.metrics_path),
                "metrics_sha256": record.metrics_sha256,
            }
            for record in rows
        ],
        "endpoint_alignment_status": "PASS",
    }
    if abs(old_median - 0.14020290056294188) > 1e-15:
        raise ValueError(f"CarCircle legacy value changed unexpectedly: {old_median}")
    return payload






def _input_manifest_records(records: Sequence[RunRecord], repo_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        items.extend(
            [
                {
                    "kind": "metrics",
                    "condition": record.condition,
                    "task_name": record.task_name,
                    "slice_name": record.slice_name,
                    "split_seed": record.split_seed,
                    "model_name": record.model_name,
                    "protocol_profile_sha256": canonical_json_sha256(record.protocol_fields),
                    "path": _relative(record.metrics_path, repo_root),
                    "sha256": record.metrics_sha256,
                    "bytes": int(record.metrics_path.stat().st_size),
                },
                {
                    "kind": "model_bundle_parameter_audit_only",
                    "condition": record.condition,
                    "task_name": record.task_name,
                    "slice_name": record.slice_name,
                    "split_seed": record.split_seed,
                    "model_name": record.model_name,
                    "protocol_profile_sha256": canonical_json_sha256(record.protocol_fields),
                    "path": _relative(record.bundle_path, repo_root),
                    "sha256": record.bundle_sha256,
                    "bytes": int(record.bundle_path.stat().st_size),
                },
            ]
        )
    return items


def build_protocol_field_inventory(records: Sequence[RunRecord]) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    field_presence: dict[str, int] = defaultdict(int)

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in sorted(value.items()):
                visit(f"{prefix}.{key}" if prefix else str(key), child)
            return
        field_presence[prefix] += 1

    for record in records:
        profile_id = canonical_json_sha256(record.protocol_fields)
        entry = profiles.setdefault(
            profile_id,
            {
                "protocol_profile_sha256": profile_id,
                "protocol_fields": record.protocol_fields,
                "run_count": 0,
                "conditions": set(),
                "models": set(),
            },
        )
        entry["run_count"] += 1
        entry["conditions"].add(record.condition)
        entry["models"].add(record.model_name)
        visit("", record.protocol_fields)
    serializable_profiles = []
    for profile_id in sorted(profiles):
        entry = profiles[profile_id]
        serializable_profiles.append(
            {
                **entry,
                "conditions": sorted(entry["conditions"]),
                "models": sorted(entry["models"]),
            }
        )
    return {
        "schema_version": "ottp_stage3_protocol_field_inventory_v1",
        "run_count": len(records),
        "profile_count": len(serializable_profiles),
        "field_presence_counts": dict(sorted(field_presence.items())),
        "profiles": serializable_profiles,
    }


def write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name not in {"analysis_artifact_manifest.json", "SHA256SUMS"})
    payload = {
        "schema_version": "ottp_stage3_artifact_manifest_v1",
        "files": [
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for path in files
        ],
    }
    payload["file_count"] = len(payload["files"])
    payload["total_bytes"] = int(sum(item["bytes"] for item in payload["files"]))
    _write_json(output_dir / "analysis_artifact_manifest.json", payload)
    return payload


def write_sha256s(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{file_sha256(path)}  {str(path.relative_to(root)).replace(chr(92), '/')}" for path in files]
    target = root / "SHA256SUMS"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


