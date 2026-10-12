'modern-baseline matrix and statistical helpers.'

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ctsc.precursor.revision import canonical_json_sha256, file_sha256
from ctsc.precursor.stage3 import (
    bootstrap_mean_ci,
    holm_adjust,
    paired_dz,
    paired_sign_flip_pvalue,
    quantile_summary,
)


FREEZE_ID = "OTTP-R26-G1-v1"
TASKS_EXPECTED = 38
SPLIT_SEEDS = (0, 1, 2)
TUNING_TASKS = ("AntCircle", "easydense", "CarPush2", "AntVelocity")
TUNING_MODELS = ("gru", "flat_mlp", "transformer")
FINAL_MODELS = ("gru", "flat_mlp", "transformer")
LEARNING_RATES = (3e-4, 1e-3, 3e-3)
LR_TIE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Stage5Condition:
    condition_id: str
    feature_profile: str
    q_low: float
    q_high: float
    role: str


FINAL_CONDITIONS = (
    Stage5Condition("canonical_full", "full", 0.75, 0.90, "confirmatory"),
    Stage5Condition("sensitivity_full", "full", 0.80, 0.95, "descriptive_sensitivity"),
    Stage5Condition("canonical_current_only", "current_only", 0.75, 0.90, "confirmatory"),
)
CONDITIONS_BY_ID = {condition.condition_id: condition for condition in FINAL_CONDITIONS}


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if str(field) not in fields:
                fields.append(str(field))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def artifact_manifest(root: str | Path) -> dict[str, Any]:
    output = Path(root)
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "SHA256SUMS"}
    )
    payload = {
        "schema_version": "ottp_stage5_artifact_manifest_v1",
        "files": [
            {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for path in files
        ],
    }
    payload["file_count"] = len(payload["files"])
    payload["total_bytes"] = int(sum(row["bytes"] for row in payload["files"]))
    write_json(output / "artifact_manifest.json", payload)
    return payload


def write_sha256s(root: str | Path) -> Path:
    output = Path(root)
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    target = output / "SHA256SUMS"
    target.write_text(
        "".join(f"{file_sha256(path)}  {str(path.relative_to(output)).replace(chr(92), '/')}\n" for path in files),
        encoding="utf-8",
    )
    return target


def frozen_matrix_manifest(tasks: Sequence[str]) -> dict[str, Any]:
    ordered_tasks = tuple(str(task) for task in tasks)
    if len(ordered_tasks) != TASKS_EXPECTED or len(set(ordered_tasks)) != TASKS_EXPECTED:
        raise ValueError(f"Stage 5 requires exactly {TASKS_EXPECTED} unique tasks")
    if not set(TUNING_TASKS).issubset(ordered_tasks):
        raise ValueError("The four frozen learning-rate tasks are not contained in the registry")
    payload: dict[str, Any] = {
        "schema_version": "ottp_stage5_frozen_matrix_v1",
        "freeze_id": FREEZE_ID,
        "tasks": list(ordered_tasks),
        "split_seeds": list(SPLIT_SEEDS),
        "tuning": {
            "tasks": list(TUNING_TASKS),
            "models": list(TUNING_MODELS),
            "learning_rates": list(LEARNING_RATES),
            "selection_metric": "mean_validation_step_catastrophic_auprc_across_four_tasks",
            "tie_rule": "within_strictly_less_than_1e-4_of_best_choose_lower_lr",
            "expected_model_runs": len(TUNING_TASKS) * len(TUNING_MODELS) * len(LEARNING_RATES),
            "old_model_required_lr": 1e-3,
        },
        "final": {
            "models_trained": ["transformer"],
            "conditions": [asdict(condition) for condition in FINAL_CONDITIONS],
            "expected_transformer_runs": TASKS_EXPECTED * len(SPLIT_SEEDS) * len(FINAL_CONDITIONS),
        },
        "architecture": {
            "layers": 2,
            "heads": 4,
            "ffn_multiplier": 4,
            "activation": "relu",
            "normalization": "post_norm",
            "dropout": 0.0,
            "learned_position_length": 32,
            "mask": "strict_upper_triangular_causal",
            "readout": "last_position",
            "head": "LayerNorm-Linear-ReLU-Linear",
            "d_model_candidates": list(range(16, 129, 4)),
            "parameter_tolerance_percent": 10.0,
        },
        "statistics": {
            "unit": "task_after_arithmetic_mean_over_split_seeds_0_1_2",
            "bootstrap_rounds": 20000,
            "bootstrap_seed": 260922,
            "permutation_rounds": 100000,
            "permutation_seed": 260921,
            "families": {"F1": 3, "F2": 3, "F3": 2},
        },
        "excluded": [
            "future_only",
            "gamma sweep",
            "Mamba",
            "Decision Transformer",
            "post-hoc tasks, seeds, or models",
            "test-informed learning-rate selection",
        ],
    }
    payload["matrix_sha256"] = canonical_json_sha256(payload)
    return payload


def tuning_jobs() -> list[dict[str, Any]]:
    return [
        {
            "job_id": f"tuning__{task}__{model}__lr_{lr:.0e}",
            "task_name": task,
            "model_name": model,
            "learning_rate": float(lr),
            "split_seed": 0,
        }
        for task in TUNING_TASKS
        for model in TUNING_MODELS
        for lr in LEARNING_RATES
    ]


def final_jobs(tasks: Sequence[str]) -> list[dict[str, Any]]:
    manifest = frozen_matrix_manifest(tasks)
    del manifest
    return [
        {
            "job_id": f"final__{condition.condition_id}__{task}__ss{seed}",
            "condition_id": condition.condition_id,
            "task_name": task,
            "split_seed": int(seed),
            "model_name": "transformer",
        }
        for condition in FINAL_CONDITIONS
        for task in tasks
        for seed in SPLIT_SEEDS
    ]


def select_learning_rates(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    'Select one LR per model using validation Step AUPRC only.'

    expected = {
        (task, model, float(lr))
        for task in TUNING_TASKS
        for model in TUNING_MODELS
        for lr in LEARNING_RATES
    }
    index: dict[tuple[str, str, float], float] = {}
    for row in rows:
        key = (str(row["task_name"]), str(row["model_name"]), float(row["learning_rate"]))
        if key in index:
            raise ValueError(f"Duplicate Stage 5 tuning cell: {key}")
        value = float(row["validation_step_catastrophic_auprc"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite validation metric for {key}")
        index[key] = value
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise ValueError(f"Stage 5 tuning coverage mismatch; missing={missing}, extra={extra}")

    score_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {
        "schema_version": "ottp_stage5_validation_only_lr_selection_v1",
        "selection_uses_test_metrics": False,
        "tie_tolerance": LR_TIE_TOLERANCE,
        "tie_comparison": "strictly_less_than",
        "models": {},
    }
    for model in TUNING_MODELS:
        means: dict[float, float] = {}
        for lr in LEARNING_RATES:
            values = [index[(task, model, float(lr))] for task in TUNING_TASKS]
            means[float(lr)] = float(np.mean(values))
            score_rows.append(
                {
                    "model_name": model,
                    "learning_rate": float(lr),
                    "validation_task_count": len(values),
                    "mean_validation_step_catastrophic_auprc": means[float(lr)],
                    **{f"validation_{task}_step_catastrophic_auprc": value for task, value in zip(TUNING_TASKS, values)},
                }
            )
        best = max(means.values())
        candidates = sorted(lr for lr, mean in means.items() if best - mean < LR_TIE_TOLERANCE)
        selected = float(candidates[0])
        selection["models"][model] = {
            "selected_learning_rate": selected,
            "best_mean_validation_step_catastrophic_auprc": float(best),
            "selected_mean_validation_step_catastrophic_auprc": float(means[selected]),
            "tie_candidate_learning_rates": candidates,
            "four_task_means": {format(lr, ".0e"): value for lr, value in sorted(means.items())},
        }
    selection["selection_sha256"] = canonical_json_sha256(selection)
    return score_rows, selection


def learning_rate_compatibility_gate(selection: Mapping[str, Any]) -> dict[str, Any]:
    selected = {
        model: float(selection["models"][model]["selected_learning_rate"])
        for model in TUNING_MODELS
    }
    compatible = math.isclose(selected["gru"], 1e-3, abs_tol=1e-15) and math.isclose(
        selected["flat_mlp"], 1e-3, abs_tol=1e-15
    )
    return {
        "schema_version": "ottp_stage5_lr_compatibility_gate_v1",
        "status": "PASS" if compatible else "G5_BLOCKED_LR_COMPATIBILITY",
        "required_existing_old_model_learning_rate": 1e-3,
        "selected_learning_rates": selected,
        "test_metrics_read_before_selection_lock": False,
        "final_matrix_authorized": bool(compatible),
    }


def _comparison_stats(
    values: Sequence[float],
    *,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    permutation_rounds: int,
    permutation_seed: int,
) -> dict[str, Any]:
    ci_low, ci_high = bootstrap_mean_ci(values, rounds=bootstrap_rounds, seed=bootstrap_seed)
    return {
        **quantile_summary(values),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "paired_sign_flip_p_raw": paired_sign_flip_pvalue(
            values, rounds=permutation_rounds, seed=permutation_seed
        ),
        "paired_dz": paired_dz(values),
    }


def build_frozen_statistics(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_rounds: int = 20000,
    bootstrap_seed: int = 260922,
    permutation_rounds: int = 100000,
    permutation_seed: int = 260921,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    'Build the frozen F1/F2/F3 families from 38 task-level means.'

    index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in task_rows:
        key = (str(row["condition_id"]), str(row["model_name"]), str(row["task_name"]))
        if key in index:
            raise ValueError(f"Duplicate Stage 5 task-level row: {key}")
        index[key] = row
    tasks = sorted({key[2] for key in index if key[0] == "canonical_full"})
    if len(tasks) != TASKS_EXPECTED:
        raise ValueError(f"Stage 5 statistics require {TASKS_EXPECTED} tasks, got {len(tasks)}")

    definitions: list[dict[str, str]] = []
    for model in FINAL_MODELS:
        definitions.append(
            {
                "family": "F1_canonical_learnability",
                "comparison": f"{model}_vs_matched_step_prevalence",
                "model_name": model,
                "left_condition": "canonical_full",
                "right": "matched_step_prevalence",
                "value_kind": "step_auprc_minus_matched_step_prevalence",
            }
        )
    for model in FINAL_MODELS:
        definitions.append(
            {
                "family": "F2_canonical_history_increment",
                "comparison": f"{model}_full_minus_current_only",
                "model_name": model,
                "left_condition": "canonical_full",
                "right": "canonical_current_only",
                "value_kind": "step_auprc_difference_same_endpoint",
            }
        )
    for comparator in ("gru", "flat_mlp"):
        definitions.append(
            {
                "family": "F3_modern_baseline",
                "comparison": f"transformer_minus_{comparator}",
                "model_name": "transformer",
                "left_condition": "canonical_full",
                "right": comparator,
                "value_kind": "step_auprc_difference_same_condition",
            }
        )

    paired_rows: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []
    for definition in definitions:
        differences: list[float] = []
        for task in tasks:
            left = index[(definition["left_condition"], definition["model_name"], task)]
            left_value = float(left["step_auprc"])
            if definition["right"] == "matched_step_prevalence":
                right_value = float(left["matched_step_prevalence"])
                right_label = "matched_step_prevalence"
            elif definition["family"] == "F2_canonical_history_increment":
                right = index[(definition["right"], definition["model_name"], task)]
                right_value = float(right["step_auprc"])
                right_label = definition["right"]
            else:
                right = index[("canonical_full", definition["right"], task)]
                right_value = float(right["step_auprc"])
                right_label = definition["right"]
            difference = left_value - right_value
            differences.append(difference)
            paired_rows.append(
                {
                    "family": definition["family"],
                    "comparison": definition["comparison"],
                    "task_name": task,
                    "source_platform": left.get("source_platform", ""),
                    "model_name": definition["model_name"],
                    "left_condition_or_model": definition["left_condition"],
                    "right_condition_or_model": right_label,
                    "value_kind": definition["value_kind"],
                    "left_value": left_value,
                    "right_value": right_value,
                    "paired_difference": difference,
                }
            )
        statistics.append(
            {
                "family": definition["family"],
                "comparison": definition["comparison"],
                "model_name": definition["model_name"],
                **_comparison_stats(
                    differences,
                    bootstrap_rounds=bootstrap_rounds,
                    bootstrap_seed=bootstrap_seed,
                    permutation_rounds=permutation_rounds,
                    permutation_seed=permutation_seed,
                ),
                "bootstrap_rounds": int(bootstrap_rounds),
                "bootstrap_seed": int(bootstrap_seed),
                "permutation_rounds": int(permutation_rounds),
                "permutation_seed": int(permutation_seed),
            }
        )
    for family, expected_size in (
        ("F1_canonical_learnability", 3),
        ("F2_canonical_history_increment", 3),
        ("F3_modern_baseline", 2),
    ):
        family_rows = [row for row in statistics if row["family"] == family]
        if len(family_rows) != expected_size:
            raise AssertionError(f"Frozen Holm family {family} changed size")
        adjusted = holm_adjust(
            {str(row["comparison"]): float(row["paired_sign_flip_p_raw"]) for row in family_rows}
        )
        for row in family_rows:
            row["holm_family_size"] = expected_size
            row["holm_adjusted_p"] = adjusted[str(row["comparison"])]
            row["reject_holm_0_05"] = bool(row["holm_adjusted_p"] <= 0.05)
    return paired_rows, statistics
