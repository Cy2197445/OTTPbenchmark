'Frozen Stage 4 target-robustness matrix and read-only analysis helpers.'

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

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
MODELS = ("gru", "flat_mlp")
PANEL_TASKS = (
    "BallRun",
    "AntCircle",
    "mediumsparse",
    "easydense",
    "PointCircle2",
    "CarPush2",
    "SwimmerVelocity",
    "AntVelocity",
)


@dataclass(frozen=True)
class FrozenCondition:
    condition_id: str
    matrix: str
    target_variant: str
    gamma: float
    q_low: float
    q_high: float
    feature_profile: str
    reused_g0: bool = False
    role: str = "confirmatory"


CORE_CONDITIONS = (
    FrozenCondition(
        "core_current_inclusive_full",
        "core",
        "current_inclusive",
        0.99,
        0.75,
        0.90,
        "full",
        reused_g0=True,
    ),
    FrozenCondition(
        "core_current_inclusive_masked",
        "core",
        "current_inclusive",
        0.99,
        0.75,
        0.90,
        "history_without_current_cost",
    ),
    FrozenCondition("core_future_only_full", "core", "future_only", 0.99, 0.75, 0.90, "full"),
    FrozenCondition(
        "core_future_only_masked",
        "core",
        "future_only",
        0.99,
        0.75,
        0.90,
        "history_without_current_cost",
    ),
)

SENSITIVITY_CONDITIONS = (
    FrozenCondition(
        "sens_g099_q75_q90",
        "sensitivity",
        "current_inclusive",
        0.99,
        0.75,
        0.90,
        "full",
        reused_g0=True,
        role="canonical_reference",
    ),
    FrozenCondition(
        "sens_g095_q75_q90",
        "sensitivity",
        "current_inclusive",
        0.95,
        0.75,
        0.90,
        "full",
        role="gamma_sensitivity",
    ),
    FrozenCondition(
        "sens_g097_q75_q90",
        "sensitivity",
        "current_inclusive",
        0.97,
        0.75,
        0.90,
        "full",
        role="gamma_sensitivity",
    ),
    FrozenCondition(
        "sens_g099_q70_q85",
        "sensitivity",
        "current_inclusive",
        0.99,
        0.70,
        0.85,
        "full",
        role="quantile_sensitivity",
    ),
    FrozenCondition(
        "sens_g099_q80_q95",
        "sensitivity",
        "current_inclusive",
        0.99,
        0.80,
        0.95,
        "full",
        role="quantile_sensitivity",
    ),
)

ALL_CONDITIONS = (*CORE_CONDITIONS, *SENSITIVITY_CONDITIONS)
NEW_CONDITIONS = tuple(condition for condition in ALL_CONDITIONS if not condition.reused_g0)
CONDITIONS_BY_ID = {condition.condition_id: condition for condition in ALL_CONDITIONS}


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or [])
    if not fields:
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(str(field))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def frozen_matrix_manifest(tasks: Sequence[str]) -> dict[str, Any]:
    ordered_tasks = tuple(str(task) for task in tasks)
    if len(ordered_tasks) != TASKS_EXPECTED or len(set(ordered_tasks)) != TASKS_EXPECTED:
        raise ValueError(f"Stage 4 requires exactly {TASKS_EXPECTED} unique tasks")
    if not set(PANEL_TASKS).issubset(ordered_tasks):
        raise ValueError("Frozen eight-task sensitivity panel is not contained in the task list")
    payload = {
        "schema_version": "ottp_stage4_frozen_matrix_v1",
        "freeze_id": FREEZE_ID,
        "tasks": list(ordered_tasks),
        "split_seeds": list(SPLIT_SEEDS),
        "models": list(MODELS),
        "panel_tasks": list(PANEL_TASKS),
        "conditions": [asdict(condition) for condition in ALL_CONDITIONS],
        "expected": {
            "core_model_results": TASKS_EXPECTED * len(SPLIT_SEEDS) * len(CORE_CONDITIONS) * len(MODELS),
            "core_reused_g0_model_results": TASKS_EXPECTED * len(SPLIT_SEEDS) * len(MODELS),
            "core_new_model_results": TASKS_EXPECTED * len(SPLIT_SEEDS) * 3 * len(MODELS),
            "sensitivity_model_results": len(PANEL_TASKS)
            * len(SPLIT_SEEDS)
            * len(SENSITIVITY_CONDITIONS)
            * len(MODELS),
            "sensitivity_reused_g0_model_results": len(PANEL_TASKS) * len(SPLIT_SEEDS) * len(MODELS),
            "sensitivity_new_model_results": len(PANEL_TASKS) * len(SPLIT_SEEDS) * 4 * len(MODELS),
            "new_suite_jobs": TASKS_EXPECTED * len(SPLIT_SEEDS) * 3
            + len(PANEL_TASKS) * len(SPLIT_SEEDS) * 4,
            "new_model_results": TASKS_EXPECTED * len(SPLIT_SEEDS) * 3 * len(MODELS)
            + len(PANEL_TASKS) * len(SPLIT_SEEDS) * 4 * len(MODELS),
        },
        "excluded": [
            "gamma-by-quantile Cartesian grid",
            "post-hoc task selection",
            "post-hoc seed selection",
            "Transformer in Stage 4",
        ],
    }
    payload["matrix_sha256"] = canonical_json_sha256(payload)
    return payload


def condition_tasks(condition: FrozenCondition, all_tasks: Sequence[str]) -> tuple[str, ...]:
    return tuple(PANEL_TASKS) if condition.matrix == "sensitivity" else tuple(str(task) for task in all_tasks)


def expected_new_jobs(tasks: Sequence[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for condition in NEW_CONDITIONS:
        for task_name in condition_tasks(condition, tasks):
            for split_seed in SPLIT_SEEDS:
                jobs.append(
                    {
                        "job_id": f"{condition.condition_id}__{task_name}__ss{split_seed}",
                        "condition_id": condition.condition_id,
                        "matrix": condition.matrix,
                        "task_name": task_name,
                        "split_seed": int(split_seed),
                    }
                )
    return jobs


def sidecar_name(condition: FrozenCondition) -> str | None:
    if condition.target_variant == "current_inclusive" and math.isclose(condition.gamma, 0.99):
        return None
    gamma_token = str(condition.gamma).replace(".", "p")
    return f"{condition.target_variant}_g{gamma_token}_q{int(condition.q_low * 100)}_q{int(condition.q_high * 100)}.npz"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_model(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _suite_candidates(experiment_root: Path, condition: FrozenCondition, task_name: str, split_seed: int) -> list[Path]:
    base = experiment_root / "task_runs" / condition.matrix / task_name / condition.condition_id
    return sorted(base.glob(f"attempt-*/split_seed_{split_seed}/train_seed_{split_seed}/suite_summary.json"))


def resolve_successful_suite(
    experiment_root: str | Path,
    condition: FrozenCondition,
    task_name: str,
    split_seed: int,
) -> Path:
    candidates = _suite_candidates(Path(experiment_root), condition, task_name, split_seed)
    valid: list[Path] = []
    for path in candidates:
        payload = _load_json(path)
        if (
            str(payload.get("target_variant")) == condition.target_variant
            and math.isclose(float(payload.get("gamma")), condition.gamma, abs_tol=1e-12)
            and str(payload.get("feature_profile")) == condition.feature_profile
            and int(payload.get("split_seed")) == int(split_seed)
            and set(_canonical_model(name) for name in payload.get("models", {})) == set(MODELS)
        ):
            valid.append(path)
    if not valid:
        raise FileNotFoundError(
            f"Missing successful Stage 4 suite: {condition.condition_id}/{task_name}/seed_{split_seed}"
        )
    return valid[-1]


def _metric_row(
    *,
    metrics_path: Path,
    task_name: str,
    source_platform: str,
    split_seed: int,
    condition: FrozenCondition,
    reused_g0: bool,
) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    model_name = _canonical_model(str(metrics["model_name"]))
    label_protocol = dict(metrics.get("label_protocol") or {})
    expected_protocol = {
        "target_variant": condition.target_variant,
        "gamma": condition.gamma,
        "split_seed": int(split_seed),
        "step_q_low": condition.q_low,
        "step_q_high": condition.q_high,
        "outcome_q_low": condition.q_low,
        "outcome_q_high": condition.q_high,
    }
    if model_name not in MODELS:
        raise ValueError(f"Unexpected Stage 4 model in {metrics_path}: {model_name}")
    if str(metrics.get("feature_profile")) != condition.feature_profile:
        raise ValueError(f"Feature-profile mismatch in {metrics_path}")
    for field, expected in expected_protocol.items():
        observed = label_protocol.get(field, metrics.get(field))
        # The frozen G0 canonical metrics predate explicit storage of these
        # three fields.  Their values are fixed by the canonical directory,
        # split path, and Stage 1 protocol.  Keep this compatibility exception
        # narrow: a present value must still match, and every newly trained
        # Stage 4 result must carry all fields explicitly.
        if reused_g0 and observed is None and field in {"target_variant", "gamma", "split_seed"}:
            continue
        if isinstance(expected, float):
            if observed is None or not math.isclose(float(observed), expected, abs_tol=1e-12):
                raise ValueError(f"Protocol mismatch in {metrics_path}: {field}={observed} != {expected}")
        elif observed != expected:
            raise ValueError(f"Protocol mismatch in {metrics_path}: {field}={observed} != {expected}")
    step = metrics["label_step"]["test"]["overall"]
    outcome = metrics["label_outcome"]["test"]["overall"]
    step_auprc = float(step["catastrophic_auprc"])
    step_prevalence = float(step["catastrophic_prevalence"])
    numeric_values = {
        "step_auprc": step_auprc,
        "matched_step_prevalence": step_prevalence,
        "outcome_auprc": float(outcome["catastrophic_auprc"]),
        "matched_outcome_prevalence": float(outcome["catastrophic_prevalence"]),
        "step_ece": float(step["catastrophic_ece"]),
        "step_lead_median": float(step["lead_time_median_at_0.5"]),
    }
    if not all(math.isfinite(value) for value in numeric_values.values()):
        raise ValueError(f"Non-finite Stage 4 metric in {metrics_path}: {numeric_values}")
    return {
        "matrix": condition.matrix,
        "condition_id": condition.condition_id,
        "condition_role": condition.role,
        "task_name": task_name,
        "source_platform": source_platform,
        "split_seed": int(split_seed),
        "model_name": model_name,
        "target_variant": condition.target_variant,
        "gamma": condition.gamma,
        "q_low": condition.q_low,
        "q_high": condition.q_high,
        "feature_profile": condition.feature_profile,
        "step_auprc": step_auprc,
        "matched_step_prevalence": step_prevalence,
        "step_auprc_minus_prevalence": step_auprc - step_prevalence,
        "outcome_auprc": numeric_values["outcome_auprc"],
        "matched_outcome_prevalence": numeric_values["matched_outcome_prevalence"],
        "step_ece": numeric_values["step_ece"],
        "step_lead_median": numeric_values["step_lead_median"],
        "reused_g0": bool(reused_g0),
        "metrics_path": str(metrics_path),
        "metrics_sha256": file_sha256(metrics_path),
    }


def collect_result_rows(
    *,
    experiment_root: str | Path,
    g0_root: str | Path,
    task_platforms: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    experiment = Path(experiment_root).resolve()
    g0 = Path(g0_root).resolve()
    tasks = tuple(sorted(task_platforms))
    rows: list[dict[str, Any]] = []
    label_audits: list[dict[str, Any]] = []
    for condition in ALL_CONDITIONS:
        for task_name in condition_tasks(condition, tasks):
            for split_seed in SPLIT_SEEDS:
                if condition.reused_g0:
                    for model_name in MODELS:
                        metrics_path = (
                            g0
                            / "task_runs"
                            / task_name
                            / "canonical"
                            / f"seed_{split_seed}"
                            / model_name
                            / "metrics.json"
                        )
                        if not metrics_path.is_file():
                            raise FileNotFoundError(f"Missing G0 reuse metrics: {metrics_path}")
                        rows.append(
                            _metric_row(
                                metrics_path=metrics_path,
                                task_name=task_name,
                                source_platform=task_platforms[task_name],
                                split_seed=split_seed,
                                condition=condition,
                                reused_g0=True,
                            )
                        )
                    audit_source = resolve_successful_suite(
                        experiment,
                        CONDITIONS_BY_ID["core_current_inclusive_masked"],
                        task_name,
                        split_seed,
                    )
                    audit_payload = _load_json(audit_source)
                    label_audits.append(
                        {
                            "matrix": condition.matrix,
                            "condition_id": condition.condition_id,
                            "task_name": task_name,
                            "split_seed": split_seed,
                            "audit_source_suite": str(audit_source),
                            "audit_reused_from_equivalent_label_protocol": True,
                            "label_audit": audit_payload["label_audit"],
                        }
                    )
                    continue
                suite_path = resolve_successful_suite(experiment, condition, task_name, split_seed)
                suite = _load_json(suite_path)
                label_audits.append(
                    {
                        "matrix": condition.matrix,
                        "condition_id": condition.condition_id,
                        "task_name": task_name,
                        "split_seed": split_seed,
                        "audit_source_suite": str(suite_path),
                        "audit_reused_from_equivalent_label_protocol": False,
                        "label_audit": suite["label_audit"],
                    }
                )
                for model_summary in suite["models"].values():
                    metrics_path = Path(str(model_summary["metrics_path"]))
                    if not metrics_path.is_absolute():
                        metrics_path = (experiment / metrics_path).resolve()
                    rows.append(
                        _metric_row(
                            metrics_path=metrics_path,
                            task_name=task_name,
                            source_platform=task_platforms[task_name],
                            split_seed=split_seed,
                            condition=condition,
                            reused_g0=False,
                        )
                    )
    expected_rows = TASKS_EXPECTED * len(SPLIT_SEEDS) * len(CORE_CONDITIONS) * len(MODELS)
    expected_rows += len(PANEL_TASKS) * len(SPLIT_SEEDS) * len(SENSITIVITY_CONDITIONS) * len(MODELS)
    if len(rows) != expected_rows:
        raise ValueError(f"Stage 4 result coverage mismatch: {len(rows)} != {expected_rows}")
    return rows, label_audits


def task_level_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        key = (str(row["matrix"]), str(row["condition_id"]), str(row["task_name"]), str(row["model_name"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        if sorted(int(row["split_seed"]) for row in rows) != list(SPLIT_SEEDS):
            raise ValueError(f"Split-seed coverage mismatch for {key}")
        first = rows[0]
        mean = lambda field: float(np.mean([float(row[field]) for row in rows]))
        output.append(
            {
                "matrix": key[0],
                "condition_id": key[1],
                "task_name": key[2],
                "model_name": key[3],
                "source_platform": first["source_platform"],
                "target_variant": first["target_variant"],
                "gamma": float(first["gamma"]),
                "q_low": float(first["q_low"]),
                "q_high": float(first["q_high"]),
                "feature_profile": first["feature_profile"],
                "seed_aggregation": "arithmetic_mean_split_seeds_0_1_2",
                "step_auprc": mean("step_auprc"),
                "matched_step_prevalence": mean("matched_step_prevalence"),
                "step_auprc_minus_prevalence": mean("step_auprc_minus_prevalence"),
                "outcome_auprc": mean("outcome_auprc"),
                "matched_outcome_prevalence": mean("matched_outcome_prevalence"),
                "step_ece": mean("step_ece"),
                "step_lead_median": mean("step_lead_median"),
                "reused_g0": bool(first["reused_g0"]),
            }
        )
    return output


def cell_summary_rows(task_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    'Summarize raw AUPRC, matched prevalence, and lift over task means.'

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in task_rows:
        key = (str(row["matrix"]), str(row["condition_id"]), str(row["model_name"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (matrix, condition_id, model_name), rows in sorted(grouped.items()):
        task_names = [str(row["task_name"]) for row in rows]
        if len(task_names) != len(set(task_names)):
            raise ValueError(f"Duplicate task rows in Stage 4 cell: {(matrix, condition_id, model_name)}")
        auprc = quantile_summary([float(row["step_auprc"]) for row in rows])
        prevalence = quantile_summary([float(row["matched_step_prevalence"]) for row in rows])
        lift = quantile_summary([float(row["step_auprc_minus_prevalence"]) for row in rows])
        output.append(
            {
                "matrix": matrix,
                "condition_id": condition_id,
                "model_name": model_name,
                "task_count": len(rows),
                "step_auprc_mean": auprc["mean"],
                "step_auprc_median": auprc["median"],
                "step_auprc_iqr": auprc["iqr"],
                "matched_step_prevalence_mean": prevalence["mean"],
                "matched_step_prevalence_median": prevalence["median"],
                "matched_step_prevalence_iqr": prevalence["iqr"],
                "baseline_adjusted_lift_mean": lift["mean"],
                "baseline_adjusted_lift_median": lift["median"],
                "baseline_adjusted_lift_iqr": lift["iqr"],
                "positive_lift_task_count": lift["positive_count"],
                "positive_lift_task_proportion": lift["positive_proportion"],
            }
        )
    return output


def platform_equal_weighted_rows(
    paired_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    'Return per-platform paired means and equal-platform-weight summaries.'

    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in paired_rows:
        key = (
            str(row["family"]),
            str(row["comparison"]),
            str(row["model_name"]),
            str(row["source_platform"]),
        )
        grouped.setdefault(key, []).append(float(row["paired_difference"]))
    details: list[dict[str, Any]] = []
    for (family, comparison, model_name, platform), values in sorted(grouped.items()):
        summary = quantile_summary(values)
        details.append(
            {
                "family": family,
                "comparison": comparison,
                "model_name": model_name,
                "source_platform": platform,
                "task_count": summary["n"],
                "platform_mean_paired_difference": summary["mean"],
                "platform_median_paired_difference": summary["median"],
                "positive_task_count": summary["positive_count"],
                "positive_task_proportion": summary["positive_proportion"],
            }
        )
    summaries: list[dict[str, Any]] = []
    comparison_keys = sorted({(row["family"], row["comparison"], row["model_name"]) for row in details})
    for family, comparison, model_name in comparison_keys:
        selected = [
            row
            for row in details
            if (row["family"], row["comparison"], row["model_name"])
            == (family, comparison, model_name)
        ]
        platform_means = [float(row["platform_mean_paired_difference"]) for row in selected]
        summary = quantile_summary(platform_means)
        summaries.append(
            {
                "family": family,
                "comparison": comparison,
                "model_name": model_name,
                "platform_count": len(selected),
                "task_count": sum(int(row["task_count"]) for row in selected),
                "platform_equal_weighted_mean_difference": summary["mean"],
                "median_platform_mean_difference": summary["median"],
                "iqr_platform_mean_difference": summary["iqr"],
                "positive_platform_count": summary["positive_count"],
                "positive_platform_proportion": summary["positive_proportion"],
                "role": "descriptive_only_not_a_hypothesis_family",
            }
        )
    return details, summaries


def _stats(values: Sequence[float], *, bootstrap_rounds: int, bootstrap_seed: int, permutation_rounds: int, permutation_seed: int) -> dict[str, Any]:
    summary = quantile_summary(values)
    ci_low, ci_high = bootstrap_mean_ci(values, rounds=bootstrap_rounds, seed=bootstrap_seed)
    exact = len(values) <= 20
    return {
        **summary,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "paired_sign_flip_p_raw": paired_sign_flip_pvalue(values, rounds=permutation_rounds, seed=permutation_seed),
        "permutation_mode": "exact_all_sign_patterns" if exact else "monte_carlo_add_one",
        "permutation_effective_draws": (1 << len(values)) if exact else int(permutation_rounds),
        "permutation_add_one_correction": not exact,
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
    index = {
        (str(row["condition_id"]), str(row["model_name"]), str(row["task_name"])): row
        for row in task_rows
    }
    tasks = sorted({str(row["task_name"]) for row in task_rows if row["matrix"] == "core"})
    definitions: list[tuple[str, str, str, str, str]] = []
    for target, full_id, masked_id in (
        ("current_inclusive", "core_current_inclusive_full", "core_current_inclusive_masked"),
        ("future_only", "core_future_only_full", "core_future_only_masked"),
    ):
        for model in MODELS:
            definitions.append(("F4_anti_shortcut", f"{target}_{model}_full_minus_masked", model, full_id, masked_id))
    for model in MODELS:
        definitions.append(
            (
                "F4_anti_shortcut",
                f"future_only_{model}_masked_minus_prevalence",
                model,
                "core_future_only_masked",
                "matched_prevalence",
            )
        )
    for condition in SENSITIVITY_CONDITIONS:
        if condition.reused_g0:
            continue
        for model in MODELS:
            definitions.append(
                (
                    "F5_parameter_sensitivity",
                    f"{condition.condition_id}_{model}_lift_minus_canonical_lift",
                    model,
                    condition.condition_id,
                    "sens_g099_q75_q90",
                )
            )

    paired_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    for family, comparison, model, left_id, right_id in definitions:
        comparison_tasks = tasks if family == "F4_anti_shortcut" else list(PANEL_TASKS)
        differences: list[float] = []
        for task_name in comparison_tasks:
            left = index[(left_id, model, task_name)]
            if right_id == "matched_prevalence":
                left_value = float(left["step_auprc"])
                right_value = float(left["matched_step_prevalence"])
                value_kind = "raw_step_auprc_minus_matched_step_prevalence"
            elif family == "F5_parameter_sensitivity":
                right = index[(right_id, model, task_name)]
                left_value = float(left["step_auprc_minus_prevalence"])
                right_value = float(right["step_auprc_minus_prevalence"])
                value_kind = "baseline_adjusted_lift_difference"
            else:
                right = index[(right_id, model, task_name)]
                left_value = float(left["step_auprc"])
                right_value = float(right["step_auprc"])
                value_kind = "raw_step_auprc_difference_same_target"
            difference = left_value - right_value
            differences.append(difference)
            paired_rows.append(
                {
                    "family": family,
                    "comparison": comparison,
                    "model_name": model,
                    "task_name": task_name,
                    "source_platform": left["source_platform"],
                    "left_condition_id": left_id,
                    "right_condition_id": right_id,
                    "value_kind": value_kind,
                    "left_value": left_value,
                    "right_value": right_value,
                    "paired_difference": difference,
                }
            )
        stats_rows.append(
            {
                "family": family,
                "comparison": comparison,
                "model_name": model,
                **_stats(
                    differences,
                    bootstrap_rounds=bootstrap_rounds,
                    bootstrap_seed=bootstrap_seed,
                    permutation_rounds=permutation_rounds,
                    permutation_seed=permutation_seed,
                ),
                "bootstrap_rounds": bootstrap_rounds,
                "bootstrap_seed": bootstrap_seed,
                "permutation_rounds": permutation_rounds,
                "permutation_seed": permutation_seed,
            }
        )
    for family in ("F4_anti_shortcut", "F5_parameter_sensitivity"):
        family_rows = [row for row in stats_rows if row["family"] == family]
        adjusted = holm_adjust({str(row["comparison"]): float(row["paired_sign_flip_p_raw"]) for row in family_rows})
        for row in family_rows:
            row["holm_family_size"] = len(family_rows)
            row["holm_adjusted_p"] = adjusted[str(row["comparison"])]
            row["reject_holm_0_05"] = bool(row["holm_adjusted_p"] <= 0.05)
    return paired_rows, stats_rows


def build_current_reuse_parity_audit(
    *,
    experiment_root: str | Path,
    g0_root: str | Path,
    tasks: Sequence[str],
    split_seeds: Sequence[int] = SPLIT_SEEDS,
) -> list[dict[str, Any]]:
    experiment = Path(experiment_root).resolve()
    g0 = Path(g0_root).resolve()
    condition = CONDITIONS_BY_ID["core_current_inclusive_masked"]
    rows: list[dict[str, Any]] = []
    for task_name in sorted(tasks):
        for split_seed in split_seeds:
            suite_path = resolve_successful_suite(experiment, condition, task_name, split_seed)
            suite = _load_json(suite_path)
            masked_bundle_path = Path(str(suite["models"]["gru"]["bundle_path"]))
            g0_bundle_path = g0 / "task_runs" / task_name / "canonical" / f"seed_{split_seed}" / "gru" / "precursor_model.pt"
            masked = torch.load(masked_bundle_path, map_location="cpu", weights_only=False)
            original = torch.load(g0_bundle_path, map_location="cpu", weights_only=False)
            split_match = all(
                original["split_manifest"].get(field) == masked["split_manifest"].get(field)
                for field in ("train_groups", "val_groups", "test_groups")
            )
            thresholds_match = original["step_thresholds"] == masked["step_thresholds"]
            outcome_thresholds_match = original["outcome_thresholds"] == masked["outcome_thresholds"]
            normalization_match = np.array_equal(
                np.asarray(original["normalization_mean"]), np.asarray(masked["normalization_mean"])
            ) and np.array_equal(np.asarray(original["normalization_std"]), np.asarray(masked["normalization_std"]))
            status = split_match and thresholds_match and outcome_thresholds_match and normalization_match
            rows.append(
                {
                    "task_name": task_name,
                    "split_seed": split_seed,
                    "g0_bundle_path": str(g0_bundle_path),
                    "masked_bundle_path": str(masked_bundle_path),
                    "split_membership_exact_match": split_match,
                    "split_manifest_metadata_extension_allowed": True,
                    "step_thresholds_exact_match": thresholds_match,
                    "outcome_thresholds_exact_match": outcome_thresholds_match,
                    "normalization_exact_match": normalization_match,
                    "current_inclusive_canonical_reuse_status": "PASS" if status else "FAIL",
                }
            )
    if not all(row["current_inclusive_canonical_reuse_status"] == "PASS" for row in rows):
        raise ValueError("Current-inclusive canonical preprocessing parity failed")
    return rows


def artifact_manifest(root: str | Path) -> dict[str, Any]:
    output = Path(root)
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in {"artifact_manifest.json", "SHA256SUMS"})
    payload = {
        "schema_version": "ottp_stage4_artifact_manifest_v1",
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
        "\n".join(
            f"{file_sha256(path)}  {str(path.relative_to(output)).replace(chr(92), '/')}" for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    return target
