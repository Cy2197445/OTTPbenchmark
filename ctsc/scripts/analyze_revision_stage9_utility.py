"""Validate and aggregate offline alarm-budget results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np

from ctsc.precursor.revision import canonical_json_sha256, file_sha256, require_free_space
from ctsc.precursor.stage3 import bootstrap_mean_ci
from ctsc.precursor.stage5 import artifact_manifest, write_csv, write_json, write_sha256s
from ctsc.precursor.stage9 import STAGE9_BUDGETS, STAGE9_MODELS


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "manifests/experiment_registry.json"
DEFAULT_INPUT = ROOT / "runs/offline_utility/cell_runs"
DEFAULT_OUTPUT = ROOT / "results/offline_utility"
PLATFORM_TO_ENVIRONMENT = {
    "bullet_safety_gym_suite": "BulletSafetyGym",
    "safety_gymnasium_suite": "SafetyGymnasium",
    "safety_gymnasium_velocity_suite": "SafetyGymnasium",
    "safe_metadrive_suite": "MetaDrive",
}
MEASURES = (
    "test_step_alarm_rate",
    "test_episode_alarm_rate",
    "catastrophic_episode_recall",
    "episode_alarm_precision",
    "noncat_episode_false_alarm_rate",
    "recalled_first_alarm_steps_to_episode_end_median",
    "recalled_first_alarm_steps_to_episode_end_mean",
)
PAIRED_MEASURES = (
    "catastrophic_episode_recall",
    "episode_alarm_precision",
    "noncat_episode_false_alarm_rate",
    "recalled_first_alarm_steps_to_episode_end_median",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))






def _validate_result(row: dict[str, Any], *, task: str, platform: str, seed: int, model: str, gate_hash: str) -> None:
    if row.get("schema_version") != "ottp_revision_stage9_utility_result_v1":
        raise ValueError("Unknown Stage 9 result schema")
    if (row["task_name"], row["source_platform"], row["split_seed"], row["model_name"]) != (
        task, platform, seed, model
    ):
        raise ValueError("Stage 9 result identity mismatch")
    if row.get("threshold_source") != "validation_step_scores_only" or row.get("outcome_use") != "test_episode_evaluation_only":
        raise ValueError("Stage 9 threshold/outcome boundary mismatch")
    if row.get("lead_definition") != "first_alarm_steps_to_episode_end_not_catastrophe_onset":
        raise ValueError("Stage 9 lead definition mismatch")
    if row["parents"]["g5a_gate_sha256"] != gate_hash:
        raise ValueError("G5A gate hash changed after Stage 9 replay")
    if row["result_sha256"] != canonical_json_sha256(row["results"]):
        raise ValueError("Stage 9 result checksum mismatch")
    if set(row["results"]) != {str(value) for value in STAGE9_BUDGETS}:
        raise ValueError("Stage 9 budgets incomplete")
    for budget in STAGE9_BUDGETS:
        result = row["results"][str(budget)]
        if result["budget"] != budget or result["candidate_count"] <= 0:
            raise ValueError("Invalid Stage 9 budget/threshold selection")
        if not 0 <= result["validation_step_alarm_rate"] <= 1:
            raise ValueError("Invalid validation alarm rate")
        steps = int(result["test_step_count"])
        episodes = int(result["test_episode_count"])
        positives = int(result["test_catastrophic_episode_count"])
        negatives = int(result["test_noncatastrophic_episode_count"])
        alarmed = int(result["test_alarmed_episode_count"])
        recalled = int(result["test_recalled_catastrophic_episode_count"])
        false_alarms = int(result["test_false_alarm_episode_count"])
        if not steps > 0 or not episodes > 0 or episodes != positives + negatives or alarmed != recalled + false_alarms:
            raise ValueError("Stage 9 episode denominator accounting failed")
        if not 0 <= int(result["test_step_alarm_count"]) <= steps or not 0 <= alarmed <= episodes:
            raise ValueError("Stage 9 alarm counts out of range")
        if not 0 <= recalled <= positives or not 0 <= false_alarms <= negatives:
            raise ValueError("Stage 9 episode counts out of range")
        expected = {
            "test_step_alarm_rate": result["test_step_alarm_count"] / steps,
            "test_episode_alarm_rate": alarmed / episodes,
            "catastrophic_episode_recall": recalled / positives if positives else None,
            "episode_alarm_precision": recalled / alarmed if alarmed else None,
            "noncat_episode_false_alarm_rate": false_alarms / negatives if negatives else None,
        }
        for metric, exact in expected.items():
            observed = result[metric]
            if exact is None:
                if observed is not None:
                    raise ValueError(f"Undefined {metric} was reported as a number")
            elif observed is None or not np.isclose(float(observed), exact, rtol=0, atol=1e-12):
                raise ValueError(f"Stage 9 {metric} does not reconcile to its denominator")
        for metric in MEASURES:
            value = result[metric]
            if value is not None and not np.isfinite(value):
                raise ValueError(f"Non-finite Stage 9 {metric}")






def _mean_defined(rows: list[dict[str, Any]], measure: str) -> tuple[float | None, int]:
    values = [float(row[measure]) for row in rows if row[measure] is not None]
    return (float(np.mean(values)) if values else None, len(values))


def _task_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = (row["task_name"], row["model_name"], float(row["alarm_budget"]))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (task, model, budget), source in sorted(grouped.items()):
        if len(source) != 3 or {int(row["split_seed"]) for row in source} != {0, 1, 2}:
            raise ValueError("Task/model/budget split coverage is not 3/3")
        output: dict[str, Any] = {
            "task_name": task, "source_platform": source[0]["source_platform"],
            "simulation_environment": source[0]["simulation_environment"],
            "model_name": model, "alarm_budget": budget, "split_seed_count": 3,
        }
        for metric in MEASURES:
            value, count = _mean_defined(source, metric)
            output[metric + "_split_mean"] = value
            output[metric + "_defined_seed_count"] = count
        rows.append(output)
    if len(rows) != 38 * 3 * 3:
        raise ValueError("Task-level Stage 9 table is incomplete")
    return rows


def _group_summary(rows: list[dict[str, Any]], *, group_name: str, group_value: str, rounds: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {group_name: group_value, "task_count": len(rows)}
    for metric in MEASURES:
        values = [float(row[metric + "_split_mean"]) for row in rows if row[metric + "_split_mean"] is not None]
        output[metric + "_defined_task_count"] = len(values)
        output[metric + "_task_mean"] = float(np.mean(values)) if values else None
        if values:
            low, high = bootstrap_mean_ci(values, rounds=rounds, seed=seed)
        else:
            low, high = None, None
        output[metric + "_ci_low"] = low
        output[metric + "_ci_high"] = high
    return output


def _summaries(task_rows: list[dict[str, Any]], *, rounds: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall: list[dict[str, Any]] = []
    environment: list[dict[str, Any]] = []
    registry_group: list[dict[str, Any]] = []
    for model in STAGE9_MODELS:
        for budget in STAGE9_BUDGETS:
            selected = [row for row in task_rows if row["model_name"] == model and row["alarm_budget"] == budget]
            if len(selected) != 38:
                raise ValueError("Overall Stage 9 task coverage incomplete")
            overall.append({"model_name": model, "alarm_budget": budget, **_group_summary(selected, group_name="scope", group_value="all_38_tasks", rounds=rounds, seed=seed)})
            for simulation_environment in sorted({row["simulation_environment"] for row in selected}):
                subset = [row for row in selected if row["simulation_environment"] == simulation_environment]
                environment.append({"model_name": model, "alarm_budget": budget, **_group_summary(subset, group_name="simulation_environment", group_value=simulation_environment, rounds=rounds, seed=seed)})
            for source_platform in sorted({row["source_platform"] for row in selected}):
                subset = [row for row in selected if row["source_platform"] == source_platform]
                registry_group.append({"model_name": model, "alarm_budget": budget, **_group_summary(subset, group_name="registry_task_group", group_value=source_platform, rounds=rounds, seed=seed)})
    return overall, environment, registry_group


def _paired(task_rows: list[dict[str, Any]], *, rounds: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {(row["task_name"], row["model_name"], row["alarm_budget"]): row for row in task_rows}
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for budget in STAGE9_BUDGETS:
        for left, right in (("transformer", "gru"), ("transformer", "flat_mlp"), ("gru", "flat_mlp")):
            for measure in PAIRED_MEASURES:
                differences: list[float] = []
                for task in sorted({row["task_name"] for row in task_rows}):
                    l_value = indexed[(task, left, budget)][measure + "_split_mean"]
                    r_value = indexed[(task, right, budget)][measure + "_split_mean"]
                    raw_rows.append({
                        "task_name": task, "alarm_budget": budget, "metric": measure,
                        "left_model": left, "right_model": right,
                        "left_task_mean": l_value, "right_task_mean": r_value,
                        "paired_difference": float(l_value) - float(r_value) if l_value is not None and r_value is not None else None,
                    })
                    if l_value is not None and r_value is not None:
                        differences.append(float(l_value) - float(r_value))
                low, high = bootstrap_mean_ci(differences, rounds=rounds, seed=seed) if differences else (None, None)
                rows.append({
                    "alarm_budget": budget, "metric": measure, "difference": f"{left}_minus_{right}",
                    "paired_task_count": len(differences),
                    "mean_task_difference": float(np.mean(differences)) if differences else None,
                    "ci_low": low, "ci_high": high,
                    "positive_task_count": sum(value > 0 for value in differences),
                    "zero_task_count": sum(value == 0 for value in differences),
                    "negative_task_count": sum(value < 0 for value in differences),
                    "inference_status": "descriptive_bootstrap_not_confirmatory",
                })
    return rows, raw_rows




def _fmt_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def collect_results(input_root: Path) -> list[dict[str, Any]]:
    registry = _read_json(REGISTRY)
    rows = []
    for task_info in registry["tasks"]:
        task = task_info["task_name"]
        platform = task_info["source_platform"]
        for seed in (0, 1, 2):
            shared = []
            populations = set()
            for model in STAGE9_MODELS:
                record = _read_json(input_root / task / f"split_seed_{seed}" / f"{model}.json")
                _validate_result(record, task=task, platform=platform, seed=seed, model=model,
                                 gate_hash=record["parents"]["g5a_gate_sha256"])
                shared.append(record["parents"]["shared_preprocessing_sha256"])
                for budget in STAGE9_BUDGETS:
                    result = record["results"][str(budget)]
                    populations.add((result["test_step_count"], result["test_episode_count"],
                                     result["test_catastrophic_episode_count"]))
                    rows.append({"task_name": task, "source_platform": platform,
                                 "simulation_environment": PLATFORM_TO_ENVIRONMENT[platform],
                                 "split_seed": seed, "model_name": model, "alarm_budget": budget,
                                 "threshold": result["threshold"],
                                 "validation_step_alarm_rate": result["validation_step_alarm_rate"],
                                 **{k: v for k, v in result.items() if k not in {"threshold", "budget", "validation_step_alarm_rate"}},
                                 "source_result_sha256": record["result_sha256"]})
            if shared[0] != shared[1] or shared[0] != shared[2] or len(populations) != 1:
                raise ValueError(f"Model preprocessing or test populations differ: {task}/{seed}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-rounds", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=260922)
    args = parser.parse_args()
    from ctsc.scripts.run_benchmark import output_directory
    output = output_directory(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if args.bootstrap_rounds <= 0:
        raise ValueError("bootstrap-rounds must be positive")
    seed_rows = collect_results(args.input_dir.resolve())
    task_rows = _task_rows(seed_rows)
    overall, environment, groups = _summaries(task_rows, rounds=args.bootstrap_rounds, seed=args.bootstrap_seed)
    paired, raw = _paired(task_rows, rounds=args.bootstrap_rounds, seed=args.bootstrap_seed)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("seed_level_1026_budget_results", seed_rows), ("task_level_342_budget_results", task_rows),
                       ("overall_9_model_budget_summary", overall), ("simulation_environment_27_descriptive_summary", environment),
                       ("registry_task_group_36_descriptive_summary", groups), ("paired_model_difference_summary_descriptive", paired),
                       ("paired_model_task_differences_raw", raw)):
        write_csv(output / f"{name}.csv", rows)
    print(json.dumps({"seed_rows": len(seed_rows), "task_rows": len(task_rows), "output": str(output)}))


if __name__ == "__main__":
    main()






