'Aggregate the Unified38 offline actionability / downstream-utility study.'

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.actionability import (
    DEFAULT_ACTIONABILITY_BUDGETS,
    DEFAULT_ACTIONABILITY_MODELS,
    choose_threshold_for_alarm_budget,
    normalize_actionability_model_name,
    parse_budget_values,
    summarize_episode_alarm_utility,
)
from ctsc.precursor.public_source_catalog import dsrl_catalog_records


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default="work/reference_utility/predictions")
    parser.add_argument("--output-dir", type=str, default="work/reference_utility/aggregates")
    parser.add_argument("--task-names", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_ACTIONABILITY_MODELS))
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_ACTIONABILITY_BUDGETS))
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _source_platform_map() -> dict[str, str]:
    return {str(row["task_name"]): str(row["source_platform"]) for row in dsrl_catalog_records()}


def _rows_to_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        "scores": np.asarray([float(row["score_catastrophic"]) for row in rows], dtype=np.float64),
        "episode_uid": np.asarray([int(row["episode_uid"]) for row in rows], dtype=np.int64),
        "outcome_label": np.asarray([int(row["outcome_label"]) for row in rows], dtype=np.int64),
        "steps_remaining": np.asarray([float(row["steps_remaining"]) for row in rows], dtype=np.float64),
    }


def main() -> None:
    args = parse_args()
    input_root = _resolve(args.input_root)
    output_dir = _resolve(args.output_dir)
    task_names = (
        list(dict.fromkeys(str(name) for name in args.task_names))
        if args.task_names
        else [str(row["task_name"]) for row in dsrl_catalog_records()]
    )
    split_seeds = [int(seed) for seed in args.split_seeds]
    models = [normalize_actionability_model_name(model_name) for model_name in args.models]
    budgets = parse_budget_values(args.budgets)
    source_platform_map = _source_platform_map()

    transition_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    task_budget_groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)

    for task_name in task_names:
        for split_seed in split_seeds:
            observed_models: list[str] = []
            for model_name in models:
                model_dir = input_root / task_name / "canonical" / f"seed_{int(split_seed)}" / model_name
                rows_path = model_dir / "prediction_rows.csv"
                summary_path = model_dir / "prediction_summary.json"
                if not rows_path.exists():
                    continue
                prediction_rows = _read_csv(rows_path)
                prediction_summary = _load_json(summary_path) if summary_path.exists() else {}
                observed_models.append(model_name)
                transition_rows.append(
                    {
                        "task_name": task_name,
                        "slice_name": "canonical",
                        "split_seed": int(split_seed),
                        "model_name": model_name,
                        "source_platform": source_platform_map[task_name],
                        "row_count": int(prediction_summary.get("row_count", len(prediction_rows))),
                        "val_row_count": int(prediction_summary.get("split_counts", {}).get("val", 0)),
                        "test_row_count": int(prediction_summary.get("split_counts", {}).get("test", 0)),
                        "val_step_cat_prevalence": float(prediction_summary.get("val_step_cat_prevalence", 0.0)),
                        "val_outcome_cat_prevalence": float(prediction_summary.get("val_outcome_cat_prevalence", 0.0)),
                        "test_step_cat_prevalence": float(prediction_summary.get("test_step_cat_prevalence", 0.0)),
                        "test_outcome_cat_prevalence": float(prediction_summary.get("test_outcome_cat_prevalence", 0.0)),
                    }
                )
                split_groups = {
                    split_name: [row for row in prediction_rows if str(row["split_name"]) == split_name]
                    for split_name in ["val", "test"]
                }
                if not split_groups["val"] or not split_groups["test"]:
                    continue
                val_arrays = _rows_to_arrays(split_groups["val"])
                test_arrays = _rows_to_arrays(split_groups["test"])
                for budget in budgets:
                    threshold_selection = choose_threshold_for_alarm_budget(val_arrays["scores"], float(budget))
                    val_summary = summarize_episode_alarm_utility(
                        scores=val_arrays["scores"],
                        threshold=float(threshold_selection.threshold),
                        episode_ids=val_arrays["episode_uid"],
                        outcome_labels=val_arrays["outcome_label"],
                        steps_remaining=val_arrays["steps_remaining"],
                    )
                    test_summary = summarize_episode_alarm_utility(
                        scores=test_arrays["scores"],
                        threshold=float(threshold_selection.threshold),
                        episode_ids=test_arrays["episode_uid"],
                        outcome_labels=test_arrays["outcome_label"],
                        steps_remaining=test_arrays["steps_remaining"],
                    )
                    budget_row = {
                        "task_name": task_name,
                        "slice_name": "canonical",
                        "split_seed": int(split_seed),
                        "budget": float(budget),
                        "model_name": model_name,
                        "source_platform": source_platform_map[task_name],
                        "selected_threshold": float(threshold_selection.threshold),
                        "val_step_alarm_rate": float(val_summary.step_alarm_rate),
                        "val_episode_alarm_rate": float(val_summary.episode_alarm_rate),
                        "test_step_alarm_rate": float(test_summary.step_alarm_rate),
                        "test_episode_alarm_rate": float(test_summary.episode_alarm_rate),
                        "test_catastrophic_episode_recall": float(test_summary.catastrophic_episode_recall),
                        "test_noncat_episode_false_alarm_rate": float(test_summary.noncat_episode_false_alarm_rate),
                        "test_episode_alarm_precision": float(test_summary.episode_alarm_precision),
                        "test_recalled_catastrophic_lead_time_median": float(test_summary.recalled_catastrophic_lead_time_median),
                        "test_recalled_catastrophic_lead_time_mean": float(test_summary.recalled_catastrophic_lead_time_mean),
                        "test_catastrophic_episode_count": int(test_summary.catastrophic_episode_count),
                        "test_noncat_episode_count": int(test_summary.noncat_episode_count),
                        "test_alarmed_episode_count": int(test_summary.alarmed_episode_count),
                        "test_recalled_catastrophic_episode_count": int(test_summary.recalled_catastrophic_episode_count),
                    }
                    budget_rows.append(budget_row)
                    task_budget_groups[(task_name, model_name, float(budget))].append(budget_row)
            coverage_rows.append(
                {
                    "task_name": task_name,
                    "slice_name": "canonical",
                    "split_seed": int(split_seed),
                    "observed_models": json.dumps(sorted(observed_models)),
                    "coverage_status": "pass" if sorted(observed_models) == sorted(models) else "fail",
                }
            )

    task_pair_rows: list[dict[str, Any]] = []
    for (task_name, model_name, budget), rows in sorted(task_budget_groups.items()):
        task_pair_rows.append(
            {
                "task_name": task_name,
                "slice_name": "canonical",
                "budget": float(budget),
                "model_name": model_name,
                "source_platform": source_platform_map[task_name],
                "seed_count": len(rows),
                "selected_threshold_median": _median([float(row["selected_threshold"]) for row in rows]),
                "val_step_alarm_rate_median": _median([float(row["val_step_alarm_rate"]) for row in rows]),
                "test_step_alarm_rate_median": _median([float(row["test_step_alarm_rate"]) for row in rows]),
                "test_episode_alarm_rate_median": _median([float(row["test_episode_alarm_rate"]) for row in rows]),
                "test_catastrophic_episode_recall_median": _median([float(row["test_catastrophic_episode_recall"]) for row in rows]),
                "test_noncat_episode_false_alarm_rate_median": _median([float(row["test_noncat_episode_false_alarm_rate"]) for row in rows]),
                "test_episode_alarm_precision_median": _median([float(row["test_episode_alarm_precision"]) for row in rows]),
                "test_recalled_catastrophic_lead_time_median": _median([float(row["test_recalled_catastrophic_lead_time_median"]) for row in rows]),
            }
        )

    _write_csv(output_dir / "unified_actionability_transition_table.csv", transition_rows)
    _write_csv(output_dir / "unified_actionability_budget_table.csv", budget_rows)
    _write_csv(output_dir / "unified_actionability_task_pairs.csv", task_pair_rows)
    _write_csv(output_dir / "unified_actionability_coverage.csv", coverage_rows)
    _write_json(
        output_dir / "Unified38_actionability_offline_utility_summary.json",
        {
            "title": "Unified38 Offline Actionability / Downstream Utility Summary",
            "task_count": len(task_names),
            "split_seed_count": len(split_seeds),
            "model_count": len(models),
            "budget_count": len(budgets),
            "transition_row_count": len(transition_rows),
            "budget_row_count": len(budget_rows),
            "task_pair_row_count": len(task_pair_rows),
            "coverage_row_count": len(coverage_rows),
            "coverage_pass_count": int(sum(row["coverage_status"] == "pass" for row in coverage_rows)),
            "coverage_fail_count": int(sum(row["coverage_status"] == "fail" for row in coverage_rows)),
        },
    )
    print(
        json.dumps(
            {
                "transition_table": str(output_dir / "unified_actionability_transition_table.csv"),
                "budget_table": str(output_dir / "unified_actionability_budget_table.csv"),
                "task_pairs": str(output_dir / "unified_actionability_task_pairs.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
