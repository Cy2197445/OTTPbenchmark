'Aggregate current-branch Unified38 fresh-rerun suite summaries.'

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.public_source_catalog import dsrl_catalog_records


REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_BASELINES = ["gru", "temporal-cnn"]
NONSEQUENTIAL_BASELINES = ["flat-mlp", "linear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default="work/reference_diagnostics/task_runs")
    parser.add_argument("--output-dir", type=str, default="work/reference_diagnostics/aggregates")
    parser.add_argument("--task-names", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--slices", type=str, nargs="+", default=["canonical", "sensitivity"])
    return parser.parse_args()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _median_metric(summaries: list[dict[str, Any]], key: str) -> float:
    return float(median(float(summary[key]) for summary in summaries))


def _best_sequence_baseline(models: dict[str, dict[str, Any]]) -> str:
    return max(SEQUENCE_BASELINES, key=lambda name: float(models[name]["step_auprc"]))


def _source_platform_map() -> dict[str, str]:
    return {str(row["task_name"]): str(row["source_platform"]) for row in dsrl_catalog_records()}


def main() -> None:
    args = parse_args()
    input_root = REPO_ROOT / args.input_root
    output_dir = REPO_ROOT / args.output_dir
    task_names = (
        list(dict.fromkeys(str(name) for name in args.task_names))
        if args.task_names
        else [str(row["task_name"]) for row in dsrl_catalog_records()]
    )
    split_seeds = [int(seed) for seed in args.split_seeds]
    slices = [str(name) for name in args.slices]
    source_platform_map = _source_platform_map()

    suite_payloads: dict[tuple[str, str, int], dict[str, Any]] = {}
    for task_name in task_names:
        for slice_name in slices:
            for split_seed in split_seeds:
                summary_path = input_root / task_name / slice_name / f"seed_{int(split_seed)}" / "suite_summary.json"
                if summary_path.exists():
                    suite_payloads[(task_name, slice_name, int(split_seed))] = _load_json(summary_path)

    seed_model_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    aggregate_summary: list[dict[str, Any]] = []
    for task_name in task_names:
        observed_seeds: set[int] = set()
        slice_model_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for slice_name in slices:
            for split_seed in split_seeds:
                suite = suite_payloads.get((task_name, slice_name, int(split_seed)))
                if suite is None:
                    continue
                observed_seeds.add(int(split_seed))
                for model_name, metrics in suite["models"].items():
                    seed_model_rows.append(
                        {
                            "task_name": task_name,
                            "branch": "current_branch",
                            "slice_name": slice_name,
                            "source_platform": source_platform_map[task_name],
                            "split_seed": int(split_seed),
                            "model_name": str(model_name).replace("-", "_"),
                            "step_auprc": float(metrics["test_step_catastrophic_auprc"]),
                            "outcome_auprc": float(metrics["test_outcome_catastrophic_auprc"]),
                            "ece": float(metrics["test_catastrophic_ece"]),
                            "normalized_lead_time": float(metrics["test_lead_time_median_at_0.5"]),
                            "raw_step_lead_time": float(metrics["test_lead_time_median_at_0.5"]),
                        }
                    )
                    slice_model_groups[(slice_name, str(model_name))].append(
                        {
                            "step_auprc": float(metrics["test_step_catastrophic_auprc"]),
                            "outcome_auprc": float(metrics["test_outcome_catastrophic_auprc"]),
                            "ece": float(metrics["test_catastrophic_ece"]),
                            "normalized_lead_time": float(metrics["test_lead_time_median_at_0.5"]),
                            "raw_step_lead_time": float(metrics["test_lead_time_median_at_0.5"]),
                        }
                    )

        if not slice_model_groups:
            continue

        slice_level: dict[str, dict[str, Any]] = {}
        for slice_name in slices:
            models: dict[str, dict[str, Any]] = {}
            for model_name in [*SEQUENCE_BASELINES, *NONSEQUENTIAL_BASELINES, "prevalence"]:
                summaries = slice_model_groups.get((slice_name, model_name))
                if not summaries:
                    continue
                models[model_name] = {
                    "step_auprc": _median_metric(summaries, "step_auprc"),
                    "outcome_auprc": _median_metric(summaries, "outcome_auprc"),
                    "ece": _median_metric(summaries, "ece"),
                    "normalized_lead_time": _median_metric(summaries, "normalized_lead_time"),
                    "raw_step_lead_time": _median_metric(summaries, "raw_step_lead_time"),
                }
            if models:
                slice_level[slice_name] = models

        if "canonical" not in slice_level or "sensitivity" not in slice_level:
            continue

        def _row_for(slice_name: str) -> dict[str, Any]:
            models = slice_level[slice_name]
            best_seq = _best_sequence_baseline(models)
            best_step = float(models[best_seq]["step_auprc"])
            return {
                "best_sequence_baseline": str(best_seq).replace("-", "_"),
                "gap_vs_flat": best_step - float(models["flat-mlp"]["step_auprc"]),
                "gap_vs_linear": best_step - float(models["linear"]["step_auprc"]),
                "normalized_lead_time": float(models[best_seq]["normalized_lead_time"]),
                "raw_step_lead_time": float(models[best_seq]["raw_step_lead_time"]),
                "strict_task_level_sequence_support": all(
                    float(models[seq_name]["step_auprc"]) > float(models[nonseq_name]["step_auprc"])
                    for seq_name in SEQUENCE_BASELINES
                    for nonseq_name in NONSEQUENTIAL_BASELINES
                ),
            }

        canonical_row = _row_for("canonical")
        sensitivity_row = _row_for("sensitivity")
        registry_rows.append(
            {
                "task_name": task_name,
                "branch": "current_branch",
                "source_platform": source_platform_map[task_name],
                "best_sequence_baseline": canonical_row["best_sequence_baseline"],
                "canonical_gap_vs_flat": canonical_row["gap_vs_flat"],
                "canonical_gap_vs_linear": canonical_row["gap_vs_linear"],
                "canonical_normalized_lead_time": canonical_row["normalized_lead_time"],
                "canonical_raw_step_lead_time": canonical_row["raw_step_lead_time"],
                "canonical_strict_task_level_sequence_support": canonical_row["strict_task_level_sequence_support"],
                "sensitivity_best_sequence_baseline": sensitivity_row["best_sequence_baseline"],
                "sensitivity_gap_vs_flat": sensitivity_row["gap_vs_flat"],
                "sensitivity_gap_vs_linear": sensitivity_row["gap_vs_linear"],
                "sensitivity_normalized_lead_time": sensitivity_row["normalized_lead_time"],
                "sensitivity_raw_step_lead_time": sensitivity_row["raw_step_lead_time"],
                "sensitivity_strict_task_level_sequence_support": sensitivity_row["strict_task_level_sequence_support"],
                "canonical_majority_wins": "",
                "split_seeds_observed": json.dumps(sorted(observed_seeds)),
                "split_seed_coverage_status": "pass" if sorted(observed_seeds) == split_seeds else "fail",
            }
        )
        aggregate_summary.append(
            {
                "task_name": task_name,
                "source_platform": source_platform_map[task_name],
                "observed_split_seeds": sorted(observed_seeds),
                "split_seed_coverage_status": "pass" if sorted(observed_seeds) == split_seeds else "fail",
            }
        )

    if not registry_rows:
        raise ValueError("No completed baseline suites found; use analyze_benchmark.py for run_benchmark.py outputs")
    _write_csv(output_dir / "unified_seed_model_table_current.csv", seed_model_rows)
    _write_csv(output_dir / "unified_task_registry_current.csv", registry_rows)
    _write_json(
        output_dir / "Unified38_current_run_summary.json",
        {
            "title": "Unified38 Current Run Summary",
            "task_count": len(task_names),
            "completed_task_count": len(registry_rows),
            "split_seeds": split_seeds,
            "slices": slices,
            "seed_model_row_count": len(seed_model_rows),
            "task_registry_row_count": len(registry_rows),
            "tasks": aggregate_summary,
        },
    )
    print(
        json.dumps(
            {
                "task_registry": str(output_dir / "unified_task_registry_current.csv"),
                "seed_model_table": str(output_dir / "unified_seed_model_table_current.csv"),
                "completed_task_count": len(registry_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
