'Aggregate the Unified38 current-facing target-definition ablation study.'

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.ablation import ABLATION_GROUP_MODE_REGISTRY
from ctsc.precursor.public_source_catalog import dsrl_catalog_records


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_SPECS = {
    "task_framing": [
        {
            "comparison_name": "step_level_vs_broadcast",
            "reference_mode": "step_level_current_3way",
            "mode_name": "episode_outcome_step_broadcast",
        },
        {
            "comparison_name": "step_level_vs_terminal_only",
            "reference_mode": "step_level_current_3way",
            "mode_name": "episode_outcome_terminal_only",
        },
        {
            "comparison_name": "broadcast_vs_terminal_only",
            "reference_mode": "episode_outcome_step_broadcast",
            "mode_name": "episode_outcome_terminal_only",
        },
    ],
    "labels": [
        {
            "comparison_name": "step_3way_vs_unsafe_vs_safe",
            "reference_mode": "step_3way_q75_q90",
            "mode_name": "binary_unsafe_vs_safe",
        },
        {
            "comparison_name": "step_3way_vs_cat_vs_rest",
            "reference_mode": "step_3way_q75_q90",
            "mode_name": "binary_cat_vs_rest",
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default="work/target_definition/task_runs")
    parser.add_argument("--output-dir", type=str, default="work/target_definition/aggregates")
    parser.add_argument("--task-names", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
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


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


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
    source_platform_map = _source_platform_map()

    suite_rows: list[dict[str, Any]] = []
    seed_model_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    mode_model_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for task_name in task_names:
        for split_seed in split_seeds:
            for ablation_group, modes in ABLATION_GROUP_MODE_REGISTRY.items():
                observed_modes: list[str] = []
                for ablation_mode in modes:
                    summary_path = (
                        input_root
                        / task_name
                        / "canonical"
                        / f"seed_{int(split_seed)}"
                        / ablation_group
                        / ablation_mode
                        / "suite_summary.json"
                    )
                    if not summary_path.exists():
                        continue
                    suite = _load_json(summary_path)
                    observed_modes.append(ablation_mode)
                    suite_rows.append(
                        {
                            "task_name": task_name,
                            "slice_name": "canonical",
                            "split_seed": int(split_seed),
                            "ablation_group": ablation_group,
                            "ablation_mode": ablation_mode,
                            "source_platform": source_platform_map[task_name],
                            "selection_target_type": suite["selection_target_type"],
                            "primary_eval_target": suite["primary_eval_target"],
                            "secondary_eval_target": suite["secondary_eval_target"],
                            "train_mask_mode": suite["train_mask_mode"],
                            "best_sequence_baseline": suite["best_sequence_baseline"],
                            "best_sequence_test_step_auprc": float(suite["best_sequence_test_step_auprc"]),
                            "gap_vs_flat": None if suite["gap_vs_flat"] is None else float(suite["gap_vs_flat"]),
                            "gap_vs_linear": None if suite["gap_vs_linear"] is None else float(suite["gap_vs_linear"]),
                            "strict_task_level_sequence_support": bool(suite["strict_task_level_sequence_support"]),
                            "effective_train_count": int(suite["effective_split_sizes"]["train"]),
                            "effective_val_count": int(suite["effective_split_sizes"]["val"]),
                            "effective_test_count": int(suite["effective_split_sizes"]["test"]),
                        }
                    )
                    for model_name, metrics in suite["models"].items():
                        normalized_model = str(model_name).replace("-", "_")
                        row = {
                            "task_name": task_name,
                            "slice_name": "canonical",
                            "split_seed": int(split_seed),
                            "ablation_group": ablation_group,
                            "ablation_mode": ablation_mode,
                            "source_platform": source_platform_map[task_name],
                            "model_name": normalized_model,
                            "selection_target_type": str(metrics["selection_target_type"]),
                            "primary_eval_target": str(metrics["primary_eval_target"]),
                            "secondary_eval_target": str(metrics["secondary_eval_target"]),
                            "train_mask_mode": str(metrics["train_mask_mode"]),
                            "selection_target_auprc": float(metrics["test_selection_target_auprc"]),
                            "step_auprc": float(metrics["test_step_catastrophic_auprc"]),
                            "outcome_auprc": float(metrics["test_outcome_catastrophic_auprc"]),
                            "ece": float(metrics["test_catastrophic_ece"]),
                            "raw_step_lead_time": float(metrics["test_lead_time_median_at_0.5"]),
                        }
                        seed_model_rows.append(row)
                        mode_model_groups[(task_name, "canonical", ablation_group, ablation_mode, normalized_model)].append(row)
                coverage_rows.append(
                    {
                        "task_name": task_name,
                        "slice_name": "canonical",
                        "split_seed": int(split_seed),
                        "ablation_group": ablation_group,
                        "observed_modes": json.dumps(sorted(observed_modes)),
                        "coverage_status": "pass" if sorted(observed_modes) == sorted(modes) else "fail",
                    }
                )

    pair_rows_by_group: dict[str, list[dict[str, Any]]] = {"task_framing": [], "labels": []}
    for ablation_group, comparisons in COMPARISON_SPECS.items():
        model_names = sorted(
            {
                key[4]
                for key in mode_model_groups
                if key[2] == ablation_group
            }
        )
        for task_name in task_names:
            for model_name in model_names:
                for comparison in comparisons:
                    reference_rows = mode_model_groups.get(
                        (task_name, "canonical", ablation_group, comparison["reference_mode"], model_name),
                        [],
                    )
                    mode_rows = mode_model_groups.get(
                        (task_name, "canonical", ablation_group, comparison["mode_name"], model_name),
                        [],
                    )
                    if not reference_rows or not mode_rows:
                        continue
                    pair_rows_by_group[ablation_group].append(
                        {
                            "task_name": task_name,
                            "slice_name": "canonical",
                            "ablation_group": ablation_group,
                            "comparison_name": comparison["comparison_name"],
                            "reference_mode": comparison["reference_mode"],
                            "mode_name": comparison["mode_name"],
                            "model_name": model_name,
                            "source_platform": source_platform_map[task_name],
                            "reference_step_auprc": _median([float(row["step_auprc"]) for row in reference_rows]),
                            "mode_step_auprc": _median([float(row["step_auprc"]) for row in mode_rows]),
                            "step_auprc_delta": _median([float(row["step_auprc"]) for row in reference_rows]) - _median([float(row["step_auprc"]) for row in mode_rows]),
                            "reference_outcome_auprc": _median([float(row["outcome_auprc"]) for row in reference_rows]),
                            "mode_outcome_auprc": _median([float(row["outcome_auprc"]) for row in mode_rows]),
                            "outcome_auprc_delta": _median([float(row["outcome_auprc"]) for row in reference_rows]) - _median([float(row["outcome_auprc"]) for row in mode_rows]),
                            "reference_ece": _median([float(row["ece"]) for row in reference_rows]),
                            "mode_ece": _median([float(row["ece"]) for row in mode_rows]),
                            "ece_delta": _median([float(row["ece"]) for row in reference_rows]) - _median([float(row["ece"]) for row in mode_rows]),
                            "reference_raw_step_lead_time": _median([float(row["raw_step_lead_time"]) for row in reference_rows]),
                            "mode_raw_step_lead_time": _median([float(row["raw_step_lead_time"]) for row in mode_rows]),
                            "raw_step_lead_time_delta": _median([float(row["raw_step_lead_time"]) for row in reference_rows]) - _median([float(row["raw_step_lead_time"]) for row in mode_rows]),
                            "reference_selection_target_auprc": _median([float(row["selection_target_auprc"]) for row in reference_rows]),
                            "mode_selection_target_auprc": _median([float(row["selection_target_auprc"]) for row in mode_rows]),
                            "selection_target_auprc_delta": _median([float(row["selection_target_auprc"]) for row in reference_rows]) - _median([float(row["selection_target_auprc"]) for row in mode_rows]),
                        }
                    )

    _write_csv(output_dir / "unified_target_definition_suite_table.csv", suite_rows)
    _write_csv(output_dir / "unified_target_definition_seed_model_table.csv", seed_model_rows)
    _write_csv(output_dir / "task_framing_task_pairs.csv", pair_rows_by_group["task_framing"])
    _write_csv(output_dir / "label_family_task_pairs.csv", pair_rows_by_group["labels"])
    _write_csv(output_dir / "unified_target_definition_coverage.csv", coverage_rows)
    _write_json(
        output_dir / "Unified38_target_definition_ablation_summary.json",
        {
            "title": "Unified38 Target Definition Ablation Summary",
            "task_count": len(task_names),
            "split_seed_count": len(split_seeds),
            "ablation_groups": list(ABLATION_GROUP_MODE_REGISTRY),
            "suite_row_count": len(suite_rows),
            "seed_model_row_count": len(seed_model_rows),
            "task_framing_paired_task_model_row_count": len(pair_rows_by_group["task_framing"]),
            "label_family_paired_task_model_row_count": len(pair_rows_by_group["labels"]),
            "coverage_pass_count": int(sum(row["coverage_status"] == "pass" for row in coverage_rows)),
            "coverage_row_count": len(coverage_rows),
        },
    )
    print(
        json.dumps(
            {
                "suite_table": str(output_dir / "unified_target_definition_suite_table.csv"),
                "seed_model_table": str(output_dir / "unified_target_definition_seed_model_table.csv"),
                "task_framing_pairs": str(output_dir / "task_framing_task_pairs.csv"),
                "label_family_pairs": str(output_dir / "label_family_task_pairs.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
