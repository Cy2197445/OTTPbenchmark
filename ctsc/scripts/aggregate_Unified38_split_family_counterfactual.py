'Aggregate the Unified38 split-family counterfactual study.'

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

from ctsc.precursor.data import SPLIT_FAMILY_REGISTRY
from ctsc.precursor.public_source_catalog import dsrl_catalog_records


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default="work/split_family/task_runs")
    parser.add_argument("--output-dir", type=str, default="work/split_family/aggregates")
    parser.add_argument("--task-names", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--slices", type=str, nargs="+", default=["canonical", "sensitivity"])
    parser.add_argument("--split-families", type=str, nargs="+", choices=list(SPLIT_FAMILY_REGISTRY), default=list(SPLIT_FAMILY_REGISTRY))
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


def _suite_diagnostic_value(suite: dict[str, Any], section: str, key: str) -> float:
    return float(suite.get("split_diagnostics", {}).get(section, {}).get(key, 0.0))


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
    split_families = [str(name) for name in args.split_families]
    source_platform_map = _source_platform_map()

    suite_payloads: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    suite_rows: list[dict[str, Any]] = []
    seed_model_rows: list[dict[str, Any]] = []
    paired_groups: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    coverage_rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for slice_name in slices:
            for split_seed in split_seeds:
                observed_families: list[str] = []
                for split_family in split_families:
                    summary_path = input_root / task_name / slice_name / f"seed_{int(split_seed)}" / split_family / "suite_summary.json"
                    if not summary_path.exists():
                        continue
                    suite = _load_json(summary_path)
                    suite_payloads[(task_name, slice_name, int(split_seed), split_family)] = suite
                    observed_families.append(split_family)
                    suite_rows.append(
                        {
                            "task_name": task_name,
                            "slice_name": slice_name,
                            "split_seed": int(split_seed),
                            "split_family": split_family,
                            "source_platform": source_platform_map[task_name],
                            "best_sequence_baseline": suite["best_sequence_baseline"],
                            "best_sequence_test_step_auprc": float(suite["best_sequence_test_step_auprc"]),
                            "gap_vs_flat": None if suite["gap_vs_flat"] is None else float(suite["gap_vs_flat"]),
                            "gap_vs_linear": None if suite["gap_vs_linear"] is None else float(suite["gap_vs_linear"]),
                            "strict_task_level_sequence_support": bool(suite["strict_task_level_sequence_support"]),
                            "group_overlap_count": int(_suite_diagnostic_value(suite, "group_overlap", "overlapping_group_count")),
                            "group_overlap_ratio": float(_suite_diagnostic_value(suite, "group_overlap", "overlapping_group_ratio")),
                            "fragmented_episode_count": int(_suite_diagnostic_value(suite, "episode_fragmentation", "fragmented_episode_count")),
                            "fragmented_episode_ratio": float(_suite_diagnostic_value(suite, "episode_fragmentation", "fragmented_episode_ratio")),
                        }
                    )
                    for model_name, metrics in suite["models"].items():
                        normalized_model = str(model_name).replace("-", "_")
                        row = {
                            "task_name": task_name,
                            "slice_name": slice_name,
                            "split_seed": int(split_seed),
                            "split_family": split_family,
                            "source_platform": source_platform_map[task_name],
                            "model_name": normalized_model,
                            "step_auprc": float(metrics["test_step_catastrophic_auprc"]),
                            "outcome_auprc": float(metrics["test_outcome_catastrophic_auprc"]),
                            "ece": float(metrics["test_catastrophic_ece"]),
                            "raw_step_lead_time": float(metrics["test_lead_time_median_at_0.5"]),
                            "group_overlap_count": int(_suite_diagnostic_value(suite, "group_overlap", "overlapping_group_count")),
                            "group_overlap_ratio": float(_suite_diagnostic_value(suite, "group_overlap", "overlapping_group_ratio")),
                            "fragmented_episode_count": int(_suite_diagnostic_value(suite, "episode_fragmentation", "fragmented_episode_count")),
                            "fragmented_episode_ratio": float(_suite_diagnostic_value(suite, "episode_fragmentation", "fragmented_episode_ratio")),
                        }
                        seed_model_rows.append(row)
                        paired_groups[(task_name, slice_name, normalized_model, split_family)].setdefault("rows", []).append(row)
                coverage_rows.append(
                    {
                        "task_name": task_name,
                        "slice_name": slice_name,
                        "split_seed": int(split_seed),
                        "observed_split_families": json.dumps(sorted(observed_families)),
                        "coverage_status": "pass" if sorted(observed_families) == sorted(split_families) else "fail",
                    }
                )

    paired_rows: list[dict[str, Any]] = []
    model_names = sorted({key[2] for key in paired_groups})
    for task_name in task_names:
        for slice_name in slices:
            for model_name in model_names:
                episode_rows = paired_groups.get((task_name, slice_name, model_name, "episode_group"), {}).get("rows", [])
                random_rows = paired_groups.get((task_name, slice_name, model_name, "transition_random"), {}).get("rows", [])
                if not episode_rows or not random_rows:
                    continue
                paired_rows.append(
                    {
                        "task_name": task_name,
                        "slice_name": slice_name,
                        "model_name": model_name,
                        "source_platform": source_platform_map[task_name],
                        "episode_group_step_auprc": _median([float(row["step_auprc"]) for row in episode_rows]),
                        "transition_random_step_auprc": _median([float(row["step_auprc"]) for row in random_rows]),
                        "step_auprc_delta": _median([float(row["step_auprc"]) for row in random_rows]) - _median([float(row["step_auprc"]) for row in episode_rows]),
                        "episode_group_outcome_auprc": _median([float(row["outcome_auprc"]) for row in episode_rows]),
                        "transition_random_outcome_auprc": _median([float(row["outcome_auprc"]) for row in random_rows]),
                        "outcome_auprc_delta": _median([float(row["outcome_auprc"]) for row in random_rows]) - _median([float(row["outcome_auprc"]) for row in episode_rows]),
                        "episode_group_ece": _median([float(row["ece"]) for row in episode_rows]),
                        "transition_random_ece": _median([float(row["ece"]) for row in random_rows]),
                        "ece_delta": _median([float(row["ece"]) for row in random_rows]) - _median([float(row["ece"]) for row in episode_rows]),
                        "episode_group_raw_step_lead_time": _median([float(row["raw_step_lead_time"]) for row in episode_rows]),
                        "transition_random_raw_step_lead_time": _median([float(row["raw_step_lead_time"]) for row in random_rows]),
                        "raw_step_lead_time_delta": _median([float(row["raw_step_lead_time"]) for row in random_rows]) - _median([float(row["raw_step_lead_time"]) for row in episode_rows]),
                        "transition_random_group_overlap_count": _median([float(row["group_overlap_count"]) for row in random_rows]),
                        "transition_random_group_overlap_ratio": _median([float(row["group_overlap_ratio"]) for row in random_rows]),
                        "transition_random_fragmented_episode_count": _median([float(row["fragmented_episode_count"]) for row in random_rows]),
                        "transition_random_fragmented_episode_ratio": _median([float(row["fragmented_episode_ratio"]) for row in random_rows]),
                    }
                )

    _write_csv(output_dir / "unified_split_family_suite_table.csv", suite_rows)
    _write_csv(output_dir / "unified_seed_model_table_split_family.csv", seed_model_rows)
    _write_csv(output_dir / "unified_split_family_task_pairs.csv", paired_rows)
    _write_csv(output_dir / "unified_split_family_coverage.csv", coverage_rows)
    _write_json(
        output_dir / "Unified38_split_family_counterfactual_summary.json",
        {
            "title": "Unified38 Split Family Counterfactual Summary",
            "task_count": len(task_names),
            "split_seed_count": len(split_seeds),
            "split_families": split_families,
            "suite_row_count": len(suite_rows),
            "seed_model_row_count": len(seed_model_rows),
            "paired_task_model_row_count": len(paired_rows),
            "coverage_pass_count": int(sum(row["coverage_status"] == "pass" for row in coverage_rows)),
            "coverage_row_count": len(coverage_rows),
        },
    )
    print(
        json.dumps(
            {
                "suite_table": str(output_dir / "unified_split_family_suite_table.csv"),
                "seed_model_table": str(output_dir / "unified_seed_model_table_split_family.csv"),
                "task_pairs": str(output_dir / "unified_split_family_task_pairs.csv"),
                "paired_task_model_row_count": len(paired_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
