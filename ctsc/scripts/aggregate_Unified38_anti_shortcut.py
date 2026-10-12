'Aggregate current-branch Unified38 anti-shortcut task summaries.'

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.public_source_catalog import dsrl_catalog_records


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_CLAIMS_PALETTE = [
    "not purely current-step cost driven",
    "non-cost history contributes",
    "historical cost remains important",
    "shortcut competition remains central",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default="work/anti_shortcut")
    parser.add_argument("--output-dir", type=str, default="work/anti_shortcut")
    parser.add_argument("--task-names", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _median_or_zero(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _discover_summaries(input_root: Path) -> list[dict[str, Any]]:
    return [_load_json(path) for path in sorted(input_root.rglob("boundary_summary.json"))]


def main() -> None:
    args = parse_args()
    input_root = REPO_ROOT / args.input_root
    output_dir = REPO_ROOT / args.output_dir
    split_seeds = [int(seed) for seed in args.split_seeds]
    task_names = (
        list(dict.fromkeys(str(name) for name in args.task_names))
        if args.task_names
        else [str(row["task_name"]) for row in dsrl_catalog_records()]
    )

    payloads = _discover_summaries(input_root)
    by_key = {
        (str(payload.get("task_name")), int(payload.get("split_seed", -1))): payload
        for payload in payloads
        if str(payload.get("status")) == "success"
    }

    task_seed_rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    completed_task_count = 0
    for task_name in task_names:
        completed_split_seeds: list[int] = []
        missing_split_seeds: list[int] = []
        boundary_tag_counts: dict[str, int] = {}
        source_platform = "unknown_source_platform"
        for split_seed in split_seeds:
            payload = by_key.get((task_name, int(split_seed)))
            if payload is None:
                missing_split_seeds.append(int(split_seed))
                continue
            completed_split_seeds.append(int(split_seed))
            source_platform = str(payload.get("source_platform", source_platform))
            boundary_tag = str(payload.get("boundary_tag", "pending_branch_wide_coverage"))
            boundary_tag_counts[boundary_tag] = boundary_tag_counts.get(boundary_tag, 0) + 1
            task_seed_rows.append(
                {
                    "task_name": task_name,
                    "split_seed": int(split_seed),
                    "source_platform": source_platform,
                    "boundary_tag": boundary_tag,
                    "full_step_auprc": float(payload["full_reference"]["step_catastrophic_auprc"]),
                    "current_only_step_auprc": float(payload["profiles"]["current_only"]["step_catastrophic_auprc"]),
                    "cost_step_only_step_auprc": float(payload["profiles"]["cost_step_only"]["step_catastrophic_auprc"]),
                    "history_without_current_cost_step_auprc": float(
                        payload["profiles"]["history_without_current_cost"]["step_catastrophic_auprc"]
                    ),
                    "history_obs_action_step_only_step_auprc": float(
                        payload["profiles"]["history_obs_action_step_only"]["step_catastrophic_auprc"]
                    ),
                }
            )
        representative_boundary_tag = (
            max(boundary_tag_counts.items(), key=lambda item: (int(item[1]), str(item[0])))[0]
            if boundary_tag_counts
            else "pending_branch_wide_coverage"
        )
        if completed_split_seeds == split_seeds:
            completed_task_count += 1
        tasks.append(
            {
                "task_name": task_name,
                "source_platform": source_platform,
                "completed_split_seeds": completed_split_seeds,
                "missing_split_seeds": missing_split_seeds,
                "task_seed_coverage_status": "pass" if completed_split_seeds == split_seeds else "fail",
                "boundary_tag_counts": boundary_tag_counts,
                "boundary_tag": representative_boundary_tag,
            }
        )

    pooled_boundary_tag_counts: dict[str, int] = {}
    for row in task_seed_rows:
        tag = str(row["boundary_tag"])
        pooled_boundary_tag_counts[tag] = pooled_boundary_tag_counts.get(tag, 0) + 1

    completed_task_seed_count = len(task_seed_rows)
    expected_task_seed_count = len(task_names) * len(split_seeds)
    summary = {
        "phase": "Unified38_current_boundary",
        "boundary_source_mode": "current_branch",
        "protocol_reference": "manifests/unified38_active_protocol.json",
        "split_seeds": list(split_seeds),
        "task_count": len(task_names),
        "completed_task_count": completed_task_count,
        "completed_task_seed_count": completed_task_seed_count,
        "expected_task_seed_count": expected_task_seed_count,
        "task_seed_coverage_status": "pass" if completed_task_seed_count == expected_task_seed_count else "fail",
        "safe_claims_palette": list(SAFE_CLAIMS_PALETTE),
        "pooled_all_seeds": {
            "task_seed_count": completed_task_seed_count,
            "boundary_tag_counts": pooled_boundary_tag_counts,
            "full_gt_current_only_count": int(
                sum(float(row["full_step_auprc"]) > float(row["current_only_step_auprc"]) for row in task_seed_rows)
            ),
            "history_without_current_cost_gt_cost_step_only_count": int(
                sum(
                    float(row["history_without_current_cost_step_auprc"]) > float(row["cost_step_only_step_auprc"])
                    for row in task_seed_rows
                )
            ),
            "history_obs_action_step_only_gt_current_only_count": int(
                sum(
                    float(row["history_obs_action_step_only_step_auprc"]) > float(row["current_only_step_auprc"])
                    for row in task_seed_rows
                )
            ),
            "median_full_step_auprc": _median_or_zero([float(row["full_step_auprc"]) for row in task_seed_rows]),
            "median_current_only_step_auprc": _median_or_zero([float(row["current_only_step_auprc"]) for row in task_seed_rows]),
            "median_history_without_current_cost_step_auprc": _median_or_zero(
                [float(row["history_without_current_cost_step_auprc"]) for row in task_seed_rows]
            ),
        },
        "tasks": tasks,
        "task_seed_rows": task_seed_rows,
    }

    _write_json(output_dir / "Unified38_boundary_aggregate.json", summary)
    _write_csv(output_dir / "Unified38_boundary_task_table.csv", tasks)
    _write_csv(output_dir / "Unified38_boundary_task_seed_table.csv", task_seed_rows)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "Unified38_boundary_aggregate.json"),
                "task_count": len(task_names),
                "completed_task_count": completed_task_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
