'Analyze current Unified38 canonical GRU profile-matrix runs.'

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "work" / "profile_matrix"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "analysis"
CORE_COMPARISONS = {
    "full_minus_current_only_step_auprc": "full vs current_only",
    "history_without_current_cost_minus_cost_step_only_step_auprc": "history_without_current_cost vs cost_step_only",
    "history_obs_action_step_only_minus_current_only_step_auprc": "history_obs_action_step_only vs current_only",
}
EXTENDED_TAGS = [
    "order-sensitive",
    "current-cost-heavy",
    "history-visible",
    "mostly-current-like",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default=str(DEFAULT_INPUT_ROOT.relative_to(REPO_ROOT)))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)))
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
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


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)) if values else None,
        "median": float(statistics.median(values)) if values else None,
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "positive_count": int(sum(value > 0 for value in values)),
    }


def _discover_task_summaries(input_root: Path) -> list[dict[str, Any]]:
    return [_read_json(path) for path in sorted(input_root.rglob("profile_matrix_summary.json"))]


def _task_table(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in summaries:
        profiles = payload["profiles"]
        row: dict[str, Any] = {
            "task_name": str(payload["task_name"]),
            "source_platform": str(payload["source_platform"]),
            "split_seed": int(payload["split_seed"]),
        }
        for profile_name, metrics in profiles.items():
            normalized = profile_name.replace("-", "_")
            row[f"{normalized}_step_auprc"] = float(metrics["test_step_catastrophic_auprc"])
            row[f"{normalized}_outcome_auprc"] = float(metrics["test_outcome_catastrophic_auprc"])
        if "core_contrasts" in payload:
            for key, value in payload["core_contrasts"].items():
                row[key] = float(value)
        if "extended_contrasts" in payload:
            for key, value in payload["extended_contrasts"].items():
                row[key] = float(value)

        full_minus_current = float(row.get("full_minus_current_only_step_auprc", 0.0))
        full_minus_shuffled = float(row.get("full_minus_full_shuffled_step_auprc", 0.0))
        cost_minus_lagged = float(row.get("cost_step_only_minus_lagged_cost_step_step_auprc", 0.0))
        full_minus_current_wo_cost = float(row.get("full_minus_current_without_cost_step_auprc", 0.0))
        history_minus_cost = float(row.get("history_without_current_cost_minus_cost_step_only_step_auprc", 0.0))

        tag_memberships: list[str] = []
        if full_minus_shuffled >= 0.02:
            tag_memberships.append("order-sensitive")
        if full_minus_current_wo_cost >= 0.02 and cost_minus_lagged >= 0.01:
            tag_memberships.append("current-cost-heavy")
        if full_minus_current >= 0.02 or history_minus_cost >= 0.02:
            tag_memberships.append("history-visible")
        if full_minus_current <= 0.01:
            tag_memberships.append("mostly-current-like")
        row["tag_memberships"] = ";".join(tag_memberships)
        for tag in EXTENDED_TAGS:
            row[f"is_{tag.replace('-', '_')}"] = bool(tag in tag_memberships)
        rows.append(row)
    return rows


def _anti_shortcut_summary(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "title": "Unified38 Canonical Anti-Shortcut Seed0 Summary",
        "coverage": {"task_count": len(task_rows), "slice_name": "canonical", "split_seeds": [0]},
        "contrasts": {},
    }
    for key in CORE_COMPARISONS:
        values = [float(row[key]) for row in task_rows]
        summary["contrasts"][key] = _stats(values)

    full_vs_current = summary["contrasts"]["full_minus_current_only_step_auprc"]
    history_vs_cost = summary["contrasts"]["history_without_current_cost_minus_cost_step_only_step_auprc"]
    summary["decision_gate"] = {
        "full_minus_current_positive_tasks": int(full_vs_current["positive_count"]),
        "history_without_current_cost_minus_cost_step_only_positive_tasks": int(history_vs_cost["positive_count"]),
        "full_minus_current_mean": float(full_vs_current["mean"]),
        "meets_no_seed_expansion_threshold": bool(
            int(full_vs_current["positive_count"]) >= 30
            and int(history_vs_cost["positive_count"]) >= 30
            and float(full_vs_current["mean"]) >= 0.02
        ),
    }
    return summary


def _extended_summary(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "title": "Unified38 Canonical Extended Profiles Seed0 Summary",
        "coverage": {"task_count": len(task_rows), "slice_name": "canonical", "split_seeds": [0]},
        "contrasts": {
            "full_minus_full_shuffled_step_auprc": _stats(
                [float(row["full_minus_full_shuffled_step_auprc"]) for row in task_rows]
            ),
            "cost_step_only_minus_lagged_cost_step_step_auprc": _stats(
                [float(row["cost_step_only_minus_lagged_cost_step_step_auprc"]) for row in task_rows]
            ),
            "full_minus_current_without_cost_step_auprc": _stats(
                [float(row["full_minus_current_without_cost_step_auprc"]) for row in task_rows]
            ),
        },
        "tag_counts": {},
    }
    for tag in EXTENDED_TAGS:
        key = f"is_{tag.replace('-', '_')}"
        summary["tag_counts"][tag] = int(sum(bool(row[key]) for row in task_rows))
    return summary


def _group_tasks(task_rows: list[dict[str, Any]], flag_key: str) -> list[str]:
    return [str(row["task_name"]) for row in task_rows if bool(row[flag_key])]


def _write_anti_shortcut_report(path: Path, summary: dict[str, Any], task_rows: list[dict[str, Any]]) -> None:
    full_vs_current = summary["contrasts"]["full_minus_current_only_step_auprc"]
    history_vs_cost = summary["contrasts"]["history_without_current_cost_minus_cost_step_only_step_auprc"]
    hist_obs_vs_current = summary["contrasts"]["history_obs_action_step_only_minus_current_only_step_auprc"]
    weakest = sorted(task_rows, key=lambda row: float(row["full_minus_current_only_step_auprc"]))[:10]

    lines = [
        "# GRU feature-profile comparison",
        "",
        "## Coverage",
        "",
        f"- tasks: `{summary['coverage']['task_count']}`",
        f"- slice: `{summary['coverage']['slice_name']}`",
        f"- split seed: `{summary['coverage']['split_seeds'][0]}`",
        "",
        "## Results",
        "",
        f"- `full - current_only`: mean `{full_vs_current['mean']:.4f}`，median `{full_vs_current['median']:.4f}`，positive `{full_vs_current['positive_count']}/{full_vs_current['n']}`。",
        f"- `history_without_current_cost - cost_step_only`: mean `{history_vs_cost['mean']:.4f}`，median `{history_vs_cost['median']:.4f}`，positive `{history_vs_cost['positive_count']}/{history_vs_cost['n']}`。",
        f"- `history_obs_action_step_only - current_only`: mean `{hist_obs_vs_current['mean']:.4f}`，median `{hist_obs_vs_current['median']:.4f}`，positive `{hist_obs_vs_current['positive_count']}/{hist_obs_vs_current['n']}`。",
        "",
        "## Feature contrasts",
        "",
        "- This comparison uses the 38-task canonical GRU matrix with split seed 0.",
        "- Full-prefix, current-only, and cost-masked inputs quantify the contribution of history and current cost.",
        "- The main three-model results are reported separately.",
        "",
        "## Seed-expansion criterion",
        "",
        f"- `meets_no_seed_expansion_threshold = {summary['decision_gate']['meets_no_seed_expansion_threshold']}`",
        "",
        "## Tasks with the smallest history gain",
        "",
    ]
    for row in weakest:
        lines.append(
            f"- `{row['task_name']}` ({row['source_platform']}): "
            f"`full-current={float(row['full_minus_current_only_step_auprc']):.4f}`, "
            f"`history-cost={float(row['history_without_current_cost_minus_cost_step_only_step_auprc']):.4f}`, "
            f"`histObs-current={float(row['history_obs_action_step_only_minus_current_only_step_auprc']):.4f}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_extended_report(path: Path, summary: dict[str, Any], task_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Extended GRU feature profiles",
        "",
        "## Coverage",
        "",
        f"- tasks: `{summary['coverage']['task_count']}`",
        "",
        "## Contrast summary",
        "",
        f"- `full - full_shuffled`: mean `{summary['contrasts']['full_minus_full_shuffled_step_auprc']['mean']:.4f}`，positive `{summary['contrasts']['full_minus_full_shuffled_step_auprc']['positive_count']}/{summary['contrasts']['full_minus_full_shuffled_step_auprc']['n']}`。",
        f"- `cost_step_only - lagged_cost_step`: mean `{summary['contrasts']['cost_step_only_minus_lagged_cost_step_step_auprc']['mean']:.4f}`，positive `{summary['contrasts']['cost_step_only_minus_lagged_cost_step_step_auprc']['positive_count']}/{summary['contrasts']['cost_step_only_minus_lagged_cost_step_step_auprc']['n']}`。",
        f"- `full - current_without_cost`: mean `{summary['contrasts']['full_minus_current_without_cost_step_auprc']['mean']:.4f}`，positive `{summary['contrasts']['full_minus_current_without_cost_step_auprc']['positive_count']}/{summary['contrasts']['full_minus_current_without_cost_step_auprc']['n']}`。",
        "",
        "## Tag distribution",
        "",
    ]
    for tag in EXTENDED_TAGS:
        lines.append(f"- `{tag}`: `{summary['tag_counts'][tag]}`")
    lines.extend(["", "## Tasks by tag", ""])
    for tag in EXTENDED_TAGS:
        flag_key = f"is_{tag.replace('-', '_')}"
        tasks = _group_tasks(task_rows, flag_key)
        lines.append(f"- `{tag}`: {', '.join(tasks) if tasks else 'none'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = _resolve(args.input_root)
    output_dir = _resolve(args.output_dir)
    summaries = _discover_task_summaries(input_root)
    if not summaries:
        raise FileNotFoundError(f"No profile_matrix_summary.json found under: {input_root}")
    missing_core = [payload["task_name"] for payload in summaries if "core_contrasts" not in payload]
    if missing_core:
        raise ValueError(f"Core anti-shortcut analysis requires the core 5 profiles; missing for tasks={missing_core}")

    task_rows = _task_table(summaries)
    anti_summary = _anti_shortcut_summary(task_rows)
    extended_summary = _extended_summary(task_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "Unified38_canonical_anti_shortcut_seed0_summary.json", anti_summary)
    _write_csv(output_dir / "Unified38_canonical_anti_shortcut_seed0_task_table.csv", task_rows)
    _write_anti_shortcut_report(output_dir / "Unified38_canonical_anti_shortcut_seed0_report_zh.md", anti_summary, task_rows)
    _write_json(output_dir / "Unified38_canonical_extended_profiles_seed0_summary.json", extended_summary)
    _write_csv(output_dir / "Unified38_canonical_extended_profiles_seed0_task_table.csv", task_rows)
    _write_extended_report(output_dir / "Unified38_canonical_extended_profiles_seed0_report_zh.md", extended_summary, task_rows)
    print(
        json.dumps(
            {
                "anti_shortcut_summary": str(output_dir / "Unified38_canonical_anti_shortcut_seed0_summary.json"),
                "extended_profiles_summary": str(output_dir / "Unified38_canonical_extended_profiles_seed0_summary.json"),
                "task_count": len(task_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
