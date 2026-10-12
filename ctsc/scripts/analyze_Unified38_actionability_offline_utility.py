'Analyze the Unified38 offline actionability / downstream-utility outputs.'

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

from ctsc.precursor.actionability import (
    ACTIONABILITY_ANCHOR_BUDGET,
    ACTIONABILITY_HEADLINE_BUDGETS,
    ACTIONABILITY_HEADLINE_MODELS,
    ACTIONABILITY_MODEL_ORDER,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE_DIR = REPO_ROOT / "work" / "reference_utility" / "aggregates"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work" / "reference_utility" / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=str, default=str(DEFAULT_AGGREGATE_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _model_budget_rows(task_pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in task_pairs:
        grouped[(row["model_name"], float(row["budget"]))].append(row)

    rows: list[dict[str, Any]] = []
    budgets = sorted({float(row["budget"]) for row in task_pairs})
    for budget in budgets:
        for model_name in ACTIONABILITY_MODEL_ORDER:
            values = grouped.get((model_name, float(budget)), [])
            if not values:
                continue
            recall = [_float(row, "test_catastrophic_episode_recall_median") for row in values]
            precision = [_float(row, "test_episode_alarm_precision_median") for row in values]
            false_alarm = [_float(row, "test_noncat_episode_false_alarm_rate_median") for row in values]
            lead_time = [_float(row, "test_recalled_catastrophic_lead_time_median") for row in values]
            step_alarm = [_float(row, "test_step_alarm_rate_median") for row in values]
            rows.append(
                {
                    "budget": float(budget),
                    "model_name": model_name,
                    "task_count": len(values),
                    "mean_catastrophic_episode_recall": _stats(recall)["mean"],
                    "median_catastrophic_episode_recall": _stats(recall)["median"],
                    "mean_episode_alarm_precision": _stats(precision)["mean"],
                    "median_episode_alarm_precision": _stats(precision)["median"],
                    "mean_noncat_episode_false_alarm_rate": _stats(false_alarm)["mean"],
                    "median_noncat_episode_false_alarm_rate": _stats(false_alarm)["median"],
                    "mean_recalled_catastrophic_lead_time_median": _stats(lead_time)["mean"],
                    "median_recalled_catastrophic_lead_time_median": _stats(lead_time)["median"],
                    "mean_test_step_alarm_rate": _stats(step_alarm)["mean"],
                    "median_test_step_alarm_rate": _stats(step_alarm)["median"],
                }
            )
    return rows


def _anchor_budget_diagnostics(task_pairs: list[dict[str, str]], *, anchor_budget: float) -> list[dict[str, Any]]:
    rows = [
        {
            "task_name": row["task_name"],
            "model_name": row["model_name"],
            "source_platform": row["source_platform"],
            "budget": float(row["budget"]),
            "test_catastrophic_episode_recall": _float(row, "test_catastrophic_episode_recall_median"),
            "test_episode_alarm_precision": _float(row, "test_episode_alarm_precision_median"),
            "test_noncat_episode_false_alarm_rate": _float(row, "test_noncat_episode_false_alarm_rate_median"),
            "test_recalled_catastrophic_lead_time_median": _float(row, "test_recalled_catastrophic_lead_time_median"),
            "test_step_alarm_rate": _float(row, "test_step_alarm_rate_median"),
            "test_episode_alarm_rate": _float(row, "test_episode_alarm_rate_median"),
        }
        for row in task_pairs
        if abs(float(row["budget"]) - float(anchor_budget)) < 1e-12
    ]
    return sorted(rows, key=lambda row: (row["model_name"], -row["test_catastrophic_episode_recall"], row["task_name"]))


def _anchor_platform_summary(task_pairs: list[dict[str, str]], *, anchor_budget: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in task_pairs:
        if abs(float(row["budget"]) - float(anchor_budget)) >= 1e-12:
            continue
        grouped[(row["model_name"], row["source_platform"])].append(row)
    rows: list[dict[str, Any]] = []
    for model_name in ACTIONABILITY_MODEL_ORDER:
        platforms = sorted({platform for (candidate_model, platform) in grouped if candidate_model == model_name})
        for platform in platforms:
            values = grouped[(model_name, platform)]
            rows.append(
                {
                    "budget": float(anchor_budget),
                    "model_name": model_name,
                    "source_platform": platform,
                    "task_count": len(values),
                    "mean_catastrophic_episode_recall": _stats(
                        [_float(row, "test_catastrophic_episode_recall_median") for row in values]
                    )["mean"],
                    "mean_episode_alarm_precision": _stats(
                        [_float(row, "test_episode_alarm_precision_median") for row in values]
                    )["mean"],
                    "mean_noncat_episode_false_alarm_rate": _stats(
                        [_float(row, "test_noncat_episode_false_alarm_rate_median") for row in values]
                    )["mean"],
                    "mean_recalled_catastrophic_lead_time_median": _stats(
                        [_float(row, "test_recalled_catastrophic_lead_time_median") for row in values]
                    )["mean"],
                }
            )
    return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int | None = None) -> list[str]:
    selected = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        values: list[str] = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    args = parse_args()
    aggregate_dir = _resolve(args.aggregate_dir)
    output_dir = _resolve(args.output_dir)
    task_pairs = _read_csv(aggregate_dir / "unified_actionability_task_pairs.csv")
    coverage_rows = _read_csv(aggregate_dir / "unified_actionability_coverage.csv")
    aggregate_summary = json.loads(
        (aggregate_dir / "Unified38_actionability_offline_utility_summary.json").read_text(encoding="utf-8")
    )

    budget_model_rows = _model_budget_rows(task_pairs)
    anchor_rows = _anchor_budget_diagnostics(task_pairs, anchor_budget=float(ACTIONABILITY_ANCHOR_BUDGET))
    anchor_platform_rows = _anchor_platform_summary(task_pairs, anchor_budget=float(ACTIONABILITY_ANCHOR_BUDGET))

    strongest_anchor = sorted(
        [row for row in anchor_rows if row["model_name"] in ACTIONABILITY_HEADLINE_MODELS],
        key=lambda row: row["test_catastrophic_episode_recall"],
        reverse=True,
    )[: int(args.top_k)]
    weakest_anchor = sorted(
        [row for row in anchor_rows if row["model_name"] in ACTIONABILITY_HEADLINE_MODELS],
        key=lambda row: row["test_catastrophic_episode_recall"],
    )[: int(args.top_k)]

    summary = {
        "title": "Unified38 Offline Actionability / Downstream Utility Analysis",
        "coverage": {
            "task_count": aggregate_summary["task_count"],
            "model_count": aggregate_summary["model_count"],
            "budget_count": aggregate_summary["budget_count"],
            "coverage_row_count": aggregate_summary["coverage_row_count"],
            "coverage_pass_count": aggregate_summary["coverage_pass_count"],
            "coverage_fail_count": aggregate_summary["coverage_fail_count"],
        },
        "headline_budgets": list(ACTIONABILITY_HEADLINE_BUDGETS),
        "anchor_budget": float(ACTIONABILITY_ANCHOR_BUDGET),
        "budget_model_rows": budget_model_rows,
        "strongest_anchor_rows": strongest_anchor,
        "weakest_anchor_rows": weakest_anchor,
        "anchor_platform_rows": anchor_platform_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "Unified38_actionability_offline_utility_analysis_summary.json", summary)
    _write_csv(output_dir / "actionability_budget_model_summary.csv", budget_model_rows)
    _write_csv(output_dir / "actionability_task_diagnostics_budget5.csv", anchor_rows)
    _write_csv(output_dir / "actionability_platform_summary_budget5.csv", anchor_platform_rows)

    report_lines = [
        "# Unified38 Offline Actionability / Downstream Utility",
        "",
        "## Coverage",
        "",
        f"- tasks: `{summary['coverage']['task_count']}`",
        f"- coverage rows: `{summary['coverage']['coverage_row_count']}`",
        f"- coverage failures: `{summary['coverage']['coverage_fail_count']}`",
        "",
        "## Headline Budget Summary",
        "",
    ]
    for budget in ACTIONABILITY_HEADLINE_BUDGETS:
        report_lines.extend(
            [
                f"### Budget {budget:.1%}",
                "",
                *_markdown_table(
                    [
                        row
                        for row in budget_model_rows
                        if abs(float(row["budget"]) - float(budget)) < 1e-12
                        and row["model_name"] in ACTIONABILITY_HEADLINE_MODELS
                    ],
                    [
                        "model_name",
                        "task_count",
                        "mean_catastrophic_episode_recall",
                        "median_catastrophic_episode_recall",
                        "mean_episode_alarm_precision",
                        "mean_noncat_episode_false_alarm_rate",
                        "mean_recalled_catastrophic_lead_time_median",
                    ],
                ),
                "",
            ]
        )

    report_lines.extend(
        [
            "## Anchor-Budget Platform Summary",
            "",
            f"Anchor budget: `{ACTIONABILITY_ANCHOR_BUDGET:.1%}`",
            "",
            *_markdown_table(
                [
                    row
                    for row in anchor_platform_rows
                    if row["model_name"] in ACTIONABILITY_HEADLINE_MODELS
                ],
                [
                    "model_name",
                    "source_platform",
                    "task_count",
                    "mean_catastrophic_episode_recall",
                    "mean_episode_alarm_precision",
                    "mean_noncat_episode_false_alarm_rate",
                    "mean_recalled_catastrophic_lead_time_median",
                ],
            ),
            "",
            "## Strongest Anchor-Budget Task Rows",
            "",
            *_markdown_table(
                strongest_anchor,
                [
                    "task_name",
                    "model_name",
                    "source_platform",
                    "test_catastrophic_episode_recall",
                    "test_episode_alarm_precision",
                    "test_noncat_episode_false_alarm_rate",
                    "test_recalled_catastrophic_lead_time_median",
                ],
                max_rows=int(args.top_k),
            ),
            "",
            "## Weakest Anchor-Budget Task Rows",
            "",
            *_markdown_table(
                weakest_anchor,
                [
                    "task_name",
                    "model_name",
                    "source_platform",
                    "test_catastrophic_episode_recall",
                    "test_episode_alarm_precision",
                    "test_noncat_episode_false_alarm_rate",
                    "test_recalled_catastrophic_lead_time_median",
                ],
                max_rows=int(args.top_k),
            ),
            "",
            "## Interpretation",
            "",
            "- This study is budget-matched: thresholds are selected on validation to target a fixed step alarm rate and then applied unchanged to test.",
            "- The primary read is episode-level utility, not step-level AUPRC ranking.",
            "- Catastrophic episode recall, precision, false-alarm burden, and recalled-catastrophic lead time should be interpreted jointly.",
            "",
        ]
    )
    (output_dir / "Unified38_actionability_offline_utility_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "analysis_summary": str(output_dir / "Unified38_actionability_offline_utility_analysis_summary.json"),
                "report": str(output_dir / "Unified38_actionability_offline_utility_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
