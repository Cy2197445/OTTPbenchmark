'Diagnose weak tasks in the current-branch Unified38 fresh-rerun layer.'

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "work" / "reference_diagnostics" / "analysis"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work" / "reference_diagnostics" / "weak_task_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-diagnostics",
        type=str,
        default=str((DEFAULT_ANALYSIS_DIR / "Unified38_current_rerun_task_diagnostics.csv").relative_to(REPO_ROOT)),
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=14)
    parser.add_argument("--sensitivity-shrink-threshold", type=float, default=0.01)
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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _quantile(values: list[float], q: float) -> float:
    'Return an inclusive linear quantile with no NumPy dependency.'
    if not values:
        raise ValueError("cannot compute quantile for an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)) if values else None,
        "median": float(statistics.median(values)) if values else None,
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
    }


def _taxonomy(record: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    categories: list[str] = []
    task_name = str(record["task_name"])
    platform = str(record["source_platform"])
    if record["bottom_q_canonical_gap"] and record["bottom_q_sensitivity_gap"]:
        categories.append("near_tie_with_flat")
    if record["weaker_sequence_below_flat_both"]:
        categories.append("sequence_family_asymmetry")
    if record["sensitivity_shrink_ge_threshold"]:
        categories.append("sensitivity_shrinkage")
    if platform == "safe_metadrive_suite" and (record["bottom_q_canonical_gap"] or record["strict_fail_both"]):
        categories.append("metadrive_flat_ceiling")
    if "Button" in task_name and (
        float(record["canonical_best_seq_step_auprc"]) <= thresholds["canonical_best_seq_q1"]
        or float(record["sensitivity_best_seq_step_auprc"]) <= thresholds["sensitivity_best_seq_q1"]
    ):
        categories.append("low_absolute_button_signal")
    if platform == "safety_gymnasium_velocity_suite" and (
        record["strict_fail_both"] or record["weaker_sequence_below_flat_both"]
    ):
        categories.append("velocity_sequence_asymmetry")
    if record["bottom_q_canonical_gap"] != record["bottom_q_sensitivity_gap"]:
        categories.append("slice_specific_margin")
    if not categories and record["watchlist"]:
        categories.append("support_margin_watchlist")
    return categories


def _diagnose(rows: list[dict[str, str]], *, sensitivity_shrink_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_gaps = [_float(row, "canonical_gap_vs_flat") for row in rows]
    sensitivity_gaps = [_float(row, "sensitivity_gap_vs_flat") for row in rows]
    canonical_best = [_float(row, "canonical_best_seq_step_auprc") for row in rows]
    sensitivity_best = [_float(row, "sensitivity_best_seq_step_auprc") for row in rows]
    thresholds = {
        "canonical_gap_q1": _quantile(canonical_gaps, 0.25),
        "sensitivity_gap_q1": _quantile(sensitivity_gaps, 0.25),
        "canonical_best_seq_q1": _quantile(canonical_best, 0.25),
        "sensitivity_best_seq_q1": _quantile(sensitivity_best, 0.25),
        "sensitivity_shrink_threshold": sensitivity_shrink_threshold,
    }

    diagnosed: list[dict[str, Any]] = []
    for row in rows:
        canonical_gap = _float(row, "canonical_gap_vs_flat")
        sensitivity_gap = _float(row, "sensitivity_gap_vs_flat")
        gap_delta = _float(row, "gap_delta_sensitivity_minus_canonical")
        canonical_strict = _bool(row["canonical_strict_support"])
        sensitivity_strict = _bool(row["sensitivity_strict_support"])
        canonical_weaker_minus_flat = _float(row, "canonical_weaker_sequence_minus_flat")
        sensitivity_weaker_minus_flat = _float(row, "sensitivity_weaker_sequence_minus_flat")
        record: dict[str, Any] = {
            "task_name": row["task_name"],
            "source_platform": row["source_platform"],
            "canonical_gap_vs_flat": canonical_gap,
            "sensitivity_gap_vs_flat": sensitivity_gap,
            "gap_delta_sensitivity_minus_canonical": gap_delta,
            "canonical_gap_vs_linear": _float(row, "canonical_gap_vs_linear"),
            "sensitivity_gap_vs_linear": _float(row, "sensitivity_gap_vs_linear"),
            "canonical_best_sequence_baseline": row["canonical_best_sequence_baseline"],
            "sensitivity_best_sequence_baseline": row["sensitivity_best_sequence_baseline"],
            "canonical_strict_support": canonical_strict,
            "sensitivity_strict_support": sensitivity_strict,
            "canonical_best_seq_step_auprc": _float(row, "canonical_best_seq_step_auprc"),
            "sensitivity_best_seq_step_auprc": _float(row, "sensitivity_best_seq_step_auprc"),
            "canonical_flat_mlp_step_auprc": _float(row, "canonical_flat_mlp_step_auprc"),
            "sensitivity_flat_mlp_step_auprc": _float(row, "sensitivity_flat_mlp_step_auprc"),
            "canonical_prevalence_step_auprc": _float(row, "canonical_prevalence_step_auprc"),
            "sensitivity_prevalence_step_auprc": _float(row, "sensitivity_prevalence_step_auprc"),
            "canonical_weaker_sequence_minus_flat": canonical_weaker_minus_flat,
            "sensitivity_weaker_sequence_minus_flat": sensitivity_weaker_minus_flat,
        }
        flags = {
            "bottom_q_canonical_gap": canonical_gap <= thresholds["canonical_gap_q1"],
            "bottom_q_sensitivity_gap": sensitivity_gap <= thresholds["sensitivity_gap_q1"],
            "strict_fail_both": not canonical_strict and not sensitivity_strict,
            "sensitivity_shrink_ge_threshold": gap_delta <= -sensitivity_shrink_threshold,
            "weaker_sequence_below_flat_both": canonical_weaker_minus_flat < 0 and sensitivity_weaker_minus_flat < 0,
        }
        record.update(flags)
        record["flag_count"] = int(sum(flags.values()))
        record["watchlist"] = record["flag_count"] > 0
        record["core_weak_task"] = bool(record["flag_count"] >= 2 or record["strict_fail_both"])
        categories = _taxonomy(record, thresholds)
        record["taxonomy"] = ";".join(categories)
        diagnosed.append(record)

    diagnosed.sort(
        key=lambda record: (
            -int(record["core_weak_task"]),
            -int(record["flag_count"]),
            float(record["canonical_gap_vs_flat"]),
            float(record["sensitivity_gap_vs_flat"]),
            str(record["task_name"]),
        )
    )
    return diagnosed, thresholds


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int | None = None) -> list[str]:
    selected = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], *, top_k: int) -> None:
    core_rows = [row for row in rows if row["core_weak_task"]]
    watch_rows = [row for row in rows if row["watchlist"]]
    strongest_low_absolute = sorted(rows, key=lambda row: row["canonical_best_seq_step_auprc"])[:6]
    shrink_rows = sorted(
        [row for row in rows if row["gap_delta_sensitivity_minus_canonical"] < 0],
        key=lambda row: row["gap_delta_sensitivity_minus_canonical"],
    )[:8]
    lines = [
        "# Unified38 Current Weak-Task Diagnosis",
        "",
        "## Rule",
        "",
        "- Input layer: current fresh rerun task diagnostics, `38/38` tasks.",
        f"- Bottom-quartile canonical gap cutoff: `{summary['thresholds']['canonical_gap_q1']:.4f}`.",
        f"- Bottom-quartile sensitivity gap cutoff: `{summary['thresholds']['sensitivity_gap_q1']:.4f}`.",
        f"- Sensitivity shrinkage cutoff: `-{summary['thresholds']['sensitivity_shrink_threshold']:.4f}` gap delta.",
        "- A core weak task is flagged when it has at least two weak-task flags, or when it fails strict sequence-family support in both canonical and sensitivity slices.",
        "",
        "## Summary",
        "",
        f"- Watchlist tasks: `{summary['watchlist_task_count']}/38`.",
        f"- Core weak tasks: `{summary['core_weak_task_count']}/38`.",
        f"- Tasks failing strict support in both slices: `{summary['strict_fail_both_count']}/38`.",
        f"- Tasks with sensitivity shrinkage at or above threshold: `{summary['sensitivity_shrink_count']}/38`.",
        f"- Bottom-quartile overlap across canonical and sensitivity: `{summary['bottom_quartile_overlap_count']}/38`.",
        "",
        "## Core Weak Tasks",
        "",
        *_markdown_table(
            core_rows,
            [
                "task_name",
                "source_platform",
                "flag_count",
                "canonical_gap_vs_flat",
                "sensitivity_gap_vs_flat",
                "canonical_strict_support",
                "sensitivity_strict_support",
                "taxonomy",
            ],
        ),
        "",
        "## Lowest Absolute Best-Sequence AUPRC",
        "",
        *_markdown_table(
            strongest_low_absolute,
            [
                "task_name",
                "source_platform",
                "canonical_best_seq_step_auprc",
                "sensitivity_best_seq_step_auprc",
                "canonical_gap_vs_flat",
                "sensitivity_gap_vs_flat",
            ],
        ),
        "",
        "## Largest Sensitivity Shrinkage",
        "",
        *_markdown_table(
            shrink_rows,
            [
                "task_name",
                "source_platform",
                "gap_delta_sensitivity_minus_canonical",
                "canonical_gap_vs_flat",
                "sensitivity_gap_vs_flat",
            ],
        ),
        "",
        "## Taxonomy Counts",
        "",
        *_markdown_table(
            [
                {"taxonomy": key, "task_count": value}
                for key, value in sorted(summary["taxonomy_counts"].items(), key=lambda item: (-item[1], item[0]))
            ],
            ["taxonomy", "task_count"],
        ),
        "",
        "## Interpretation",
        "",
        "- The weak-task layer should be read as a margin and strict-support diagnosis, not as a training-failure list: best-sequence gap remains positive for all 38 tasks in the current fresh rerun.",
        "- The main current pattern is driven by sequence-family asymmetry cases where the best sequence model beats FlatMLP but the weaker sequence model does not.",
        "- The weakest current pattern shifts toward safe_metadrive near-ties and sequence-family asymmetry, while Button tasks remain important low-absolute-signal cases.",
        "- This report uses current mainline results only; full current 38/38 anti-shortcut profile results are still a separate layer.",
        "",
        "## Full Watchlist",
        "",
        *_markdown_table(
            watch_rows,
            [
                "task_name",
                "source_platform",
                "flag_count",
                "canonical_gap_vs_flat",
                "sensitivity_gap_vs_flat",
                "taxonomy",
            ],
        ),
        "",
    ]
    _write_text(path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    task_diagnostics_path = _resolve(args.task_diagnostics)
    output_dir = _resolve(args.output_dir)
    rows = _read_csv(task_diagnostics_path)
    diagnosed_rows, thresholds = _diagnose(rows, sensitivity_shrink_threshold=float(args.sensitivity_shrink_threshold))

    watch_rows = [row for row in diagnosed_rows if row["watchlist"]]
    core_rows = [row for row in diagnosed_rows if row["core_weak_task"]]
    taxonomy_counter: Counter[str] = Counter()
    platform_core_counter: Counter[str] = Counter()
    for row in diagnosed_rows:
        if row["core_weak_task"]:
            platform_core_counter[str(row["source_platform"])] += 1
        for taxonomy in str(row["taxonomy"]).split(";"):
            if taxonomy:
                taxonomy_counter[taxonomy] += 1

    try:
        source_task_diagnostics = str(task_diagnostics_path.relative_to(REPO_ROOT))
    except ValueError:
        source_task_diagnostics = str(task_diagnostics_path)

    summary = {
        "title": "Unified38 Current Weak-Task Diagnosis",
        "source_task_diagnostics": source_task_diagnostics,
        "task_count": len(diagnosed_rows),
        "thresholds": thresholds,
        "watchlist_task_count": len(watch_rows),
        "core_weak_task_count": len(core_rows),
        "strict_fail_both_count": int(sum(row["strict_fail_both"] for row in diagnosed_rows)),
        "sensitivity_shrink_count": int(sum(row["sensitivity_shrink_ge_threshold"] for row in diagnosed_rows)),
        "bottom_quartile_overlap_count": int(
            sum(row["bottom_q_canonical_gap"] and row["bottom_q_sensitivity_gap"] for row in diagnosed_rows)
        ),
        "taxonomy_counts": dict(taxonomy_counter),
        "core_weak_by_platform": dict(platform_core_counter),
        "core_weak_tasks": [
            {
                "task_name": row["task_name"],
                "source_platform": row["source_platform"],
                "flag_count": row["flag_count"],
                "canonical_gap_vs_flat": row["canonical_gap_vs_flat"],
                "sensitivity_gap_vs_flat": row["sensitivity_gap_vs_flat"],
                "taxonomy": row["taxonomy"],
            }
            for row in core_rows
        ],
        "watchlist_tasks": [
            {
                "task_name": row["task_name"],
                "source_platform": row["source_platform"],
                "flag_count": row["flag_count"],
                "canonical_gap_vs_flat": row["canonical_gap_vs_flat"],
                "sensitivity_gap_vs_flat": row["sensitivity_gap_vs_flat"],
                "taxonomy": row["taxonomy"],
            }
            for row in watch_rows
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "Unified38_current_weak_task_diagnosis_table.csv", diagnosed_rows)
    _write_json(output_dir / "Unified38_current_weak_task_diagnosis_summary.json", summary)
    _write_report(output_dir / "Unified38_current_weak_task_diagnosis_report.md", summary, diagnosed_rows, top_k=int(args.top_k))
    print(
        json.dumps(
            {
                "weak_task_summary": str(output_dir / "Unified38_current_weak_task_diagnosis_summary.json"),
                "weak_task_report": str(output_dir / "Unified38_current_weak_task_diagnosis_report.md"),
                "task_count": len(diagnosed_rows),
                "core_weak_task_count": len(core_rows),
                "watchlist_task_count": len(watch_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
