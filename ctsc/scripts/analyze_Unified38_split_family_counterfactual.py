'Analyze the Unified38 split-family counterfactual outputs.'

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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE_DIR = REPO_ROOT / "work" / "split_family" / "aggregates"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work" / "split_family" / "analysis"
DEFAULT_MODELS = ["gru", "temporal_cnn", "flat_mlp", "linear", "prevalence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=str, default=str(DEFAULT_AGGREGATE_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
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
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "positive_count": 0,
            "ci_low": None,
            "ci_high": None,
        }
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "positive_count": int(sum(value > 0 for value in values)),
    }


def _bootstrap_ci(values: list[float], *, rounds: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(rounds), dtype=np.float64)
    for index in range(int(rounds)):
        draw = rng.choice(array, size=array.size, replace=True)
        samples[index] = float(draw.mean())
    return {
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def _merge_stats(values: list[float], *, rounds: int, seed: int) -> dict[str, Any]:
    payload = _stats(values)
    if values:
        payload.update(_bootstrap_ci(values, rounds=rounds, seed=seed))
    return payload


def _model_summary_rows(pair_rows: list[dict[str, str]], *, rounds: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["slice_name"], row["model_name"])].append(row)
    rows: list[dict[str, Any]] = []
    for slice_name in ("canonical", "sensitivity"):
        for model_name in DEFAULT_MODELS:
            values = grouped.get((slice_name, model_name), [])
            step_deltas = [_float(row, "step_auprc_delta") for row in values]
            outcome_deltas = [_float(row, "outcome_auprc_delta") for row in values]
            ece_deltas = [_float(row, "ece_delta") for row in values]
            lead_deltas = [_float(row, "raw_step_lead_time_delta") for row in values]
            rows.append(
                {
                    "slice_name": slice_name,
                    "model_name": model_name,
                    "task_count": len(values),
                    **{f"step_{key}": value for key, value in _merge_stats(step_deltas, rounds=rounds, seed=seed).items()},
                    **{f"outcome_{key}": value for key, value in _merge_stats(outcome_deltas, rounds=rounds, seed=seed + 101).items()},
                    **{f"ece_{key}": value for key, value in _merge_stats(ece_deltas, rounds=rounds, seed=seed + 202).items()},
                    **{f"lead_{key}": value for key, value in _merge_stats(lead_deltas, rounds=rounds, seed=seed + 303).items()},
                    "transition_random_group_overlap_ratio_median": (
                        float(statistics.median(_float(row, "transition_random_group_overlap_ratio") for row in values))
                        if values
                        else None
                    ),
                    "transition_random_fragmented_episode_ratio_median": (
                        float(statistics.median(_float(row, "transition_random_fragmented_episode_ratio") for row in values))
                        if values
                        else None
                    ),
                }
            )
    return rows


def _platform_summary_rows(pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["slice_name"], row["model_name"], row["source_platform"])].append(row)
    rows: list[dict[str, Any]] = []
    for (slice_name, model_name, source_platform), values in sorted(grouped.items()):
        step_deltas = [_float(row, "step_auprc_delta") for row in values]
        rows.append(
            {
                "slice_name": slice_name,
                "model_name": model_name,
                "source_platform": source_platform,
                "task_count": len(values),
                "step_delta_mean": float(sum(step_deltas) / len(step_deltas)),
                "step_delta_median": float(statistics.median(step_deltas)),
                "step_delta_positive_count": int(sum(value > 0 for value in step_deltas)),
            }
        )
    return rows


def _task_diagnostics_rows(pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "task_name": row["task_name"],
                "slice_name": row["slice_name"],
                "model_name": row["model_name"],
                "source_platform": row["source_platform"],
                "episode_group_step_auprc": _float(row, "episode_group_step_auprc"),
                "transition_random_step_auprc": _float(row, "transition_random_step_auprc"),
                "step_auprc_delta": _float(row, "step_auprc_delta"),
                "outcome_auprc_delta": _float(row, "outcome_auprc_delta"),
                "ece_delta": _float(row, "ece_delta"),
                "raw_step_lead_time_delta": _float(row, "raw_step_lead_time_delta"),
                "transition_random_group_overlap_ratio": _float(row, "transition_random_group_overlap_ratio"),
                "transition_random_fragmented_episode_ratio": _float(row, "transition_random_fragmented_episode_ratio"),
            }
            for row in pair_rows
        ],
        key=lambda row: (row["slice_name"], row["model_name"], row["step_auprc_delta"]),
    )


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


def _write_markdown(
    path: Path,
    *,
    model_rows: list[dict[str, Any]],
    platform_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    top_k: int,
) -> None:
    strongest = sorted(diagnostics, key=lambda row: row["step_auprc_delta"], reverse=True)[:top_k]
    weakest = sorted(diagnostics, key=lambda row: row["step_auprc_delta"])[:top_k]
    headline_rows = [
        row
        for row in model_rows
        if row["slice_name"] == "canonical"
    ]
    lines = [
        "# Unified38 Split-Family Counterfactual Analysis",
        "",
        "## Headline Inflation Summary",
        "",
        *_markdown_table(
            headline_rows,
            [
                "slice_name",
                "model_name",
                "task_count",
                "step_mean",
                "step_median",
                "step_positive_count",
                "step_ci_low",
                "step_ci_high",
            ],
        ),
        "",
        "## Platform Stratification",
        "",
        *_markdown_table(
            platform_rows,
            [
                "slice_name",
                "model_name",
                "source_platform",
                "task_count",
                "step_delta_mean",
                "step_delta_median",
                "step_delta_positive_count",
            ],
            max_rows=24,
        ),
        "",
        "## Strongest Inflation Task-Model Pairs",
        "",
        *_markdown_table(
            strongest,
            [
                "task_name",
                "slice_name",
                "model_name",
                "source_platform",
                "step_auprc_delta",
                "transition_random_group_overlap_ratio",
                "transition_random_fragmented_episode_ratio",
            ],
        ),
        "",
        "## Weakest Inflation Task-Model Pairs",
        "",
        *_markdown_table(
            weakest,
            [
                "task_name",
                "slice_name",
                "model_name",
                "source_platform",
                "step_auprc_delta",
                "transition_random_group_overlap_ratio",
                "transition_random_fragmented_episode_ratio",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- This study is a matched split-family counterfactual: split family changes, all remaining protocol choices stay fixed.",
        "- Positive step-delta means transition-random split inflates step catastrophic AUPRC relative to episode-group split.",
        "- Random-split leakage diagnostics are reported alongside inflation to relate larger gains align with heavier fragmentation.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    aggregate_dir = _resolve(args.aggregate_dir)
    output_dir = _resolve(args.output_dir)
    pair_rows = _read_csv(aggregate_dir / "unified_split_family_task_pairs.csv")
    coverage_rows = _read_csv(aggregate_dir / "unified_split_family_coverage.csv")
    aggregate_summary = json.loads((aggregate_dir / "Unified38_split_family_counterfactual_summary.json").read_text(encoding="utf-8"))

    model_rows = _model_summary_rows(pair_rows, rounds=int(args.bootstrap_rounds), seed=int(args.bootstrap_seed))
    platform_rows = _platform_summary_rows(pair_rows)
    diagnostics = _task_diagnostics_rows(pair_rows)

    summary = {
        "title": "Unified38 Split Family Counterfactual Analysis",
        "coverage": {
            "task_count": aggregate_summary["task_count"],
            "paired_task_model_row_count": aggregate_summary["paired_task_model_row_count"],
            "coverage_pass_count": aggregate_summary["coverage_pass_count"],
            "coverage_row_count": aggregate_summary["coverage_row_count"],
            "coverage_fail_count": int(sum(row["coverage_status"] == "fail" for row in coverage_rows)),
        },
        "model_rows": model_rows,
        "platform_rows": platform_rows,
        "strongest_inflation_pairs": sorted(diagnostics, key=lambda row: row["step_auprc_delta"], reverse=True)[: int(args.top_k)],
        "weakest_inflation_pairs": sorted(diagnostics, key=lambda row: row["step_auprc_delta"])[: int(args.top_k)],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "Unified38_split_family_counterfactual_analysis_summary.json", summary)
    _write_csv(output_dir / "Unified38_split_family_model_summary.csv", model_rows)
    _write_csv(output_dir / "Unified38_split_family_platform_summary.csv", platform_rows)
    _write_csv(output_dir / "Unified38_split_family_task_diagnostics.csv", diagnostics)
    _write_markdown(
        output_dir / "Unified38_split_family_counterfactual_report.md",
        model_rows=model_rows,
        platform_rows=platform_rows,
        diagnostics=diagnostics,
        top_k=int(args.top_k),
    )
    print(
        json.dumps(
            {
                "analysis_summary": str(output_dir / "Unified38_split_family_counterfactual_analysis_summary.json"),
                "analysis_report": str(output_dir / "Unified38_split_family_counterfactual_report.md"),
                "paired_task_model_row_count": aggregate_summary["paired_task_model_row_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
