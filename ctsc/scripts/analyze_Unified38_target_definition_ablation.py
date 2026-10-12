'Analyze the Unified38 current-facing target-definition ablation outputs.'

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
DEFAULT_AGGREGATE_DIR = REPO_ROOT / "work" / "target_definition" / "aggregates"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work" / "target_definition" / "analysis"
MODEL_ORDER = ["gru", "flat_mlp", "prevalence"]


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
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["comparison_name"], row["reference_mode"], row["mode_name"], row["model_name"])].append(row)
    rows: list[dict[str, Any]] = []
    comparison_order = sorted({(row["comparison_name"], row["reference_mode"], row["mode_name"]) for row in pair_rows})
    for comparison_name, reference_mode, mode_name in comparison_order:
        for model_name in MODEL_ORDER:
            values = grouped.get((comparison_name, reference_mode, mode_name, model_name), [])
            step_deltas = [_float(row, "step_auprc_delta") for row in values]
            outcome_deltas = [_float(row, "outcome_auprc_delta") for row in values]
            ece_deltas = [_float(row, "ece_delta") for row in values]
            lead_deltas = [_float(row, "raw_step_lead_time_delta") for row in values]
            rows.append(
                {
                    "comparison_name": comparison_name,
                    "reference_mode": reference_mode,
                    "mode_name": mode_name,
                    "model_name": model_name,
                    "task_count": len(values),
                    **{f"step_{key}": value for key, value in _merge_stats(step_deltas, rounds=rounds, seed=seed).items()},
                    **{f"outcome_{key}": value for key, value in _merge_stats(outcome_deltas, rounds=rounds, seed=seed + 101).items()},
                    **{f"ece_{key}": value for key, value in _merge_stats(ece_deltas, rounds=rounds, seed=seed + 202).items()},
                    **{f"lead_{key}": value for key, value in _merge_stats(lead_deltas, rounds=rounds, seed=seed + 303).items()},
                }
            )
    return rows


def _task_diagnostics_rows(pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "task_name": row["task_name"],
                "comparison_name": row["comparison_name"],
                "reference_mode": row["reference_mode"],
                "mode_name": row["mode_name"],
                "model_name": row["model_name"],
                "source_platform": row["source_platform"],
                "reference_step_auprc": _float(row, "reference_step_auprc"),
                "mode_step_auprc": _float(row, "mode_step_auprc"),
                "step_auprc_delta": _float(row, "step_auprc_delta"),
                "outcome_auprc_delta": _float(row, "outcome_auprc_delta"),
                "ece_delta": _float(row, "ece_delta"),
                "raw_step_lead_time_delta": _float(row, "raw_step_lead_time_delta"),
            }
            for row in pair_rows
        ],
        key=lambda row: (row["comparison_name"], row["model_name"], row["step_auprc_delta"]),
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
    group_name: str,
    model_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    top_k: int,
) -> None:
    strongest = sorted(diagnostics, key=lambda row: row["step_auprc_delta"], reverse=True)[:top_k]
    weakest = sorted(diagnostics, key=lambda row: row["step_auprc_delta"])[:top_k]
    lines = [
        f"# Unified38 {group_name.replace('_', ' ').title()} Analysis",
        "",
        "## Headline Summary",
        "",
        *_markdown_table(
            model_rows,
            [
                "comparison_name",
                "reference_mode",
                "mode_name",
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
        "## Strongest Positive Deltas",
        "",
        *_markdown_table(
            strongest,
            [
                "task_name",
                "comparison_name",
                "model_name",
                "source_platform",
                "step_auprc_delta",
            ],
            max_rows=top_k,
        ),
        "",
        "## Weakest Deltas",
        "",
        *_markdown_table(
            weakest,
            [
                "task_name",
                "comparison_name",
                "model_name",
                "source_platform",
                "step_auprc_delta",
            ],
            max_rows=top_k,
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    aggregate_dir = _resolve(args.aggregate_dir)
    output_dir = _resolve(args.output_dir)
    task_framing_pairs = _read_csv(aggregate_dir / "task_framing_task_pairs.csv")
    label_family_pairs = _read_csv(aggregate_dir / "label_family_task_pairs.csv")
    coverage_rows = _read_csv(aggregate_dir / "unified_target_definition_coverage.csv")
    aggregate_summary = json.loads((aggregate_dir / "Unified38_target_definition_ablation_summary.json").read_text(encoding="utf-8"))

    task_framing_model_rows = _model_summary_rows(
        task_framing_pairs,
        rounds=int(args.bootstrap_rounds),
        seed=int(args.bootstrap_seed),
    )
    label_family_model_rows = _model_summary_rows(
        label_family_pairs,
        rounds=int(args.bootstrap_rounds),
        seed=int(args.bootstrap_seed) + 1000,
    )
    task_framing_diagnostics = _task_diagnostics_rows(task_framing_pairs)
    label_family_diagnostics = _task_diagnostics_rows(label_family_pairs)

    summary = {
        "title": "Unified38 Target Definition Ablation Analysis",
        "coverage": {
            "task_count": aggregate_summary["task_count"],
            "coverage_pass_count": aggregate_summary["coverage_pass_count"],
            "coverage_row_count": aggregate_summary["coverage_row_count"],
            "coverage_fail_count": int(sum(row["coverage_status"] == "fail" for row in coverage_rows)),
            "task_framing_paired_task_model_row_count": aggregate_summary["task_framing_paired_task_model_row_count"],
            "label_family_paired_task_model_row_count": aggregate_summary["label_family_paired_task_model_row_count"],
        },
        "task_framing_model_rows": task_framing_model_rows,
        "label_family_model_rows": label_family_model_rows,
        "task_framing_strongest_pairs": sorted(task_framing_diagnostics, key=lambda row: row["step_auprc_delta"], reverse=True)[: int(args.top_k)],
        "task_framing_weakest_pairs": sorted(task_framing_diagnostics, key=lambda row: row["step_auprc_delta"])[: int(args.top_k)],
        "label_family_strongest_pairs": sorted(label_family_diagnostics, key=lambda row: row["step_auprc_delta"], reverse=True)[: int(args.top_k)],
        "label_family_weakest_pairs": sorted(label_family_diagnostics, key=lambda row: row["step_auprc_delta"])[: int(args.top_k)],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "Unified38_target_definition_ablation_analysis_summary.json", summary)
    _write_csv(output_dir / "task_framing_model_summary.csv", task_framing_model_rows)
    _write_csv(output_dir / "label_family_model_summary.csv", label_family_model_rows)
    _write_csv(output_dir / "task_framing_task_diagnostics.csv", task_framing_diagnostics)
    _write_csv(output_dir / "label_family_task_diagnostics.csv", label_family_diagnostics)
    _write_markdown(
        output_dir / "task_framing_report.md",
        group_name="task_framing",
        model_rows=task_framing_model_rows,
        diagnostics=task_framing_diagnostics,
        top_k=int(args.top_k),
    )
    _write_markdown(
        output_dir / "label_family_report.md",
        group_name="label_family",
        model_rows=label_family_model_rows,
        diagnostics=label_family_diagnostics,
        top_k=int(args.top_k),
    )
    combined_report = output_dir / "Unified38_target_definition_ablation_report.md"
    combined_report.write_text(
        "\n\n".join(
            [
                (output_dir / "task_framing_report.md").read_text(encoding="utf-8"),
                (output_dir / "label_family_report.md").read_text(encoding="utf-8"),
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "analysis_summary": str(output_dir / "Unified38_target_definition_ablation_analysis_summary.json"),
                "task_framing_model_summary": str(output_dir / "task_framing_model_summary.csv"),
                "label_family_model_summary": str(output_dir / "label_family_model_summary.csv"),
                "analysis_report": str(combined_report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
