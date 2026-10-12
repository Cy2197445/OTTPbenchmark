'Compare Unified38 current-only matrix results against the full-prefix rerun.'

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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT_AGGREGATE_DIR = REPO_ROOT / "work" / "current_only_diagnostics" / "aggregates"
DEFAULT_FULL_AGGREGATE_DIR = REPO_ROOT / "work" / "reference_diagnostics" / "aggregates"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work" / "current_only_diagnostics" / "analysis"
MODELS = ["gru", "temporal_cnn", "flat_mlp", "linear"]
SEQUENCE_MODELS = ["gru", "temporal_cnn"]
NONSEQUENTIAL_MODELS = ["flat_mlp", "linear"]
SLICES = ["canonical", "sensitivity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-aggregate-dir",
        type=str,
        default=str(DEFAULT_CURRENT_AGGREGATE_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--full-aggregate-dir",
        type=str,
        default=str(DEFAULT_FULL_AGGREGATE_DIR.relative_to(REPO_ROOT)),
    )
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


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)) if values else None,
        "median": float(statistics.median(values)) if values else None,
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "positive_count": int(sum(value > 0 for value in values)),
    }


def _median_by_task_slice_model(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        model_name = str(row["model_name"])
        if model_name in MODELS:
            grouped[(row["task_name"], row["slice_name"], model_name)].append(row)
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, group_rows in grouped.items():
        result[key] = {
            "step_auprc": float(statistics.median(float(row["step_auprc"]) for row in group_rows)),
            "outcome_auprc": float(statistics.median(float(row["outcome_auprc"]) for row in group_rows)),
            "ece": float(statistics.median(float(row["ece"]) for row in group_rows)),
            "source_platform": str(group_rows[0]["source_platform"]),
            "seed_count": len({int(row["split_seed"]) for row in group_rows}),
        }
    return result


def _best_seq(values: dict[str, float]) -> str:
    return max(SEQUENCE_MODELS, key=lambda model: values[model])


def _strict(values: dict[str, float]) -> bool:
    return all(values[seq] > values[nonseq] for seq in SEQUENCE_MODELS for nonseq in NONSEQUENTIAL_MODELS)


def _task_rows(
    *,
    current_rows: list[dict[str, str]],
    full_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    current = _median_by_task_slice_model(current_rows)
    full = _median_by_task_slice_model(full_rows)
    task_names = sorted({key[0] for key in current} & {key[0] for key in full})
    diagnostics: list[dict[str, Any]] = []
    for task_name in task_names:
        record: dict[str, Any] = {"task_name": task_name}
        source_platform = None
        complete = True
        for slice_name in SLICES:
            current_values: dict[str, float] = {}
            full_values: dict[str, float] = {}
            for model_name in MODELS:
                current_key = (task_name, slice_name, model_name)
                full_key = (task_name, slice_name, model_name)
                if current_key not in current or full_key not in full:
                    complete = False
                    continue
                current_values[model_name] = float(current[current_key]["step_auprc"])
                full_values[model_name] = float(full[full_key]["step_auprc"])
                source_platform = source_platform or str(current[current_key]["source_platform"])
                record[f"{slice_name}_{model_name}_current_step_auprc"] = current_values[model_name]
                record[f"{slice_name}_{model_name}_full_step_auprc"] = full_values[model_name]
                record[f"{slice_name}_{model_name}_history_gain"] = full_values[model_name] - current_values[model_name]
            if set(current_values) == set(MODELS) and set(full_values) == set(MODELS):
                current_best = _best_seq(current_values)
                full_best = _best_seq(full_values)
                record[f"{slice_name}_current_best_sequence_baseline"] = current_best
                record[f"{slice_name}_full_best_sequence_baseline"] = full_best
                record[f"{slice_name}_current_best_sequence_vs_flat"] = current_values[current_best] - current_values["flat_mlp"]
                record[f"{slice_name}_current_best_sequence_vs_linear"] = current_values[current_best] - current_values["linear"]
                record[f"{slice_name}_current_strict_sequence_support"] = _strict(current_values)
                record[f"{slice_name}_full_best_sequence_vs_window_flat"] = full_values[full_best] - full_values["flat_mlp"]
                record[f"{slice_name}_full_best_sequence_vs_linear"] = full_values[full_best] - full_values["linear"]
                record[f"{slice_name}_full_strict_sequence_support"] = _strict(full_values)
        if complete:
            record["source_platform"] = source_platform or ""
            diagnostics.append(record)
    return diagnostics


def _summary(diagnostics: list[dict[str, Any]], current_rows: list[dict[str, str]], full_rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "title": "Unified38 Current-Only Matrix vs Full-Prefix Analysis",
        "coverage": {
            "task_count": len(diagnostics),
            "current_only_seed_model_row_count": len(current_rows),
            "full_prefix_seed_model_row_count": len(full_rows),
            "models": MODELS,
            "slices": SLICES,
        },
        "history_gain": {},
        "current_only_internal": {},
        "full_prefix_reference": {},
    }
    for slice_name in SLICES:
        summary["history_gain"][slice_name] = {
            model_name: _stats([float(row[f"{slice_name}_{model_name}_history_gain"]) for row in diagnostics])
            for model_name in MODELS
        }
        summary["current_only_internal"][slice_name] = {
            "best_sequence_vs_flat": _stats([float(row[f"{slice_name}_current_best_sequence_vs_flat"]) for row in diagnostics]),
            "best_sequence_vs_linear": _stats([float(row[f"{slice_name}_current_best_sequence_vs_linear"]) for row in diagnostics]),
            "strict_support_count": int(sum(bool(row[f"{slice_name}_current_strict_sequence_support"]) for row in diagnostics)),
        }
        summary["full_prefix_reference"][slice_name] = {
            "best_sequence_vs_window_flat": _stats(
                [float(row[f"{slice_name}_full_best_sequence_vs_window_flat"]) for row in diagnostics]
            ),
            "best_sequence_vs_linear": _stats([float(row[f"{slice_name}_full_best_sequence_vs_linear"]) for row in diagnostics]),
            "strict_support_count": int(sum(bool(row[f"{slice_name}_full_strict_sequence_support"]) for row in diagnostics)),
        }
    return summary


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_markdown(path: Path, summary: dict[str, Any], diagnostics: list[dict[str, Any]], *, top_k: int) -> None:
    weakest_gru_gain = sorted(diagnostics, key=lambda row: row["canonical_gru_history_gain"])[:top_k]
    strongest_current_seq = sorted(
        diagnostics,
        key=lambda row: row["canonical_current_best_sequence_vs_flat"],
        reverse=True,
    )[:top_k]
    lines = [
        "# Unified38 Current-Only Matrix vs Full-Prefix Analysis",
        "",
        "## Coverage",
        "",
        f"- tasks: `{summary['coverage']['task_count']}`",
        f"- current-only seed-model rows: `{summary['coverage']['current_only_seed_model_row_count']}`",
        f"- full-prefix seed-model rows: `{summary['coverage']['full_prefix_seed_model_row_count']}`",
        f"- models: `{', '.join(summary['coverage']['models'])}`",
        "",
        "## History Gain",
        "",
    ]
    for slice_name in SLICES:
        lines.append(f"### {slice_name}")
        for model_name in MODELS:
            stat = summary["history_gain"][slice_name][model_name]
            lines.append(
                f"- `{model_name}` full-prefix minus current-only: mean `{stat['mean']:.4f}`, "
                f"median `{stat['median']:.4f}`, positive `{stat['positive_count']}/{stat['n']}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Current-Only Internal Structure",
            "",
        ]
    )
    for slice_name in SLICES:
        stat_flat = summary["current_only_internal"][slice_name]["best_sequence_vs_flat"]
        stat_linear = summary["current_only_internal"][slice_name]["best_sequence_vs_linear"]
        lines.append(
            f"- `{slice_name}` best current-only sequence vs FlatMLP-current: mean `{stat_flat['mean']:.4f}`, "
            f"median `{stat_flat['median']:.4f}`, positive `{stat_flat['positive_count']}/{stat_flat['n']}`; "
            f"strict support `{summary['current_only_internal'][slice_name]['strict_support_count']}/{stat_flat['n']}`."
        )
        lines.append(
            f"- `{slice_name}` best current-only sequence vs Linear-current: mean `{stat_linear['mean']:.4f}`, "
            f"median `{stat_linear['median']:.4f}`, positive `{stat_linear['positive_count']}/{stat_linear['n']}`."
        )
    lines.extend(
        [
            "",
            "## Full-Prefix Reference",
            "",
        ]
    )
    for slice_name in SLICES:
        stat = summary["full_prefix_reference"][slice_name]["best_sequence_vs_window_flat"]
        lines.append(
            f"- `{slice_name}` best full-prefix sequence vs window-flat FlatMLP: mean `{stat['mean']:.4f}`, "
            f"median `{stat['median']:.4f}`, positive `{stat['positive_count']}/{stat['n']}`; "
            f"strict support `{summary['full_prefix_reference'][slice_name]['strict_support_count']}/{stat['n']}`."
        )
    lines.extend(
        [
            "",
            "## Weakest Canonical GRU History Gain",
            "",
            *_markdown_table(
                weakest_gru_gain,
                [
                    "task_name",
                    "source_platform",
                    "canonical_gru_history_gain",
                    "canonical_temporal_cnn_history_gain",
                    "canonical_flat_mlp_history_gain",
                    "canonical_linear_history_gain",
                ],
            ),
            "",
            "## Strongest Current-Only Sequence Advantage",
            "",
            *_markdown_table(
                strongest_current_seq,
                [
                    "task_name",
                    "source_platform",
                    "canonical_current_best_sequence_baseline",
                    "canonical_current_best_sequence_vs_flat",
                    "canonical_current_best_sequence_vs_linear",
                ],
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    current_dir = _resolve(args.current_aggregate_dir)
    full_dir = _resolve(args.full_aggregate_dir)
    output_dir = _resolve(args.output_dir)
    current_rows = _read_csv(current_dir / "unified_seed_model_table_current.csv")
    full_rows = _read_csv(full_dir / "unified_seed_model_table_current.csv")
    diagnostics = _task_rows(current_rows=current_rows, full_rows=full_rows)
    summary = _summary(diagnostics, current_rows, full_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "Unified38_current_only_matrix_summary.json", summary)
    _write_csv(output_dir / "Unified38_current_only_matrix_task_diagnostics.csv", diagnostics)
    _write_markdown(output_dir / "Unified38_current_only_vs_full_prefix_report.md", summary, diagnostics, top_k=int(args.top_k))
    print(
        json.dumps(
            {
                "summary": str(output_dir / "Unified38_current_only_matrix_summary.json"),
                "report": str(output_dir / "Unified38_current_only_vs_full_prefix_report.md"),
                "task_count": summary["coverage"]["task_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
