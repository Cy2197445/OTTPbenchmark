'Summarize and diagnose the current-branch Unified38 fresh-rerun results.'

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
DEFAULT_AGGREGATE_DIR = REPO_ROOT / "work" / "reference_diagnostics" / "aggregates"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "work" / "reference_diagnostics" / "analysis"
SEQUENCE_MODELS = ["gru", "temporal_cnn"]
NONSEQUENTIAL_MODELS = ["flat_mlp", "linear"]
REFERENCE_MODELS = ["prevalence"]
ALL_MODELS = [*SEQUENCE_MODELS, *NONSEQUENTIAL_MODELS, *REFERENCE_MODELS]


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


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)) if values else None,
        "median": float(statistics.median(values)) if values else None,
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "positive_count": int(sum(value > 0 for value in values)),
    }


def _median_metric(rows: list[dict[str, str]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _task_model_medians(seed_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["task_name"], row["slice_name"], row["model_name"])].append(row)
    return {
        key: {
            "step_auprc": _median_metric(rows, "step_auprc"),
            "outcome_auprc": _median_metric(rows, "outcome_auprc"),
            "ece": _median_metric(rows, "ece"),
            "lead_time": _median_metric(rows, "normalized_lead_time"),
        }
        for key, rows in grouped.items()
    }


def _platform_stats(registry_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in registry_rows:
        grouped[row["source_platform"]].append(row)
    result: list[dict[str, Any]] = []
    for platform, rows in sorted(grouped.items()):
        canonical_gaps = [_float(row, "canonical_gap_vs_flat") for row in rows]
        sensitivity_gaps = [_float(row, "sensitivity_gap_vs_flat") for row in rows]
        result.append(
            {
                "source_platform": platform,
                "task_count": len(rows),
                "canonical_gap_mean": _stats(canonical_gaps)["mean"],
                "canonical_gap_median": _stats(canonical_gaps)["median"],
                "canonical_positive": int(sum(value > 0 for value in canonical_gaps)),
                "canonical_strict_support": int(sum(_bool(row["canonical_strict_task_level_sequence_support"]) for row in rows)),
                "sensitivity_gap_mean": _stats(sensitivity_gaps)["mean"],
                "sensitivity_gap_median": _stats(sensitivity_gaps)["median"],
                "sensitivity_positive": int(sum(value > 0 for value in sensitivity_gaps)),
                "sensitivity_strict_support": int(sum(_bool(row["sensitivity_strict_task_level_sequence_support"]) for row in rows)),
            }
        )
    return result


def _seed_level_stats(seed_rows: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in seed_rows:
        grouped[(row["task_name"], row["slice_name"], int(row["split_seed"]))][row["model_name"]] = float(row["step_auprc"])
    result: dict[str, Any] = {}
    for slice_name in ["canonical", "sensitivity"]:
        slice_rows = [models for (_, slice_value, _), models in grouped.items() if slice_value == slice_name]
        best_vs_flat = []
        best_vs_linear = []
        strict = []
        for models in slice_rows:
            best_seq = max(SEQUENCE_MODELS, key=lambda model: models[model])
            best_vs_flat.append(models[best_seq] - models["flat_mlp"])
            best_vs_linear.append(models[best_seq] - models["linear"])
            strict.append(all(models[seq] > models[nonseq] for seq in SEQUENCE_MODELS for nonseq in NONSEQUENTIAL_MODELS))
        result[slice_name] = {
            "task_seed_count": len(slice_rows),
            "best_sequence_vs_flat": _stats(best_vs_flat),
            "best_sequence_vs_linear": _stats(best_vs_linear),
            "strict_support_count": int(sum(strict)),
        }
    return result


def _model_stats(seed_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["slice_name"], row["model_name"])].append(float(row["step_auprc"]))
    rows: list[dict[str, Any]] = []
    for slice_name in ["canonical", "sensitivity"]:
        for model_name in ALL_MODELS:
            values = grouped[(slice_name, model_name)]
            stat = _stats(values)
            rows.append(
                {
                    "slice_name": slice_name,
                    "model_name": model_name,
                    "mean_step_auprc": stat["mean"],
                    "median_step_auprc": stat["median"],
                    "min_step_auprc": stat["min"],
                    "max_step_auprc": stat["max"],
                    "row_count": stat["n"],
                }
            )
    return rows


def _diagnostic_rows(registry_rows: list[dict[str, str]], model_medians: dict[tuple[str, str, str], dict[str, float]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for row in registry_rows:
        task_name = row["task_name"]
        record: dict[str, Any] = {
            "task_name": task_name,
            "source_platform": row["source_platform"],
            "canonical_gap_vs_flat": _float(row, "canonical_gap_vs_flat"),
            "sensitivity_gap_vs_flat": _float(row, "sensitivity_gap_vs_flat"),
            "gap_delta_sensitivity_minus_canonical": _float(row, "sensitivity_gap_vs_flat") - _float(row, "canonical_gap_vs_flat"),
            "canonical_gap_vs_linear": _float(row, "canonical_gap_vs_linear"),
            "sensitivity_gap_vs_linear": _float(row, "sensitivity_gap_vs_linear"),
            "canonical_best_sequence_baseline": row["best_sequence_baseline"],
            "sensitivity_best_sequence_baseline": row["sensitivity_best_sequence_baseline"],
            "canonical_strict_support": _bool(row["canonical_strict_task_level_sequence_support"]),
            "sensitivity_strict_support": _bool(row["sensitivity_strict_task_level_sequence_support"]),
        }
        for slice_name in ["canonical", "sensitivity"]:
            values = {model: model_medians[(task_name, slice_name, model)]["step_auprc"] for model in ALL_MODELS}
            best_seq = max(SEQUENCE_MODELS, key=lambda model: values[model])
            weaker_seq = min(SEQUENCE_MODELS, key=lambda model: values[model])
            record[f"{slice_name}_best_seq_step_auprc"] = values[best_seq]
            record[f"{slice_name}_flat_mlp_step_auprc"] = values["flat_mlp"]
            record[f"{slice_name}_linear_step_auprc"] = values["linear"]
            record[f"{slice_name}_prevalence_step_auprc"] = values["prevalence"]
            record[f"{slice_name}_weaker_sequence_minus_flat"] = values[weaker_seq] - values["flat_mlp"]
            record[f"{slice_name}_best_sequence_minus_prevalence"] = values[best_seq] - values["prevalence"]
        diagnostics.append(record)
    return diagnostics


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int | None = None) -> list[str]:
    selected = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_markdown(path: Path, summary: dict[str, Any], platform_rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    weakest = sorted(diagnostics, key=lambda row: row["canonical_gap_vs_flat"])[:12]
    sensitivity_weakest = sorted(diagnostics, key=lambda row: row["sensitivity_gap_vs_flat"])[:12]
    strict_fail_both = [
        row for row in diagnostics if not row["canonical_strict_support"] and not row["sensitivity_strict_support"]
    ]
    lines = [
        "# Unified38 Current Fresh Rerun Analysis",
        "",
        "## Coverage",
        "",
        f"- tasks: `{summary['coverage']['task_count']}`",
        f"- suite summaries: `{summary['coverage']['suite_count']}`",
        f"- model metric rows: `{summary['coverage']['seed_model_row_count']}`",
        f"- audit status: `{summary['coverage']['audit_status']}`",
        "",
        "## Main Results",
        "",
        f"- Canonical gap vs FlatMLP: mean `{summary['canonical']['gap_vs_flat']['mean']:.4f}`, median `{summary['canonical']['gap_vs_flat']['median']:.4f}`, positive `{summary['canonical']['gap_vs_flat']['positive_count']}/38`.",
        f"- Canonical gap vs Linear: mean `{summary['canonical']['gap_vs_linear']['mean']:.4f}`, median `{summary['canonical']['gap_vs_linear']['median']:.4f}`, positive `{summary['canonical']['gap_vs_linear']['positive_count']}/38`.",
        f"- Sensitivity gap vs FlatMLP: mean `{summary['sensitivity']['gap_vs_flat']['mean']:.4f}`, median `{summary['sensitivity']['gap_vs_flat']['median']:.4f}`, positive `{summary['sensitivity']['gap_vs_flat']['positive_count']}/38`.",
        f"- Sensitivity gap vs Linear: mean `{summary['sensitivity']['gap_vs_linear']['mean']:.4f}`, median `{summary['sensitivity']['gap_vs_linear']['median']:.4f}`, positive `{summary['sensitivity']['gap_vs_linear']['positive_count']}/38`.",
        f"- Strict sequence-family support: canonical `{summary['canonical']['strict_support_count']}/38`, sensitivity `{summary['sensitivity']['strict_support_count']}/38`, both slices `{summary['both_slice_strict_support_count']}/38`.",
        f"- Gap delta sensitivity minus canonical: mean `{summary['sensitivity_minus_canonical_gap']['mean']:.4f}`, median `{summary['sensitivity_minus_canonical_gap']['median']:.4f}`, positive `{summary['sensitivity_minus_canonical_gap']['positive_count']}/38`.",
        "",
        "## Platform Stratification",
        "",
        *_markdown_table(
            platform_rows,
            [
                "source_platform",
                "task_count",
                "canonical_gap_mean",
                "canonical_gap_median",
                "canonical_strict_support",
                "sensitivity_gap_mean",
                "sensitivity_gap_median",
                "sensitivity_strict_support",
            ],
        ),
        "",
        "## Weakest Canonical Gap Tasks",
        "",
        *_markdown_table(
            weakest,
            [
                "task_name",
                "source_platform",
                "canonical_gap_vs_flat",
                "sensitivity_gap_vs_flat",
                "canonical_strict_support",
                "sensitivity_strict_support",
                "canonical_weaker_sequence_minus_flat",
            ],
        ),
        "",
        "## Weakest Sensitivity Gap Tasks",
        "",
        *_markdown_table(
            sensitivity_weakest,
            [
                "task_name",
                "source_platform",
                "canonical_gap_vs_flat",
                "sensitivity_gap_vs_flat",
                "canonical_strict_support",
                "sensitivity_strict_support",
                "sensitivity_weaker_sequence_minus_flat",
            ],
        ),
        "",
        "## Tasks Failing Strict Support In Both Slices",
        "",
        *_markdown_table(
            strict_fail_both,
            [
                "task_name",
                "source_platform",
                "canonical_gap_vs_flat",
                "sensitivity_gap_vs_flat",
                "canonical_weaker_sequence_minus_flat",
                "sensitivity_weaker_sequence_minus_flat",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- The current fresh rerun preserves the broad positive best-sequence-vs-FlatMLP and best-sequence-vs-Linear direction on all 38 tasks.",
        "- The strict support criterion is much more conservative because it requires both sequential baselines to beat both nonsequential baselines; it is not equivalent to best-sequence support.",
        "- The current rerun is therefore a positive but more conservative current benchmark main-result layer, and weak-task diagnostics should use this current layer.",
        "- Anti-shortcut boundary fields are not refreshed by this mainline rerun; a full current 38/38 anti-shortcut profile run is still separate.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    aggregate_dir = _resolve(args.aggregate_dir)
    output_dir = _resolve(args.output_dir)
    registry_rows = _read_csv(aggregate_dir / "unified_task_registry_current.csv")
    seed_rows = _read_csv(aggregate_dir / "unified_seed_model_table_current.csv")
    audit_path = aggregate_dir.parent / "audit" / "Unified38_current_rerun_audit_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"status": "unknown"}
    model_medians = _task_model_medians(seed_rows)
    diagnostics = _diagnostic_rows(registry_rows, model_medians)

    canonical_gap_flat = [_float(row, "canonical_gap_vs_flat") for row in registry_rows]
    canonical_gap_linear = [_float(row, "canonical_gap_vs_linear") for row in registry_rows]
    sensitivity_gap_flat = [_float(row, "sensitivity_gap_vs_flat") for row in registry_rows]
    sensitivity_gap_linear = [_float(row, "sensitivity_gap_vs_linear") for row in registry_rows]
    gap_delta = [row["gap_delta_sensitivity_minus_canonical"] for row in diagnostics]
    platform_rows = _platform_stats(registry_rows)
    model_rows = _model_stats(seed_rows)
    seed_stats = _seed_level_stats(seed_rows)

    best_counts = {
        "canonical": dict(Counter(row["best_sequence_baseline"] for row in registry_rows)),
        "sensitivity": dict(Counter(row["sensitivity_best_sequence_baseline"] for row in registry_rows)),
    }
    summary = {
        "title": "Unified38 Current Fresh Rerun Analysis",
        "coverage": {
            "task_count": len(registry_rows),
            "suite_count": len(registry_rows) * 2 * 3,
            "seed_model_row_count": len(seed_rows),
            "audit_status": audit.get("status", "unknown"),
        },
        "canonical": {
            "gap_vs_flat": _stats(canonical_gap_flat),
            "gap_vs_linear": _stats(canonical_gap_linear),
            "strict_support_count": int(sum(_bool(row["canonical_strict_task_level_sequence_support"]) for row in registry_rows)),
            "best_sequence_counts": best_counts["canonical"],
        },
        "sensitivity": {
            "gap_vs_flat": _stats(sensitivity_gap_flat),
            "gap_vs_linear": _stats(sensitivity_gap_linear),
            "strict_support_count": int(sum(_bool(row["sensitivity_strict_task_level_sequence_support"]) for row in registry_rows)),
            "best_sequence_counts": best_counts["sensitivity"],
        },
        "both_slice_strict_support_count": int(
            sum(
                _bool(row["canonical_strict_task_level_sequence_support"])
                and _bool(row["sensitivity_strict_task_level_sequence_support"])
                for row in registry_rows
            )
        ),
        "sensitivity_minus_canonical_gap": _stats(gap_delta),
        "seed_level": seed_stats,
        "platform_rows": platform_rows,
        "model_rows": model_rows,
        "weakest_canonical_gap_tasks": sorted(diagnostics, key=lambda row: row["canonical_gap_vs_flat"])[: int(args.top_k)],
        "weakest_sensitivity_gap_tasks": sorted(diagnostics, key=lambda row: row["sensitivity_gap_vs_flat"])[: int(args.top_k)],
        "strict_support_fail_both_slices": [
            row for row in diagnostics if not row["canonical_strict_support"] and not row["sensitivity_strict_support"]
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "Unified38_current_rerun_analysis_summary.json", summary)
    _write_csv(output_dir / "Unified38_current_rerun_task_diagnostics.csv", diagnostics)
    _write_csv(output_dir / "Unified38_current_rerun_platform_summary.csv", platform_rows)
    _write_csv(output_dir / "Unified38_current_rerun_model_summary.csv", model_rows)
    _write_markdown(output_dir / "Unified38_current_rerun_analysis_report.md", summary, platform_rows, diagnostics)
    print(
        json.dumps(
            {
                "analysis_summary": str(output_dir / "Unified38_current_rerun_analysis_summary.json"),
                "analysis_report": str(output_dir / "Unified38_current_rerun_analysis_report.md"),
                "canonical_gap_vs_flat_mean": summary["canonical"]["gap_vs_flat"]["mean"],
                "canonical_strict_support": summary["canonical"]["strict_support_count"],
                "sensitivity_strict_support": summary["sensitivity"]["strict_support_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
