'Analyze canonical Unified38 GRU window-sweep runs.'

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
DEFAULT_INPUT_ROOT = REPO_ROOT / "work" / "window_sweep"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "analysis"
EXPECTED_WINDOWS = [1, 4, 8, 16, 32]


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


def _discover_summaries(input_root: Path) -> list[dict[str, Any]]:
    return [_read_json(path) for path in sorted(input_root.rglob("window_sweep_summary.json"))]


def _task_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in summaries:
        windows = {int(key): value for key, value in payload["windows"].items()}
        missing = [window for window in EXPECTED_WINDOWS if window not in windows]
        if missing:
            raise ValueError(f"Window sweep analysis requires windows {EXPECTED_WINDOWS}; missing {missing} for {payload['task_name']}")
        row: dict[str, Any] = {
            "task_name": str(payload["task_name"]),
            "source_platform": str(payload["source_platform"]),
            "split_seed": int(payload["split_seed"]),
        }
        values = [float(windows[window]["test_step_catastrophic_auprc"]) for window in EXPECTED_WINDOWS]
        for window in EXPECTED_WINDOWS:
            row[f"window_{window}_step_auprc"] = float(windows[window]["test_step_catastrophic_auprc"])
        row["window_32_minus_window_1_step_auprc"] = float(row["window_32_step_auprc"] - row["window_1_step_auprc"])
        row["near_monotonic"] = bool(all(values[idx + 1] >= values[idx] - 0.005 for idx in range(len(values) - 1)))
        row["strict_monotonic"] = bool(all(values[idx + 1] >= values[idx] for idx in range(len(values) - 1)))
        row["saturates_by_8"] = bool((row["window_8_step_auprc"] - row["window_1_step_auprc"] >= 0.01) and (row["window_32_step_auprc"] - row["window_8_step_auprc"] <= 0.01))
        row["saturates_by_16"] = bool(
            not row["saturates_by_8"]
            and (row["window_16_step_auprc"] - row["window_1_step_auprc"] >= 0.01)
            and (row["window_32_step_auprc"] - row["window_16_step_auprc"] <= 0.01)
        )
        row["still_growing_to_32"] = bool(row["window_32_step_auprc"] - max(row["window_8_step_auprc"], row["window_16_step_auprc"]) >= 0.015)
        rows.append(row)
    return rows


def _platform_summary(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[str(row["source_platform"])].append(row)
    rows: list[dict[str, Any]] = []
    for source_platform, entries in sorted(grouped.items()):
        payload: dict[str, Any] = {"source_platform": source_platform, "task_count": len(entries)}
        for window in EXPECTED_WINDOWS:
            values = [float(entry[f"window_{window}_step_auprc"]) for entry in entries]
            payload[f"window_{window}_mean_step_auprc"] = float(sum(values) / len(values))
        payload["window_32_minus_window_1_mean"] = float(
            sum(float(entry["window_32_minus_window_1_step_auprc"]) for entry in entries) / len(entries)
        )
        rows.append(payload)
    return rows


def _summary(task_rows: list[dict[str, Any]], platform_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_window = {}
    for window in EXPECTED_WINDOWS:
        per_window[str(window)] = _stats([float(row[f"window_{window}_step_auprc"]) for row in task_rows])
    delta_stats = _stats([float(row["window_32_minus_window_1_step_auprc"]) for row in task_rows])
    aggregate_curve = [float(per_window[str(window)]["mean"]) for window in EXPECTED_WINDOWS]
    aggregate_near_monotonic = all(
        aggregate_curve[idx + 1] >= aggregate_curve[idx] - 0.005 for idx in range(len(aggregate_curve) - 1)
    )
    platform_consistent = all(float(row["window_32_minus_window_1_mean"]) >= -0.005 for row in platform_rows)
    return {
        "title": "Unified38 Canonical GRU Window Sweep Summary",
        "coverage": {"task_count": len(task_rows), "slice_name": "canonical", "split_seeds": [0], "window_sizes": EXPECTED_WINDOWS},
        "per_window_step_auprc": per_window,
        "window_32_minus_window_1": delta_stats,
        "monotonicity": {
            "strict_task_count": int(sum(bool(row["strict_monotonic"]) for row in task_rows)),
            "near_task_count": int(sum(bool(row["near_monotonic"]) for row in task_rows)),
            "aggregate_near_monotonic": bool(aggregate_near_monotonic),
            "saturates_by_8_count": int(sum(bool(row["saturates_by_8"]) for row in task_rows)),
            "saturates_by_16_count": int(sum(bool(row["saturates_by_16"]) for row in task_rows)),
            "still_growing_to_32_count": int(sum(bool(row["still_growing_to_32"]) for row in task_rows)),
        },
        "decision_gate": {
            "positive_tasks": int(delta_stats["positive_count"]),
            "mean_window_32_minus_window_1": float(delta_stats["mean"]),
            "aggregate_curve_rises_or_plateaus": bool(aggregate_near_monotonic and aggregate_curve[-1] >= aggregate_curve[0] + 0.01),
            "platform_consistent": bool(platform_consistent),
            "meets_no_seed_expansion_threshold": bool(
                int(delta_stats["positive_count"]) >= 30
                and bool(aggregate_near_monotonic and aggregate_curve[-1] >= aggregate_curve[0] + 0.01)
                and bool(platform_consistent)
            ),
        },
    }


def _write_report(path: Path, summary: dict[str, Any], task_rows: list[dict[str, Any]], platform_rows: list[dict[str, Any]]) -> None:
    weakest = sorted(task_rows, key=lambda row: float(row["window_32_minus_window_1_step_auprc"]))[:10]
    strongest = sorted(task_rows, key=lambda row: float(row["window_32_minus_window_1_step_auprc"]), reverse=True)[:10]
    lines = [
        "# GRU window-size comparison",
        "",
        "## Coverage",
        "",
        f"- tasks: `{summary['coverage']['task_count']}`",
        f"- windows: `{summary['coverage']['window_sizes']}`",
        "",
        "## Mean curve",
        "",
    ]
    for window in EXPECTED_WINDOWS:
        stats = summary["per_window_step_auprc"][str(window)]
        lines.append(f"- `window={window}`: mean `{stats['mean']:.4f}`，median `{stats['median']:.4f}`")
    lines.extend(
        [
            "",
            "## Curve summary",
            "",
            f"- `window=32 - window=1`: mean `{summary['window_32_minus_window_1']['mean']:.4f}`，positive `{summary['window_32_minus_window_1']['positive_count']}/{summary['window_32_minus_window_1']['n']}`。",
            f"- near-monotonic tasks: `{summary['monotonicity']['near_task_count']}/{summary['coverage']['task_count']}`",
            f"- saturates by 8: `{summary['monotonicity']['saturates_by_8_count']}`",
            f"- saturates by 16: `{summary['monotonicity']['saturates_by_16_count']}`",
            f"- still growing to 32: `{summary['monotonicity']['still_growing_to_32_count']}`",
            f"- `meets_no_seed_expansion_threshold = {summary['decision_gate']['meets_no_seed_expansion_threshold']}`",
            "",
            "## Source-platform summaries",
            "",
        ]
    )
    for row in platform_rows:
        lines.append(
            f"- `{row['source_platform']}`: "
            + ", ".join(f"`w{window}={float(row[f'window_{window}_mean_step_auprc']):.4f}`" for window in EXPECTED_WINDOWS)
            + f", `w32-w1={float(row['window_32_minus_window_1_mean']):.4f}`"
        )
    lines.extend(["", "## Tasks with the smallest gain", ""])
    for row in weakest:
        lines.append(
            f"- `{row['task_name']}` ({row['source_platform']}): "
            f"`w1={float(row['window_1_step_auprc']):.4f}`, "
            f"`w8={float(row['window_8_step_auprc']):.4f}`, "
            f"`w16={float(row['window_16_step_auprc']):.4f}`, "
            f"`w32={float(row['window_32_step_auprc']):.4f}`"
        )
    lines.extend(["", "## Tasks with the largest gain", ""])
    for row in strongest:
        lines.append(
            f"- `{row['task_name']}` ({row['source_platform']}): "
            f"`w1={float(row['window_1_step_auprc']):.4f}`, "
            f"`w8={float(row['window_8_step_auprc']):.4f}`, "
            f"`w16={float(row['window_16_step_auprc']):.4f}`, "
            f"`w32={float(row['window_32_step_auprc']):.4f}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = _resolve(args.input_root)
    output_dir = _resolve(args.output_dir)
    summaries = _discover_summaries(input_root)
    if not summaries:
        raise FileNotFoundError(f"No window_sweep_summary.json found under: {input_root}")
    task_rows = _task_rows(summaries)
    platform_rows = _platform_summary(task_rows)
    summary = _summary(task_rows, platform_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "Unified38_gru_window_sweep_summary.json", summary)
    _write_csv(output_dir / "Unified38_gru_window_sweep_task_table.csv", task_rows)
    _write_report(output_dir / "Unified38_gru_window_sweep_report_zh.md", summary, task_rows, platform_rows)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "Unified38_gru_window_sweep_summary.json"),
                "report": str(output_dir / "Unified38_gru_window_sweep_report_zh.md"),
                "task_count": len(task_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
