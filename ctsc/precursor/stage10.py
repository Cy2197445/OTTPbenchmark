"""Construct fixed-model tables and the history-gain figure."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ctsc.precursor.stage4 import build_frozen_statistics as stage4_statistics
from ctsc.precursor.stage5 import build_frozen_statistics as stage5_statistics


MODELS = ("gru", "flat_mlp", "transformer")
CONDITIONS = ("canonical_full", "canonical_current_only", "sensitivity_full")
STATS_FIELDS = (
    "n", "mean", "median", "q1", "q3", "iqr", "min", "max",
    "positive_count", "positive_proportion", "bootstrap_ci95_low",
    "bootstrap_ci95_high", "paired_sign_flip_p_raw", "paired_dz",
    "holm_family_size", "holm_adjusted_p",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def compare_numeric_rows(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]],
                         keys: Sequence[str], fields: Sequence[str], *, tolerance: float = 1e-11) -> None:
    def indexed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], Mapping[str, Any]]:
        result: dict[tuple[str, ...], Mapping[str, Any]] = {}
        for row in rows:
            key = tuple(str(row[field]) for field in keys)
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = row
        return result

    left, right = indexed(actual), indexed(expected)
    if left.keys() != right.keys():
        raise ValueError(f"Row key mismatch: missing={sorted(right.keys() - left.keys())[:3]}, "
                         f"extra={sorted(left.keys() - right.keys())[:3]}")
    for key, row in left.items():
        for field in fields:
            a, b = float(row[field]), float(right[key][field])
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
                raise ValueError(f"Numeric mismatch {key}/{field}: {a} versus {b}")






def validate_fair_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["condition_id"]), str(row["task_name"]), str(row["model_name"]))
        if key in index:
            raise ValueError(f"Duplicate fair-matrix row: {key}")
        if key[0] not in CONDITIONS or key[2] not in MODELS:
            raise ValueError(f"Unexpected fair-matrix condition/model: {key}")
        if row["seed_aggregation"] != "arithmetic_mean_split_seeds_0_1_2":
            raise ValueError(f"Wrong split-seed aggregation: {key}")
        for field in ("step_auprc", "matched_step_prevalence", "outcome_auprc",
                      "matched_outcome_prevalence", "step_ece"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field}: {key}")
        if not math.isclose(float(row["step_auprc_minus_prevalence"]),
                            float(row["step_auprc"]) - float(row["matched_step_prevalence"]), abs_tol=1e-11):
            raise ValueError(f"Step endpoint mismatch: {key}")
        index[key] = row
    tasks = {key[1] for key in index}
    if len(tasks) != 38 or len(index) != 38 * len(CONDITIONS) * len(MODELS):
        raise ValueError("Incomplete 38-task three-model three-condition fair matrix")
    if set(index) != {(condition, task, model) for condition in CONDITIONS for task in tasks for model in MODELS}:
        raise ValueError("Missing fair-matrix cell")
    return index


def make_tables(index: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = sorted({key[1] for key in index})
    main: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for task in tasks:
        ref = index[("canonical_full", task, "gru")]
        prevalence = float(ref["matched_step_prevalence"])
        outcome_prev = float(ref["matched_outcome_prevalence"])
        for model in MODELS:
            full = index[("canonical_full", task, model)]
            current = index[("canonical_current_only", task, model)]
            if not math.isclose(float(full["matched_step_prevalence"]), prevalence, abs_tol=1e-11):
                raise ValueError(f"Nonmatched Step prevalence: {task}")
            if not math.isclose(float(full["matched_outcome_prevalence"]), outcome_prev, abs_tol=1e-11):
                raise ValueError(f"Nonmatched Outcome prevalence: {task}")
            history.append({"task_name": task, "model_name": model,
                            "full_step_auprc": float(full["step_auprc"]),
                            "current_only_step_auprc": float(current["step_auprc"]),
                            "full_minus_current_only": float(full["step_auprc"]) - float(current["step_auprc"])})
            auxiliary.append({"task_name": task, "model_name": model,
                              "canonical_outcome_auprc": float(full["outcome_auprc"]),
                              "matched_outcome_prevalence": outcome_prev,
                              "canonical_step_ece": float(full["step_ece"]),
                              "sensitivity_outcome_auprc": float(index[("sensitivity_full", task, model)]["outcome_auprc"])})
        main.append({"task_name": task, "matched_step_prevalence": prevalence,
                     **{f"{model}_step_auprc": float(index[("canonical_full", task, model)]["step_auprc"]) for model in MODELS},
                     **{f"{model}_step_lift": float(index[("canonical_full", task, model)]["step_auprc"]) - prevalence for model in MODELS}})
    return main, auxiliary, history




def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, precision: int = 4) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)
    return "\n".join(["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |",
                      *("| " + " | ".join(fmt(row[field]) for field in fields) + " |" for row in rows)]) + "\n"




def _draw_history_figure(output: Path, stats5: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.hashsalt": "ottp-g10"})
    colors = {"gru": "#0072B2", "flat_mlp": "#D55E00", "transformer": "#009E73"}
    labels = {"gru": "GRU", "flat_mlp": "FlatMLP", "transformer": "Transformer"}
    lookup = {str(row["comparison"]): row for row in stats5}
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.axvline(0, color="#777777", linewidth=0.85, linestyle="--", zorder=0)
    for position, model in enumerate(MODELS):
        row = lookup[f"{model}_full_minus_current_only"]
        value = float(row["mean"])
        low, high = float(row["bootstrap_ci95_low"]), float(row["bootstrap_ci95_high"])
        ax.errorbar(value, position, xerr=[[value - low], [high - value]], fmt="o",
                    color=colors[model], markersize=5, capsize=2.5, linewidth=1.2)
    ax.set_yticks(range(3), [labels[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(-0.018, 0.051)
    ax.set_xlabel("Mean Step AUPRC gain (full − current-only)")
    ax.set_title("38 DSRL tasks; task-bootstrap 95% CI", fontsize=8.5)
    ax.grid(axis="x", color="#dddddd", linewidth=0.55)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.85, bottom=0.25)
    target = output / "paper_assets/figures"
    target.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "OTTP G10 frozen builder", "CreationDate": datetime(2026, 9, 24, tzinfo=timezone.utc)}
    fig.savefig(target / "history_gain_fixed_models.pdf", metadata=metadata)
    fig.savefig(target / "history_gain_fixed_models.svg", metadata={"Date": "2026-09-24", "Creator": "OTTP G10 frozen builder"})
    fig.savefig(target / "history_gain_fixed_models.png", dpi=300, metadata={"Software": "OTTP G10 frozen builder"})
    plt.close(fig)


