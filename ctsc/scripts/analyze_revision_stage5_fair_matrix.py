"""Task aggregation and model-complexity helpers."""
from __future__ import annotations
import math
from statistics import mean
from typing import Any, Mapping, Sequence
import numpy as np
from ctsc.precursor.cc1 import FAIR_MODELS, SPLIT_SEEDS

def _task_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["condition_id"]), str(row["task_name"]), str(row["model_name"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (condition, task, model), rows in sorted(grouped.items()):
        if sorted(int(row["split_seed"]) for row in rows) != list(SPLIT_SEEDS):
            raise ValueError(f"Seed coverage mismatch: {(condition, task, model)}")
        mean = lambda field: float(np.mean([float(row[field]) for row in rows]))
        output.append({
            "condition_id": condition,
            "task_name": task,
            "model_name": model,
            "source_platform": rows[0]["source_platform"],
            "seed_aggregation": "arithmetic_mean_split_seeds_0_1_2",
            "step_auprc": mean("step_auprc"),
            "matched_step_prevalence": mean("matched_step_prevalence"),
            "step_auprc_minus_prevalence": mean("step_auprc_minus_prevalence"),
            "outcome_auprc": mean("outcome_auprc"),
            "matched_outcome_prevalence": mean("matched_outcome_prevalence"),
            "step_ece": mean("step_ece"),
            "step_lead_median": mean("step_lead_median"),
        })
    if len(output) != 342:
        raise ValueError(f"Expected 342 task/model/condition rows, got {len(output)}")
    return output


def _fixed_main_table(task_rows: Sequence[Mapping[str, Any]], tasks: Sequence[str]) -> list[dict[str, Any]]:
    index = {(row["condition_id"], row["task_name"], row["model_name"]): row for row in task_rows}
    output = []
    for task in tasks:
        model_rows = {model: index[("canonical_full", task, model)] for model in FAIR_MODELS}
        prevalences = {float(row["matched_step_prevalence"]) for row in model_rows.values()}
        if len(prevalences) != 1:
            raise ValueError(f"Canonical matched prevalence differs across models for {task}")
        output.append({
            "task_name": task,
            "source_platform": model_rows["gru"]["source_platform"],
            "matched_step_prevalence": next(iter(prevalences)),
            "gru_step_auprc": model_rows["gru"]["step_auprc"],
            "flat_mlp_step_auprc": model_rows["flat_mlp"]["step_auprc"],
            "transformer_step_auprc": model_rows["transformer"]["step_auprc"],
            "gru_lift": model_rows["gru"]["step_auprc_minus_prevalence"],
            "flat_mlp_lift": model_rows["flat_mlp"]["step_auprc_minus_prevalence"],
            "transformer_lift": model_rows["transformer"]["step_auprc_minus_prevalence"],
        })
    return output


def _theoretical_macs(model: str, config: Mapping[str, Any]) -> tuple[int, str, int]:
    window = int(config["window_size"])
    classes = int(config["num_classes"])
    if model == "flat_mlp":
        hidden = int(config["hidden_dim"])
        flat_dim = int(config["input_dim"])
        feature_dim = flat_dim // window
        depth = int(config["mlp_depth"])
        macs = flat_dim * hidden + max(0, depth - 1) * hidden * hidden + hidden * classes
        return int(macs), "W*D*H + (depth-1)*H^2 + H*C", feature_dim
    feature_dim = int(config["input_dim"])
    if model == "gru":
        hidden = int(config["hidden_dim"])
        layers = int(config["num_layers"])
        macs = window * 3 * (feature_dim * hidden + hidden * hidden)
        if layers > 1:
            macs += (layers - 1) * window * 3 * (hidden * hidden + hidden * hidden)
        macs += hidden * hidden + hidden * classes
        return int(macs), "W*3*(D*H + H^2) + head(H^2 + H*C)", feature_dim
    width = int(config["transformer_d_model"])
    layers = int(config["transformer_num_layers"])
    macs = window * feature_dim * width + layers * (12 * window * width * width + 2 * window * window * width) + width * width + width * classes
    return int(macs), "W*D*M + L*(12*W*M^2 + 2*W^2*M) + M^2 + M*C", feature_dim

