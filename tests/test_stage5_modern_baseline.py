from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from ctsc.precursor.model import (
    CausalTransformerPrecursorModel,
    MODEL_REGISTRY,
    build_precursor_model,
    select_transformer_d_model,
    trainable_parameter_count,
)
from ctsc.precursor.stage5 import (
    FINAL_CONDITIONS,
    FINAL_MODELS,
    LEARNING_RATES,
    SPLIT_SEEDS,
    TUNING_MODELS,
    TUNING_TASKS,
    build_frozen_statistics,
    final_jobs,
    frozen_matrix_manifest,
    learning_rate_compatibility_gate,
    select_learning_rates,
    tuning_jobs,
)
from ctsc.scripts.run_precursor_baseline_suite import DEFAULT_MODELS


def _tasks() -> list[str]:
    return [*TUNING_TASKS, *[f"SyntheticTask{index:02d}" for index in range(34)]]


def test_frozen_stage5_matrix_counts_and_scope() -> None:
    tasks = _tasks()
    manifest = frozen_matrix_manifest(tasks)
    assert manifest["tuning"]["expected_model_runs"] == 36
    assert manifest["final"]["expected_transformer_runs"] == 342
    assert len(tuning_jobs()) == 36
    assert len(final_jobs(tasks)) == 342
    assert len(FINAL_CONDITIONS) == 3
    assert set(FINAL_MODELS) == {"gru", "flat_mlp", "transformer"}
    assert "future_only" in manifest["excluded"]
    assert "transformer" not in DEFAULT_MODELS
    assert "transformer" in MODEL_REGISTRY


def test_transformer_shape_length_limit_and_last_position_readout() -> None:
    torch.manual_seed(7)
    model = CausalTransformerPrecursorModel(input_dim=11, d_model=52)
    values = torch.randn(3, 32, 11)
    encoded = model.encode(values)
    logits = model(values)
    assert encoded.shape == (3, 32, 52)
    assert logits.shape == (3, 3)
    assert torch.allclose(logits, model.head(encoded[:, -1, :]))
    with pytest.raises(ValueError, match="Sequence length"):
        model(torch.randn(1, 33, 11))


def test_causal_mask_prevents_future_token_leakage() -> None:
    torch.manual_seed(8)
    model = CausalTransformerPrecursorModel(input_dim=11, d_model=52).eval()
    original = torch.randn(2, 32, 11)
    modified = original.clone()
    modified[:, 18:, :] = torch.randn_like(modified[:, 18:, :]) * 100.0
    with torch.no_grad():
        before = model.encode(original)
        after = model.encode(modified)
    assert torch.equal(before[:, :18, :], after[:, :18, :])
    assert not torch.equal(before[:, 18:, :], after[:, 18:, :])


@pytest.mark.parametrize(
    ("feature_dim", "expected_d_model"),
    [(11, 52), (25, 52), (32, 56), (64, 56), (76, 60), (92, 60), (263, 76)],
)
def test_mechanical_parameter_matching(feature_dim: int, expected_d_model: int) -> None:
    match = select_transformer_d_model(feature_dim)
    assert match["d_model"] == expected_d_model
    assert abs(float(match["relative_deviation_percent"])) <= 10.0


def test_all_observed_feature_dimensions_are_within_tolerance() -> None:
    for feature_dim in (11, 12, 16, 23, 24, 25, 32, 37, 43, 44, 64, 76, 80, 92, 263):
        match = select_transformer_d_model(feature_dim)
        assert abs(float(match["relative_deviation_percent"])) <= 10.0


def test_bundle_configuration_round_trip_reconstructs_transformer() -> None:
    original = build_precursor_model(
        "transformer",
        input_dim=32,
        hidden_dim=128,
        dropout=0.0,
        transformer_max_length=32,
    )
    config = {
        "model_name": "transformer",
        "input_dim": 32,
        "hidden_dim": 128,
        "dropout": 0.0,
        "transformer_d_model": original.d_model,
        "transformer_num_layers": 2,
        "transformer_num_heads": 4,
        "transformer_ffn_multiplier": 4,
        "transformer_max_length": 32,
    }
    rebuilt = build_precursor_model(**config)
    rebuilt.load_state_dict(copy.deepcopy(original.state_dict()))
    assert trainable_parameter_count(rebuilt) == trainable_parameter_count(original)
    with torch.no_grad():
        values = torch.randn(2, 32, 32)
        assert torch.equal(original(values), rebuilt(values))


def _tuning_rows() -> list[dict]:
    rows = []
    for task_index, task in enumerate(TUNING_TASKS):
        for model_index, model in enumerate(TUNING_MODELS):
            for lr_index, lr in enumerate(LEARNING_RATES):
                rows.append(
                    {
                        "task_name": task,
                        "model_name": model,
                        "learning_rate": lr,
                        "validation_step_catastrophic_auprc": 0.4 + 0.01 * lr_index + 0.001 * model_index + task_index * 1e-6,
                        "test_step_catastrophic_auprc": -1000.0 + lr_index,
                    }
                )
    return rows


def test_learning_rate_selection_is_validation_only_and_deterministic() -> None:
    rows = _tuning_rows()
    scores_a, selection_a = select_learning_rates(rows)
    changed_test = copy.deepcopy(rows)
    for index, row in enumerate(changed_test):
        row["test_step_catastrophic_auprc"] = float(index * 1000000)
    scores_b, selection_b = select_learning_rates(changed_test)
    assert scores_a == scores_b
    assert selection_a == selection_b
    assert all(
        selection_a["models"][model]["selected_learning_rate"] == 3e-3 for model in TUNING_MODELS
    )
    assert not any("test" in field for field in scores_a[0])


def test_learning_rate_strict_tie_rule_and_compatibility_gate() -> None:
    rows = _tuning_rows()
    for row in rows:
        if row["model_name"] in {"gru", "flat_mlp"}:
            row["validation_step_catastrophic_auprc"] = {
                3e-4: 0.50000,
                1e-3: 0.50009,
                3e-3: 0.49,
            }[row["learning_rate"]]
    _, selection = select_learning_rates(rows)
    assert selection["models"]["gru"]["selected_learning_rate"] == 3e-4
    assert learning_rate_compatibility_gate(selection)["status"] == "G5_BLOCKED_LR_COMPATIBILITY"

    for row in rows:
        if row["model_name"] in {"gru", "flat_mlp"}:
            row["validation_step_catastrophic_auprc"] = {
                3e-4: 0.49,
                1e-3: 0.51,
                3e-3: 0.50,
            }[row["learning_rate"]]
    _, selection = select_learning_rates(rows)
    assert learning_rate_compatibility_gate(selection)["status"] == "PASS"


def _stat_rows() -> list[dict]:
    rows = []
    for task_index, task in enumerate(_tasks()):
        prevalence = 0.1 + task_index * 1e-5
        for model_index, model in enumerate(FINAL_MODELS):
            rows.append(
                {
                    "condition_id": "canonical_full",
                    "task_name": task,
                    "model_name": model,
                    "source_platform": "synthetic",
                    "step_auprc": 0.3 + model_index * 0.03 + task_index * 1e-4,
                    "matched_step_prevalence": prevalence,
                }
            )
            rows.append(
                {
                    "condition_id": "canonical_current_only",
                    "task_name": task,
                    "model_name": model,
                    "source_platform": "synthetic",
                    "step_auprc": 0.2 + model_index * 0.03 + task_index * 1e-4,
                    "matched_step_prevalence": prevalence,
                }
            )
    return rows


def test_f1_f2_f3_task_statistics_are_deterministic_and_frozen() -> None:
    paired_a, stats_a = build_frozen_statistics(
        _stat_rows(), bootstrap_rounds=100, permutation_rounds=1000
    )
    paired_b, stats_b = build_frozen_statistics(
        _stat_rows(), bootstrap_rounds=100, permutation_rounds=1000
    )
    assert paired_a == paired_b
    assert stats_a == stats_b
    assert len(paired_a) == 38 * 8
    assert [len([row for row in stats_a if row["family"] == family]) for family in (
        "F1_canonical_learnability", "F2_canonical_history_increment", "F3_modern_baseline"
    )] == [3, 3, 2]
    assert all(row["n"] == 38 for row in stats_a)


def test_stage5_runner_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "ctsc/scripts/run_revision_stage5.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--phase" in completed.stdout
    assert "--resume" in completed.stdout
