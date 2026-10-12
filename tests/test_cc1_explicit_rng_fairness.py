from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ctsc.precursor.cc1 import (
    BRIDGE_MODELS,
    FAIR_MODELS,
    LOCKED_MODEL_LRS,
    bridge_cells,
    fair_cells,
)
from ctsc.scripts.run_revision_cc1_cell import _is_cuda_oom
from ctsc.scripts.run_revision_cc1_matrix import _cell_complete
from ctsc.scripts.analyze_revision_stage5_fair_matrix import (
    _fixed_main_table,
    _task_rows,
    _theoretical_macs,
)


def _tasks() -> list[str]:
    return [f"Task{index:02d}" for index in range(38)]


def test_cc1_frozen_matrix_counts_and_model_specific_learning_rates() -> None:
    bridge = bridge_cells(_tasks())
    fair = fair_cells(_tasks())
    assert len(bridge) == 114
    assert sum(len(cell["models"]) for cell in bridge) == 228
    assert len(fair) == 342
    assert sum(len(cell["models"]) for cell in fair) == 1026
    assert set(bridge[0]["models"]) == set(BRIDGE_MODELS)
    assert set(fair[0]["models"]) == set(FAIR_MODELS)
    assert fair[0]["model_learning_rates"] == LOCKED_MODEL_LRS
    assert {cell["condition_id"] for cell in fair} == {
        "canonical_full",
        "sensitivity_full",
        "canonical_current_only",
    }


def test_cuda_oom_detection_is_restricted_to_explicit_cuda_ooms() -> None:
    assert _is_cuda_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert _is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert not _is_cuda_oom(RuntimeError("CPU allocator ran out of memory"))
    assert not _is_cuda_oom(RuntimeError("unrelated failure"))


def test_cc1_cell_shares_preprocessing_and_resumes_models_independently(
    mock_public_dataset_files: dict[str, Path], tmp_path: Path
) -> None:
    cell_dir = tmp_path / "cell_runs" / "canonical_full" / "AntRun" / "split_seed_0"
    command = [
        sys.executable,
        "ctsc/scripts/run_revision_cc1_cell.py",
        "--dataset",
        str(mock_public_dataset_files["precursor_path"]),
        "--cell-dir",
        str(cell_dir),
        "--task-name",
        "AntRun",
        "--condition-id",
        "canonical_full",
        "--feature-profile",
        "full",
        "--q-low",
        "0.75",
        "--q-high",
        "0.9",
        "--split-seed",
        "0",
        "--model-lr",
        "gru=0.001",
        "--model-lr",
        "flat_mlp=0.003",
        "--matrix",
        "stage5_fair_matrix",
        "--device",
        "cpu",
        "--window-size",
        "4",
        "--hidden-dim",
        "8",
        "--batch-size",
        "8",
        "--epochs",
        "1",
        "--max-train-samples",
        "20",
        "--max-val-samples",
        "10",
        "--minimum-free-space-gib",
        "0",
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr + "\n" + first.stdout
    descriptor = json.loads((cell_dir / "cell_manifest.json").read_text(encoding="utf-8"))
    cell = {
        "task_name": "AntRun",
        "condition_id": "canonical_full",
        "split_seed": 0,
        "models": ["gru", "flat_mlp"],
        "model_learning_rates": {"gru": 0.001, "flat_mlp": 0.003},
    }
    assert descriptor["seed_protocol"] == "explicit_four_stream"
    assert _cell_complete(tmp_path, {**cell, "cell_id": "test"})
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((cell_dir / "models").glob("*/attempt-*/run_manifest.json"))
    ]
    assert len(manifests) == 2
    assert manifests[0]["indices"] == manifests[1]["indices"]
    assert all(manifest["training"]["loader_rng_mode"] == "explicit_loader_seed" for manifest in manifests)
    assert all(
        [manifest["protocol"][field] for field in ("split_seed", "sampling_seed", "train_seed", "loader_seed")]
        == [0, 0, 0, 0]
        for manifest in manifests
    )
    attempts_before = sorted((cell_dir / "models").glob("*/attempt-*"))
    shared_hash_before = (cell_dir / "reproducibility_record.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr + "\n" + second.stdout
    assert sorted((cell_dir / "models").glob("*/attempt-*")) == attempts_before
    assert (cell_dir / "reproducibility_record.json").read_bytes() == shared_hash_before
    resumed = json.loads((cell_dir / "cell_summary.json").read_text(encoding="utf-8"))
    assert {row["status"] for row in resumed["results"]} == {"SKIPPED_VALID"}


def test_cc1_runner_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "ctsc/scripts/run_revision_cc1_matrix.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset-dir" in completed.stdout
    assert "--conditions" in completed.stdout
    assert "--prepare-only" in completed.stdout


def test_cc1_task_aggregation_and_fixed_table_never_select_dynamic_best() -> None:
    seed_rows = []
    for task_index, task in enumerate(_tasks()):
        for condition in ("canonical_full", "sensitivity_full", "canonical_current_only"):
            for model_index, model in enumerate(FAIR_MODELS):
                for seed in (0, 1, 2):
                    prevalence = 0.1 + task_index * 1e-6
                    value = 0.2 + model_index * 0.02 + seed * 0.001
                    seed_rows.append(
                        {
                            "condition_id": condition,
                            "task_name": task,
                            "model_name": model,
                            "source_platform": "synthetic",
                            "split_seed": seed,
                            "step_auprc": value,
                            "matched_step_prevalence": prevalence,
                            "step_auprc_minus_prevalence": value - prevalence,
                            "outcome_auprc": value / 2,
                            "matched_outcome_prevalence": prevalence / 2,
                            "step_ece": 0.05,
                            "step_lead_median": 4.0,
                        }
                    )
    task_rows = _task_rows(seed_rows)
    fixed = _fixed_main_table(task_rows, _tasks())
    assert len(task_rows) == 342
    assert len(fixed) == 38
    assert set(fixed[0]) >= {
        "gru_step_auprc",
        "flat_mlp_step_auprc",
        "transformer_step_auprc",
        "matched_step_prevalence",
    }
    assert not any("best" in field for field in fixed[0])
    assert fixed[0]["gru_step_auprc"] == pytest.approx(0.201)


def test_transformer_complexity_uses_frozen_formula() -> None:
    config = {
        "window_size": 32,
        "num_classes": 3,
        "input_dim": 25,
        "transformer_d_model": 52,
        "transformer_num_layers": 2,
    }
    observed, formula, feature_dim = _theoretical_macs("transformer", config)
    expected = 32 * 25 * 52 + 2 * (12 * 32 * 52**2 + 2 * 32**2 * 52) + 52**2 + 52 * 3
    assert observed == expected
    assert feature_dim == 25
    assert formula == "W*D*M + L*(12*W*M^2 + 2*W^2*M) + M^2 + M*C"
