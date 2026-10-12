from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ctsc.precursor.model import (
    FlatMLPPrecursorModel,
    GRUPrecursorModel,
    LinearPrecursorModel,
    PrevalenceBaseline,
    TemporalCNNPrecursorModel,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_MODELS = ["gru", "temporal-cnn", "flat-mlp", "linear", "prevalence"]
CURRENT_ONLY_MODELS = ["gru", "temporal-cnn", "flat-mlp", "linear"]


def test_model_forward_shapes() -> None:
    batch_size = 4
    window_size = 8
    feature_dim = 5
    sequence = torch.randn(batch_size, window_size, feature_dim)
    flat_features = torch.randn(batch_size, window_size * feature_dim)

    assert GRUPrecursorModel(input_dim=feature_dim, hidden_dim=16)(sequence).shape == (batch_size, 3)
    assert TemporalCNNPrecursorModel(input_dim=feature_dim, hidden_dim=16, kernel_size=3)(sequence).shape == (batch_size, 3)
    assert FlatMLPPrecursorModel(input_dim=window_size * feature_dim, hidden_dim=16, mlp_depth=2)(flat_features).shape == (batch_size, 3)
    assert LinearPrecursorModel(input_dim=window_size * feature_dim)(flat_features).shape == (batch_size, 3)


def test_prevalence_baseline_outputs_valid_probabilities() -> None:
    baseline = PrevalenceBaseline(num_classes=3)
    probs = baseline.fit(torch.as_tensor([0, 0, 1, 2, 2, 2], dtype=torch.long))
    pred = baseline.predict_proba(batch_size=5)

    assert probs.shape == (3,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    assert pred.shape == (5, 3)
    assert torch.allclose(pred.sum(dim=1), torch.ones(5))


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_train_precursor_model_supports_all_baselines(mock_public_dataset_files: dict[str, Path], tmp_path: Path, model_name: str) -> None:
    output_dir = tmp_path / model_name.replace("-", "_")
    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/train_precursor_model.py",
            "--dataset",
            str(mock_public_dataset_files["precursor_path"]),
            "--output-dir",
            str(output_dir),
            "--model",
            model_name,
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-dim",
            "16",
            "--max-train-samples",
            "18",
            "--max-val-samples",
            "6",
            "--max-test-samples",
            "6",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    metrics_path = output_dir / "metrics.json"
    bundle_path = output_dir / "precursor_model.pt"
    assert metrics_path.exists()
    assert bundle_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["model_name"] == model_name
    assert metrics["feature_profile"] == "full"
    assert metrics["split_family"] == "episode_group"
    assert "label_step" in metrics
    assert "label_outcome" in metrics
    assert "critic_baseline" in metrics
    assert "training" in metrics
    if model_name == "prevalence":
        assert metrics["training"]["best_epoch"] == -1
        assert metrics["training"]["history"] == []
        assert metrics["training"]["temperature"] == pytest.approx(1.0)
    else:
        assert metrics["training"]["best_epoch"] >= 0


def test_suite_runner_writes_suite_summary(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    output_dir = tmp_path / "suite"
    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/run_precursor_baseline_suite.py",
            "--dataset",
            str(mock_public_dataset_files["precursor_path"]),
            "--output-dir",
            str(output_dir),
            "--slice-name",
            "canonical",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-dim",
            "16",
            "--max-train-samples",
            "18",
            "--max-val-samples",
            "6",
            "--max-test-samples",
            "6",
            "--step-q-low",
            "0.75",
            "--step-q-high",
            "0.90",
            "--outcome-q-low",
            "0.75",
            "--outcome-q-high",
            "0.90",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert summary["slice_name"] == "canonical"
    assert summary["feature_profile"] == "full"
    assert set(summary["models"].keys()) == set(ALL_MODELS)
    assert summary["best_sequence_baseline"] in {"gru", "temporal-cnn"}
    assert "gap_vs_flat" in summary
    assert "gap_vs_linear" in summary
    assert isinstance(summary["strict_task_level_sequence_support"], bool)
    assert summary["label_audit"]["threshold_fit_scope"] == "capped_train_only"
    assert summary["label_audit"]["normalization_fit_scope"] == "capped_train_only"
    assert set(summary["label_audit"]["step_class_distribution"]) == {"train", "val", "test"}
    assert summary["label_audit"]["step_class_distribution"]["train"]["sample_count"] == 18
    assert summary["label_sidecar"] is None


def test_suite_runner_supports_current_only_four_model_matrix(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    output_dir = tmp_path / "current_only_suite"
    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/run_precursor_baseline_suite.py",
            "--dataset",
            str(mock_public_dataset_files["precursor_path"]),
            "--output-dir",
            str(output_dir),
            "--slice-name",
            "canonical",
            "--device",
            "cpu",
            "--feature-profile",
            "current_only",
            "--models",
            *CURRENT_ONLY_MODELS,
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-dim",
            "16",
            "--max-train-samples",
            "18",
            "--max-val-samples",
            "6",
            "--max-test-samples",
            "6",
            "--step-q-low",
            "0.75",
            "--step-q-high",
            "0.90",
            "--outcome-q-low",
            "0.75",
            "--outcome-q-high",
            "0.90",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert summary["feature_profile"] == "current_only"
    assert set(summary["models"].keys()) == set(CURRENT_ONLY_MODELS)
    for model_name in CURRENT_ONLY_MODELS:
        metrics_path = output_dir / model_name.replace("-", "_") / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["feature_profile"] == "current_only"


def test_suite_runner_emits_transition_random_diagnostics(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    output_dir = tmp_path / "transition_random_suite"
    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/run_precursor_baseline_suite.py",
            "--dataset",
            str(mock_public_dataset_files["precursor_path"]),
            "--output-dir",
            str(output_dir),
            "--slice-name",
            "canonical",
            "--split-family",
            "transition_random",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-dim",
            "16",
            "--max-train-samples",
            "18",
            "--max-val-samples",
            "6",
            "--max-test-samples",
            "6",
            "--step-q-low",
            "0.75",
            "--step-q-high",
            "0.90",
            "--outcome-q-low",
            "0.75",
            "--outcome-q-high",
            "0.90",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert summary["split_family"] == "transition_random"
    assert summary["split_diagnostics"]["group_overlap"]["group_leakage_detected"] is True
    assert summary["split_diagnostics"]["episode_fragmentation"]["fragmented_episode_count"] > 0


