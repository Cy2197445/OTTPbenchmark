from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from ctsc.precursor.ablation import (
    ABLATION_GROUP_MODE_REGISTRY,
    apply_ablation_train_mask,
    resolve_unified38_ablation_spec,
)
from ctsc.precursor.data import build_splits, load_precursor_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_terminal_only_mask_filters_train_val_but_not_test(mock_public_dataset_files: dict[str, Path]) -> None:
    dataset = load_precursor_dataset(mock_public_dataset_files["precursor_path"])
    splits = build_splits(dataset, seed=0, split_family="episode_group")
    spec = resolve_unified38_ablation_spec("task_framing", "episode_outcome_terminal_only")

    filtered, audit = apply_ablation_train_mask(dataset, splits, spec=spec)

    assert audit["train_mask_mode"] == "terminal_only"
    assert filtered["train"].shape[0] < splits["train"].shape[0]
    assert filtered["val"].shape[0] <= splits["val"].shape[0]
    assert filtered["test"].shape[0] == splits["test"].shape[0]


def test_train_precursor_model_supports_binary_unsafe_vs_safe_ablation(
    mock_public_dataset_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "binary_unsafe_vs_safe"
    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/train_precursor_model.py",
            "--dataset",
            str(mock_public_dataset_files["precursor_path"]),
            "--output-dir",
            str(output_dir),
            "--model",
            "gru",
            "--ablation-group",
            "labels",
            "--ablation-mode",
            "binary_unsafe_vs_safe",
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
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["ablation_group"] == "labels"
    assert metrics["ablation_mode"] == "binary_unsafe_vs_safe"
    assert metrics["selection_target_type"] == "step_unsafe_vs_safe"
    assert metrics["primary_eval_target"] == "canonical_step_catastrophic"
    assert metrics["num_classes"] == 2
    assert metrics["training"]["best_epoch"] >= 0


def test_suite_runner_terminal_only_emits_ablation_metadata(
    mock_public_dataset_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "terminal_only_suite"
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
            "--ablation-group",
            "task_framing",
            "--ablation-mode",
            "episode_outcome_terminal_only",
            "--device",
            "cpu",
            "--models",
            "gru",
            "flat-mlp",
            "prevalence",
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
    assert summary["ablation_group"] == "task_framing"
    assert summary["ablation_mode"] == "episode_outcome_terminal_only"
    assert summary["train_mask_mode"] == "terminal_only"
    assert summary["effective_split_sizes"]["train"] < summary["split_filter_audit"]["split_filtering"]["train"]["original_count"]
    assert summary["effective_split_sizes"]["test"] == summary["split_filter_audit"]["split_filtering"]["test"]["original_count"]


def test_run_target_definition_ablation_single_task_smoke(
    mock_public_dataset_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "target_definition_registry.json"
    output_dir = tmp_path / "target_definition_ablation"
    precursor_source_dir = tmp_path / "precursor_source"
    precursor_source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(mock_public_dataset_files["precursor_path"], precursor_source_dir / "AntRun.pt")

    registry = {
        "title": "Unified38 Fresh Rerun Registry",
        "tasks": [
            {
                "task_name": "AntRun",
                "dataset_path": str(mock_public_dataset_files["raw_path"]),
                "dataset_exists_locally": True,
            }
        ],
        "mainline_fresh_rerun": {
            "split_seeds": [0],
            "slices": [
                {
                    "name": "canonical",
                    "step_q_low": 0.75,
                    "step_q_high": 0.90,
                    "outcome_q_low": 0.75,
                    "outcome_q_high": 0.90,
                }
            ],
            "shared_defaults": {
                "feature_profile": "full",
                "window_size": 32,
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "cnn_kernel_size": 3,
                "mlp_depth": 2,
                "batch_size": 8,
                "epochs": 1,
                "lr": 1e-3,
                "weight_decay": 1e-5,
                "max_train_samples": 18,
                "max_val_samples": 6,
                "max_test_samples": 6,
            },
        },
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/run_Unified38_target_definition_ablation.py",
            "--registry",
            str(registry_path),
            "--tasks",
            "AntRun",
            "--split-seeds",
            "0",
            "--precursor-source-dir",
            str(precursor_source_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--initial-concurrency",
            "2",
            "--max-concurrency",
            "2",
            "--min-concurrency",
            "2",
            "--warmup-jobs",
            "2",
            "--skip-existing",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    aggregate_summary = json.loads(
        (output_dir / "aggregates" / "Unified38_target_definition_ablation_summary.json").read_text(encoding="utf-8")
    )
    analysis_summary = json.loads(
        (output_dir / "analysis" / "Unified38_target_definition_ablation_analysis_summary.json").read_text(encoding="utf-8")
    )

    assert aggregate_summary["suite_row_count"] == sum(len(modes) for modes in ABLATION_GROUP_MODE_REGISTRY.values())
    assert aggregate_summary["seed_model_row_count"] == 18
    assert aggregate_summary["task_framing_paired_task_model_row_count"] == 9
    assert aggregate_summary["label_family_paired_task_model_row_count"] == 6
    assert analysis_summary["coverage"]["coverage_fail_count"] == 0
