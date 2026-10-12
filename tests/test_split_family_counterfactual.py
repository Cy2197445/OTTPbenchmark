from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _suite_summary(*, split_family: str, step_offset: float) -> dict[str, object]:
    return {
        "slice_name": "canonical",
        "seed": 0,
        "split_family": split_family,
        "feature_profile": "full",
        "split_manifest": {"split_family": split_family},
        "split_diagnostics": {
            "group_overlap": {
                "overlapping_group_count": 2 if split_family == "transition_random" else 0,
                "overlapping_group_ratio": 0.5 if split_family == "transition_random" else 0.0,
                "group_leakage_detected": split_family == "transition_random",
            },
            "episode_fragmentation": {
                "fragmented_episode_count": 3 if split_family == "transition_random" else 0,
                "fragmented_episode_ratio": 0.5 if split_family == "transition_random" else 0.0,
                "fragmentation_detected": split_family == "transition_random",
            },
        },
        "models": {
            "gru": {
                "test_step_catastrophic_auprc": 0.60 + step_offset,
                "test_outcome_catastrophic_auprc": 0.48 + step_offset,
                "test_catastrophic_ece": 0.05,
                "test_lead_time_median_at_0.5": 4.0,
            },
            "flat-mlp": {
                "test_step_catastrophic_auprc": 0.50 + step_offset,
                "test_outcome_catastrophic_auprc": 0.40 + step_offset,
                "test_catastrophic_ece": 0.07,
                "test_lead_time_median_at_0.5": 3.0,
            },
            "prevalence": {
                "test_step_catastrophic_auprc": 0.30 + step_offset,
                "test_outcome_catastrophic_auprc": 0.20 + step_offset,
                "test_catastrophic_ece": 0.02,
                "test_lead_time_median_at_0.5": 0.0,
            },
        },
        "best_sequence_baseline": "gru",
        "best_sequence_test_step_auprc": 0.60 + step_offset,
        "gap_vs_flat": 0.10,
        "gap_vs_linear": 0.20,
        "strict_task_level_sequence_support": True,
    }


def test_aggregate_split_family_counterfactual_pairs_deltas(tmp_path: Path) -> None:
    input_root = tmp_path / "task_runs"
    output_dir = tmp_path / "aggregates"
    for split_family, step_offset in (("episode_group", 0.0), ("transition_random", 0.1)):
        summary_path = input_root / "AntRun" / "canonical" / "seed_0" / split_family / "suite_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_suite_summary(split_family=split_family, step_offset=step_offset), indent=2), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/aggregate_Unified38_split_family_counterfactual.py",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--task-names",
            "AntRun",
            "--split-seeds",
            "0",
            "--slices",
            "canonical",
            "--split-families",
            "episode_group",
            "transition_random",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    pair_rows = (output_dir / "unified_split_family_task_pairs.csv").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "Unified38_split_family_counterfactual_summary.json").read_text(encoding="utf-8"))

    assert "AntRun" in pair_rows
    assert "0.09999999999999998" in pair_rows or "0.1" in pair_rows
    assert summary["paired_task_model_row_count"] == 3


def test_run_split_family_counterfactual_single_task_smoke(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    registry_path = tmp_path / "split_family_registry.json"
    output_dir = tmp_path / "split_family_counterfactual"
    registry = {
        "title": "Unified38 Fresh Rerun Registry",
        "tasks": [
            {
                "task_name": "AntRun",
                "dataset_path": str(mock_public_dataset_files["raw_path"]),
                "dataset_exists_locally": True,
            }
        ],
        "baseline_zoo": {
            "models": ["gru", "temporal-cnn", "flat-mlp", "linear", "prevalence"],
        },
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
            "ctsc/scripts/run_Unified38_split_family_counterfactual.py",
            "--registry",
            str(registry_path),
            "--tasks",
            "AntRun",
            "--split-seeds",
            "0",
            "--split-families",
            "episode_group",
            "transition_random",
            "--models",
            "gru",
            "flat-mlp",
            "prevalence",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "8",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    suite_summary = json.loads(
        (
            output_dir
            / "task_runs"
            / "AntRun"
            / "canonical"
            / "seed_0"
            / "transition_random"
            / "suite_summary.json"
        ).read_text(encoding="utf-8")
    )
    aggregate_summary = json.loads(
        (output_dir / "aggregates" / "Unified38_split_family_counterfactual_summary.json").read_text(encoding="utf-8")
    )
    analysis_summary = json.loads(
        (output_dir / "analysis" / "Unified38_split_family_counterfactual_analysis_summary.json").read_text(encoding="utf-8")
    )

    assert suite_summary["split_family"] == "transition_random"
    assert suite_summary["split_diagnostics"]["episode_fragmentation"]["fragmented_episode_count"] > 0
    assert aggregate_summary["paired_task_model_row_count"] == 3
    assert analysis_summary["coverage"]["paired_task_model_row_count"] == 3
