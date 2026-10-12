from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from ctsc.precursor.actionability import choose_threshold_for_alarm_budget, summarize_episode_alarm_utility


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_choose_threshold_prefers_more_conservative_tie() -> None:
    scores = [0.90, 0.80, 0.20, 0.10]
    result = choose_threshold_for_alarm_budget(scores, budget=0.375)

    assert result.alarm_rate == 0.25
    assert result.threshold > 0.8


def test_episode_alarm_summary_counts_only_recalled_catastrophic_lead_time() -> None:
    summary = summarize_episode_alarm_utility(
        scores=[0.8, 0.1, 0.9, 0.2],
        threshold=0.5,
        episode_ids=[1, 1, 2, 2],
        outcome_labels=[2, 2, 0, 0],
        steps_remaining=[5, 4, 3, 2],
    )

    assert summary.step_alarm_rate == 0.5
    assert summary.episode_alarm_rate == 1.0
    assert summary.catastrophic_episode_recall == 1.0
    assert summary.noncat_episode_false_alarm_rate == 1.0
    assert summary.episode_alarm_precision == 0.5
    assert summary.recalled_catastrophic_lead_time_median == 5.0


def test_run_actionability_offline_utility_single_task_smoke(
    mock_public_dataset_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "actionability_registry.json"
    output_dir = tmp_path / "actionability_offline_utility"
    precursor_source_dir = tmp_path / "precursor_source"
    current_run_root = tmp_path / "current_run"
    task_run_root = current_run_root / "task_runs"
    precursor_source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(mock_public_dataset_files["precursor_path"], precursor_source_dir / "AntRun.pt")

    suite_output_dir = task_run_root / "AntRun" / "canonical" / "seed_0"
    suite_completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/run_precursor_baseline_suite.py",
            "--dataset",
            str(mock_public_dataset_files["precursor_path"]),
            "--output-dir",
            str(suite_output_dir),
            "--slice-name",
            "canonical",
            "--device",
            "cpu",
            "--models",
            "gru",
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
    assert suite_completed.returncode == 0, suite_completed.stderr

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
            "ctsc/scripts/run_Unified38_actionability_offline_utility.py",
            "--registry",
            str(registry_path),
            "--task-names",
            "AntRun",
            "--split-seeds",
            "0",
            "--task-run-root",
            str(task_run_root),
            "--precursor-source-dir",
            str(precursor_source_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--models",
            "gru",
            "prevalence",
            "critic_score",
            "--budgets",
            "0.01",
            "0.05",
            "--batch-size",
            "8",
            "--concurrency",
            "1",
            "--skip-existing",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    aggregate_summary = json.loads(
        (output_dir / "aggregates" / "Unified38_actionability_offline_utility_summary.json").read_text(encoding="utf-8")
    )
    analysis_summary = json.loads(
        (
            output_dir / "analysis" / "Unified38_actionability_offline_utility_analysis_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert aggregate_summary["transition_row_count"] == 3
    assert aggregate_summary["budget_row_count"] == 6
    assert aggregate_summary["task_pair_row_count"] == 6
    assert aggregate_summary["coverage_fail_count"] == 0
    assert analysis_summary["coverage"]["coverage_fail_count"] == 0
