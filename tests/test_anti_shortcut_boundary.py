from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_anti_shortcut_task_runner_writes_boundary_summary(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    output_dir = tmp_path / "boundary_task"
    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/run_Unified38_anti_shortcut_task.py",
            "--dataset",
            str(mock_public_dataset_files["precursor_path"]),
            "--task-name",
            "AntRun",
            "--output-dir",
            str(output_dir),
            "--split-seed",
            "0",
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
    summary = json.loads((output_dir / "boundary_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert summary["task_name"] == "AntRun"
    assert summary["split_seed"] == 0
    assert set(summary["profiles"].keys()) == {
        "current_only",
        "cost_step_only",
        "history_without_current_cost",
        "history_obs_action_step_only",
    }
    assert summary["full_reference"]["feature_profile"] == "full"
    assert isinstance(summary["boundary_tag"], str)
    assert "full_minus_current_only_step_auprc" in summary["contrasts"]


def test_anti_shortcut_aggregate_builds_summary(tmp_path: Path) -> None:
    root = tmp_path / "boundary_runs"
    seed_dir = root / "tasks" / "AntRun" / "seed_0"
    seed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "Unified38_current_boundary_task",
        "status": "success",
        "task_name": "AntRun",
        "split_seed": 0,
        "source_platform": "bullet_safety_gym_suite",
        "boundary_tag": "cost-heavy but not current-only",
        "full_reference": {"step_catastrophic_auprc": 0.65},
        "profiles": {
            "current_only": {"step_catastrophic_auprc": 0.60},
            "cost_step_only": {"step_catastrophic_auprc": 0.55},
            "history_without_current_cost": {"step_catastrophic_auprc": 0.62},
            "history_obs_action_step_only": {"step_catastrophic_auprc": 0.58},
        },
    }
    (seed_dir / "boundary_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/aggregate_Unified38_anti_shortcut.py",
            "--input-root",
            str(root),
            "--output-dir",
            str(root),
            "--task-names",
            "AntRun",
            "--split-seeds",
            "0",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((root / "Unified38_boundary_aggregate.json").read_text(encoding="utf-8"))
    assert summary["phase"] == "Unified38_current_boundary"
    assert summary["task_count"] == 1
    assert summary["completed_task_count"] == 1
    assert summary["completed_task_seed_count"] == 1
    assert summary["expected_task_seed_count"] == 1
    assert summary["task_seed_coverage_status"] == "pass"
    assert summary["pooled_all_seeds"]["full_gt_current_only_count"] == 1
    assert summary["tasks"][0]["task_name"] == "AntRun"
