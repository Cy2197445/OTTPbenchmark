from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepare_precursor_dir(tmp_path: Path, precursor_path: Path) -> Path:
    precursor_dir = tmp_path / "precursors"
    precursor_dir.mkdir()
    shutil.copy2(precursor_path, precursor_dir / "AntRun.pt")
    return precursor_dir


def test_run_gru_profile_matrix_and_analyze(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    precursor_dir = _prepare_precursor_dir(tmp_path, mock_public_dataset_files["precursor_path"])
    run_root = tmp_path / "profile_matrix"
    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/run_Unified38_gru_profile_matrix.py",
            "--tasks",
            "AntRun",
            "--split-seeds",
            "0",
            "--precursor-source-dir",
            str(precursor_dir),
            "--output-dir",
            str(run_root),
            "--device",
            "cpu",
            "--concurrency",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--max-train-samples",
            "12",
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
    summary_path = run_root / "tasks" / "AntRun" / "seed_0" / "profile_matrix_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task_name"] == "AntRun"
    assert set(summary["profiles"]) >= {
        "full",
        "current_only",
        "cost_step_only",
        "history_without_current_cost",
        "history_obs_action_step_only",
        "full_shuffled",
        "lagged_cost_step",
        "current_without_cost",
    }
    assert summary["profiles"]["current_only"]["feature_profile"] == "current_only"

    analyze = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/analyze_Unified38_gru_profile_matrix.py",
            "--input-root",
            str(run_root),
            "--output-dir",
            str(run_root / "analysis"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert analyze.returncode == 0, analyze.stderr
    anti_summary = json.loads((run_root / "analysis" / "Unified38_canonical_anti_shortcut_seed0_summary.json").read_text(encoding="utf-8"))
    extended_summary = json.loads((run_root / "analysis" / "Unified38_canonical_extended_profiles_seed0_summary.json").read_text(encoding="utf-8"))
    assert anti_summary["coverage"]["task_count"] == 1
    assert extended_summary["coverage"]["task_count"] == 1


def test_analyze_gru_profile_matrix_requires_core_profiles(tmp_path: Path) -> None:
    input_root = tmp_path / "profile_matrix"
    summary_path = input_root / "tasks" / "AntRun" / "seed_0" / "profile_matrix_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "task_name": "AntRun",
                "source_platform": "bullet_safety_gym_suite",
                "split_seed": 0,
                "profiles": {
                    "full": {"test_step_catastrophic_auprc": 0.7, "test_outcome_catastrophic_auprc": 0.6},
                    "current_only": {"test_step_catastrophic_auprc": 0.6, "test_outcome_catastrophic_auprc": 0.5},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/analyze_Unified38_gru_profile_matrix.py",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "core 5 profiles" in completed.stderr


def test_run_gru_window_sweep_and_analyze(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    precursor_dir = _prepare_precursor_dir(tmp_path, mock_public_dataset_files["precursor_path"])
    run_root = tmp_path / "window_sweep"
    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/run_Unified38_gru_window_sweep.py",
            "--tasks",
            "AntRun",
            "--split-seeds",
            "0",
            "--window-sizes",
            "1",
            "4",
            "8",
            "16",
            "32",
            "--precursor-source-dir",
            str(precursor_dir),
            "--output-dir",
            str(run_root),
            "--device",
            "cpu",
            "--concurrency",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--max-train-samples",
            "12",
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
    summary_path = run_root / "tasks" / "AntRun" / "seed_0" / "window_sweep_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task_name"] == "AntRun"
    assert summary["windows"]["1"]["window_size"] == 1
    assert summary["windows"]["32"]["window_size"] == 32

    analyze = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/analyze_Unified38_gru_window_sweep.py",
            "--input-root",
            str(run_root),
            "--output-dir",
            str(run_root / "analysis"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert analyze.returncode == 0, analyze.stderr
    summary_json = json.loads((run_root / "analysis" / "Unified38_gru_window_sweep_summary.json").read_text(encoding="utf-8"))
    assert summary_json["coverage"]["task_count"] == 1


def test_analyze_gru_window_sweep_requires_full_window_set(tmp_path: Path) -> None:
    input_root = tmp_path / "window_sweep"
    summary_path = input_root / "tasks" / "AntRun" / "seed_0" / "window_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "task_name": "AntRun",
                "source_platform": "bullet_safety_gym_suite",
                "split_seed": 0,
                "windows": {
                    "1": {"test_step_catastrophic_auprc": 0.5},
                    "4": {"test_step_catastrophic_auprc": 0.6},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "ctsc/scripts/analyze_Unified38_gru_window_sweep.py",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "requires windows" in completed.stderr
