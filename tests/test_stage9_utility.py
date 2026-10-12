from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from ctsc.precursor.actionability import choose_threshold_for_alarm_budget
from ctsc.precursor.data import FeatureMatrixSequenceDataset
from ctsc.precursor.revision import file_sha256
from ctsc.precursor.metrics import binary_average_precision
from ctsc.precursor.stage9 import episode_alarm_utility, evaluate_budget, full_prefix_windows, select_validation_threshold
from ctsc.scripts.analyze_revision_stage9_utility import (
    MEASURES,
    PLATFORM_TO_ENVIRONMENT,
    _paired,
    _summaries,
    _validate_result,
)
from ctsc.scripts.run_revision_stage9_utility import _model_from_bundle, _predict_split, _predict_split_fast
from ctsc.scripts.train_precursor_model import prepare_precursor_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_vectorized_full_prefix_matches_frozen_dataset_view() -> None:
    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    uids = np.asarray([1, 1, 1, 2, 2, 2])
    positions = np.asarray([0, 1, 2, 0, 1, 2])
    indices = np.asarray([0, 1, 2, 3, 4, 5])
    dataset = {
        "episode_uid": uids, "episode_position": positions,
        "critic_score": np.zeros(6), "env_index": np.zeros(6), "steps_remaining": np.asarray([2, 1, 0, 2, 1, 0]),
    }
    reference = FeatureMatrixSequenceDataset(features, dataset, indices, np.zeros(6), window_size=4)
    windows = full_prefix_windows(features, indices, positions, uids, window_size=4)
    for index in range(6):
        np.testing.assert_array_equal(windows[index], reference[index]["sequence"].numpy())


@pytest.mark.parametrize("budget", [0.01, 0.05, 0.10, 0.375])
def test_fast_budget_selection_matches_frozen_tie_rule(budget: float) -> None:
    scores = np.asarray([0.9, 0.9, 0.8, 0.8, 0.2, 0.1, 0.1, 0.1])
    old = choose_threshold_for_alarm_budget(scores, budget)
    new = select_validation_threshold(scores, budget)
    assert new.threshold == old.threshold
    assert new.validation_step_alarm_rate == old.alarm_rate
    assert new.candidate_count == old.candidate_count


def test_episode_metrics_are_counted_once_and_first_alarm_is_earliest_step() -> None:
    summary = episode_alarm_utility(
        scores=np.asarray([0.8, 0.1, 0.9, 0.2]),
        threshold=0.5,
        episode_ids=np.asarray([7, 7, 8, 8]),
        outcome_labels=np.asarray([2, 2, 0, 0]),
        steps_remaining=np.asarray([5, 4, 3, 2]),
    )
    assert summary["test_step_alarm_count"] == 2
    assert summary["test_episode_count"] == 2
    assert summary["catastrophic_episode_recall"] == 1.0
    assert summary["episode_alarm_precision"] == 0.5
    assert summary["noncat_episode_false_alarm_rate"] == 1.0
    assert summary["recalled_first_alarm_steps_to_episode_end_median"] == 5.0

    permuted = episode_alarm_utility(
        scores=np.asarray([0.2, 0.1, 0.9, 0.8]),
        threshold=0.5,
        episode_ids=np.asarray([8, 7, 8, 7]),
        outcome_labels=np.asarray([0, 2, 0, 2]),
        steps_remaining=np.asarray([2, 4, 3, 5]),
    )
    assert permuted == summary


def test_undefined_rates_are_missing_not_zero() -> None:
    result = episode_alarm_utility(
        scores=np.asarray([0.1, 0.2]),
        threshold=0.5,
        episode_ids=np.asarray([1, 1]),
        outcome_labels=np.asarray([0, 0]),
        steps_remaining=np.asarray([1, 0]),
    )
    assert result["catastrophic_episode_recall"] is None
    assert result["episode_alarm_precision"] is None
    assert result["recalled_first_alarm_steps_to_episode_end_median"] is None
    assert result["noncat_episode_false_alarm_rate"] == 0.0


def test_test_changes_cannot_change_validation_threshold() -> None:
    kwargs = {
        "budget": 0.05,
        "test_episode_ids": np.asarray([1, 1]),
        "test_outcome_labels": np.asarray([2, 2]),
        "test_steps_remaining": np.asarray([1, 0]),
    }
    first = evaluate_budget(np.asarray([0.1, 0.3, 0.8]), test_scores=np.asarray([0.2, 0.9]), **kwargs)
    second = evaluate_budget(np.asarray([0.1, 0.3, 0.8]), test_scores=np.asarray([0.99, 0.01]), **kwargs)
    assert first["threshold"] == second["threshold"]
    assert first["validation_step_alarm_rate"] == second["validation_step_alarm_rate"]


def test_inconsistent_episode_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="not constant"):
        episode_alarm_utility(
            scores=np.asarray([0.2, 0.8]),
            threshold=0.5,
            episode_ids=np.asarray([1, 1]),
            outcome_labels=np.asarray([0, 2]),
            steps_remaining=np.asarray([1, 0]),
        )


def test_three_simulation_environments_are_not_four_registry_groups() -> None:
    registry = json.loads((ROOT / "manifests/experiment_registry.json").read_text(encoding="utf-8"))
    task_rows = []
    for task in registry["tasks"]:
        platform = task["source_platform"]
        for model in ("gru", "flat_mlp", "transformer"):
            for budget in (0.01, 0.05, 0.10):
                row = {
                    "task_name": task["task_name"], "source_platform": platform,
                    "simulation_environment": PLATFORM_TO_ENVIRONMENT[platform],
                    "model_name": model, "alarm_budget": budget,
                }
                row.update({measure + "_split_mean": 0.5 for measure in MEASURES})
                task_rows.append(row)
    overall, environment, groups = _summaries(task_rows, rounds=10, seed=260922)
    assert len(overall) == 9
    assert len(environment) == 27
    assert len(groups) == 36
    assert {row["simulation_environment"] for row in environment} == {
        "BulletSafetyGym", "SafetyGymnasium", "MetaDrive"
    }
    assert {row["task_count"] for row in environment if row["simulation_environment"] == "SafetyGymnasium"} == {21}
    pair_summary, pair_raw = _paired(task_rows, rounds=10, seed=260922)
    assert len(pair_summary) == 36
    assert len(pair_raw) == 1368
    assert all(row["paired_difference"] == 0.0 for row in pair_raw)


def test_stage9_real_smoke_result_reconciles_to_denominators() -> None:
    path = ROOT / "runs/offline_utility/smoke/BallRun/split_seed_0/gru.json"
    if not path.is_file():
        pytest.skip("Archived offline-replay result is not present")
    row = json.loads(path.read_text(encoding="utf-8"))
    _validate_result(
        row, task="BallRun", platform="bullet_safety_gym_suite", seed=0,
        model="gru", gate_hash=row["parents"]["g5a_gate_sha256"],
    )
    corrupted = json.loads(json.dumps(row))
    corrupted["results"]["0.05"]["test_false_alarm_episode_count"] += 1
    with pytest.raises(ValueError):
        _validate_result(
            corrupted, task="BallRun", platform="bullet_safety_gym_suite", seed=0,
            model="gru", gate_hash=row["parents"]["g5a_gate_sha256"],
        )


def test_stage9_cli_refuses_authority_output_directory() -> None:
    completed = subprocess.run(
        [
            sys.executable, "-B", "ctsc/scripts/run_revision_stage9_utility.py",
            "--task-names", "BallRun", "--split-seeds", "0", "--models", "gru",
            "--output-dir", "runs/main/stage9_must_not_write",
            "--experiment-root", "work/main", "--dataset-dir", "work/precursors",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "outside archived assets" in completed.stderr
    assert not (ROOT / "runs/main/stage9_must_not_write").exists()


@pytest.mark.parametrize("model_name", ["gru", "flat_mlp", "transformer"])
@pytest.mark.parametrize("split_name", ["val", "test"])
def test_vectorized_real_g5a_scores_match_standard_replay(model_name: str, split_name: str) -> None:
    experiment = Path(os.environ.get("OTTP_REPLAY_EXPERIMENT_ROOT", str(ROOT / "work/main")))
    datasets = Path(os.environ.get("OTTP_REPLAY_DATASET_DIR", str(ROOT / "work/precursors")))
    bundle_path = experiment / (
        "cell_runs/canonical_full/"
        f"BallRun/split_seed_0/models/{model_name}/attempt-001/precursor_model.pt"
    )
    dataset_path = datasets / "BallRun.pt"
    if not bundle_path.is_file() or not dataset_path.is_file():
        if "OTTP_REPLAY_EXPERIMENT_ROOT" in os.environ or "OTTP_REPLAY_DATASET_DIR" in os.environ:
            pytest.fail(f"Configured replay assets missing: {bundle_path}, {dataset_path}")
        pytest.skip("Train BallRun seed 0 in work/main and build work/precursors/BallRun.pt to run model replay checks")
    record = json.loads(bundle_path.with_name("run_manifest.json").read_text(encoding="utf-8"))
    assert file_sha256(dataset_path) == record["dataset"]["sha256"]
    assert file_sha256(bundle_path) == record["artifacts"]["precursor_model.pt"]["sha256"]
    extra = record["protocol"]["extra"]
    prepared = prepare_precursor_inputs(
        dataset_path, seed=0, split_seed=0, sampling_seed=0, train_seed=0, loader_seed=0,
        split_family="episode_group", ablation_group=None, ablation_mode=None, feature_profile="full",
        step_q_low=0.75, step_q_high=0.90, outcome_q_low=0.75, outcome_q_high=0.90,
        max_train_samples=extra["max_train_samples"], max_val_samples=extra["max_val_samples"], max_test_samples=extra["max_test_samples"],
        target_variant="current_inclusive", gamma=0.99,
    )
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, view, positive_index = _model_from_bundle(bundle, model_name, device)
    slow, slow_output = _predict_split(prepared, bundle, model, view, positive_index, split_name, device, 256)
    fast, fast_output = _predict_split_fast(prepared, bundle, model, view, positive_index, split_name, device, 256)
    np.testing.assert_array_equal(fast_output["sample_index"], slow_output["sample_index"])
    np.testing.assert_array_equal(fast_output["episode_uid"], slow_output["episode_uid"])
    np.testing.assert_array_equal(fast_output["steps_remaining"], slow_output["steps_remaining"])
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-8)
    metrics = json.loads(bundle_path.with_name("metrics.json").read_text(encoding="utf-8"))
    labels = (prepared.step_labels[fast_output["sample_index"]] == 2).astype(np.int64)
    saved_ap = float(metrics["label_step"][split_name]["overall"]["catastrophic_auprc"])
    assert binary_average_precision(fast, labels) == pytest.approx(saved_ap, abs=1e-7)
