from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ctsc.precursor.cc1 import LOCKED_MODEL_LRS
from ctsc.precursor.revision import array_sha256, file_sha256, resolve_seed_bundle
from ctsc.precursor.stage7 import MODELS, TASKS, frozen_jobs, validate_parent_run, validate_shared_files
from ctsc.scripts.train_precursor_model import prepare_precursor_inputs


def _prepared(path: Path, *, split_seed: int, train_seed: int):
    return prepare_precursor_inputs(
        path, seed=0, split_seed=split_seed, sampling_seed=0, train_seed=train_seed, loader_seed=0,
        split_family="episode_group", ablation_group=None, ablation_mode=None,
        feature_profile="full", step_q_low=0.75, step_q_high=0.9,
        outcome_q_low=0.75, outcome_q_high=0.9,
        max_train_samples=20, max_val_samples=10, max_test_samples=None,
        target_variant="current_inclusive", gamma=0.99,
    )


def test_frozen_matrix_is_120_with_only_96_new_runs():
    jobs = frozen_jobs()
    assert len(TASKS) == 8 and len(MODELS) == 3 and len(jobs) == 120
    assert sum(row["source"] == "G5A_REUSE" for row in jobs) == 24
    assert sum(row["source"] == "STAGE7_NEW" for row in jobs) == 96
    assert {row["learning_rate"] for row in jobs if row["model"] == "gru"} == {LOCKED_MODEL_LRS["gru"]}
    assert all((row["split_seed"], row["sampling_seed"], row["loader_seed"]) == (0, 0, 0) for row in jobs)


def test_train_seed_only_changes_training_randomness(mock_public_dataset_files: dict[str, Path]):
    path = mock_public_dataset_files["precursor_path"]
    first = _prepared(path, split_seed=0, train_seed=0)
    second = _prepared(path, split_seed=0, train_seed=4)
    for name in ("train", "val", "test"):
        assert array_sha256(first.uncapped_splits[name]) == array_sha256(second.uncapped_splits[name])
        assert array_sha256(first.base_splits[name]) == array_sha256(second.base_splits[name])
    assert first.step_thresholds == second.step_thresholds
    assert first.outcome_thresholds == second.outcome_thresholds
    assert np.array_equal(first.normalization_mean, second.normalization_mean)
    assert np.array_equal(first.normalization_std, second.normalization_std)
    assert (second.seeds.split_seed, second.seeds.sampling_seed, second.seeds.train_seed, second.seeds.loader_seed) == (0, 0, 4, 0)


def test_interface_can_change_split_with_train_fixed(mock_public_dataset_files: dict[str, Path]):
    path = mock_public_dataset_files["precursor_path"]
    first = _prepared(path, split_seed=0, train_seed=0)
    second = _prepared(path, split_seed=1, train_seed=0)
    assert first.seeds.train_seed == second.seeds.train_seed == 0
    assert first.seeds.loader_seed == second.seeds.loader_seed == 0
    assert first.seeds.split_seed != second.seeds.split_seed
    assert not np.array_equal(first.uncapped_splits["train"], second.uncapped_splits["train"])




def test_shared_artifact_hash_mismatch_blocks(tmp_path: Path):
    path = tmp_path / "indices.npz"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="shared artifact mismatch"):
        validate_shared_files({"files": {"indices.npz": {"sha256": "0" * 64, "bytes": 5}}}, tmp_path)


def test_output_attempt_guard_and_cuda_oom_scope(tmp_path: Path):
    from ctsc.scripts.run_revision_stage7_cell import _attempt_dir, _is_cuda_oom
    base = tmp_path / "models/gru"
    first = _attempt_dir(base)
    assert first.name == "attempt-001"
    first.mkdir(parents=True)
    assert _attempt_dir(base).name == "attempt-002"
    assert _is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not _is_cuda_oom(RuntimeError("CPU allocation failed"))


def test_stability_summary_uses_sample_sd_and_fixed_quantiles():
    from ctsc.scripts.analyze_revision_stage7 import _numeric_summary
    observed = _numeric_summary([1, 2, 3, 4, 5])
    assert observed["mean"] == 3
    assert observed["median"] == 3
    assert observed["sample_sd"] == pytest.approx(np.sqrt(2.5))
    assert observed["q25"] == 2 and observed["q75"] == 4 and observed["iqr"] == 2
    with pytest.raises(ValueError):
        _numeric_summary([1, float("nan")])
