from __future__ import annotations

import numpy as np
import pytest

from ctsc.precursor.data import (
    assign_3way_labels,
    build_group_splits,
    build_transition_random_splits,
    compute_label_thresholds,
    summarize_split_diagnostics,
)


def test_build_group_splits_avoids_group_leakage() -> None:
    dataset = {
        "env_index": np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64),
        "run_group_index": np.asarray([0, 0, 1, 1, 2, 2, 10, 10, 11, 11, 12, 12], dtype=np.int64),
    }

    splits = build_group_splits(dataset, seed=7)
    train = set(np.asarray(splits["train"], dtype=np.int64).tolist())
    val = set(np.asarray(splits["val"], dtype=np.int64).tolist())
    test = set(np.asarray(splits["test"], dtype=np.int64).tolist())

    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(12))

    manifest = splits["manifest"]
    for env_id in (0, 1):
        train_groups = set(manifest["train_groups"][env_id])
        val_groups = set(manifest["val_groups"][env_id])
        test_groups = set(manifest["test_groups"][env_id])
        assert train_groups
        assert val_groups
        assert test_groups
        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)


def test_build_transition_random_splits_is_deterministic_and_disjoint() -> None:
    dataset = {
        "env_index": np.asarray([0] * 12 + [1] * 12, dtype=np.int64),
        "run_group_index": np.asarray([0] * 6 + [1] * 6 + [10] * 6 + [11] * 6, dtype=np.int64),
        "episode_uid": np.asarray(
            [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [10] * 3 + [11] * 3 + [12] * 3 + [13] * 3,
            dtype=np.int64,
        ),
    }

    first = build_transition_random_splits(dataset, seed=5)
    second = build_transition_random_splits(dataset, seed=5)
    train = set(np.asarray(first["train"], dtype=np.int64).tolist())
    val = set(np.asarray(first["val"], dtype=np.int64).tolist())
    test = set(np.asarray(first["test"], dtype=np.int64).tolist())

    assert np.array_equal(first["train"], second["train"])
    assert np.array_equal(first["val"], second["val"])
    assert np.array_equal(first["test"], second["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(24))


def test_transition_random_split_fragments_multistep_episodes() -> None:
    dataset = {
        "env_index": np.asarray([0] * 36, dtype=np.int64),
        "run_group_index": np.asarray([0] * 18 + [1] * 18, dtype=np.int64),
        "episode_uid": np.repeat(np.arange(6, dtype=np.int64), 6),
    }

    splits = build_transition_random_splits(dataset, seed=0)
    diagnostics = summarize_split_diagnostics(dataset, splits)

    assert diagnostics["group_overlap"]["group_leakage_detected"] is True
    assert diagnostics["episode_fragmentation"]["fragmentation_detected"] is True
    assert diagnostics["episode_fragmentation"]["fragmented_episode_count"] > 0


def test_group_split_diagnostics_show_no_fragmentation_or_overlap() -> None:
    dataset = {
        "env_index": np.asarray([0] * 18, dtype=np.int64),
        "run_group_index": np.repeat(np.arange(6, dtype=np.int64), 3),
        "episode_uid": np.repeat(np.arange(6, dtype=np.int64), 3),
    }

    splits = build_group_splits(dataset, seed=2)
    diagnostics = summarize_split_diagnostics(dataset, splits)

    assert diagnostics["group_overlap"]["group_leakage_detected"] is False
    assert diagnostics["episode_fragmentation"]["fragmentation_detected"] is False


def test_compute_label_thresholds_and_assign_3way_labels() -> None:
    values = np.asarray([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=np.float64)
    env_index = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    dataset = {
        "env_index": env_index,
        "steps_remaining": values,
    }

    thresholds = compute_label_thresholds(
        dataset,
        indices=np.arange(values.size, dtype=np.int64),
        value_key="steps_remaining",
    )

    assert thresholds[0]["q75"] == pytest.approx(4.0)
    assert thresholds[0]["q90"] == pytest.approx(4.6)
    assert thresholds[1]["q75"] == pytest.approx(40.0)
    assert thresholds[1]["q90"] == pytest.approx(46.0)

    labels = assign_3way_labels(values, env_index, thresholds)
    assert labels.tolist() == [0, 0, 0, 1, 2, 0, 0, 0, 1, 2]


def test_compute_label_thresholds_rejects_invalid_quantiles() -> None:
    dataset = {
        "env_index": np.asarray([0, 0, 0], dtype=np.int64),
        "steps_remaining": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
    }

    with pytest.raises(ValueError, match="Quantiles must satisfy"):
        compute_label_thresholds(
            dataset,
            indices=np.arange(3, dtype=np.int64),
            value_key="steps_remaining",
            lower_quantile=0.9,
            upper_quantile=0.75,
        )
