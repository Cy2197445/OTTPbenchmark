from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ctsc.precursor.public_offline import summarize_public_precursor_dataset
from ctsc.precursor.revision import (
    GIB,
    aggregation_identity,
    assert_valid_run_manifest,
    build_label_sidecar,
    derive_discounted_target,
    export_reproducibility_artifacts,
    guard_revision_output_dir,
    protocol_identity,
    require_free_space,
    resolve_seed_bundle,
    seeded_output_path,
)
from ctsc.scripts.train_precursor_model import prepare_precursor_inputs, run_precursor_model


def _prepare(path: Path, **seed_overrides: int):
    return prepare_precursor_inputs(
        path,
        seed=7,
        split_seed=seed_overrides.get("split_seed"),
        sampling_seed=seed_overrides.get("sampling_seed"),
        train_seed=seed_overrides.get("train_seed"),
        loader_seed=seed_overrides.get("loader_seed"),
        split_family="episode_group",
        ablation_group=None,
        ablation_mode=None,
        feature_profile="full",
        step_q_low=0.75,
        step_q_high=0.90,
        outcome_q_low=0.75,
        outcome_q_high=0.90,
        max_train_samples=10,
        max_val_samples=5,
        max_test_samples=None,
    )


def test_seed_bundle_preserves_legacy_and_decouples_explicit_streams() -> None:
    legacy = resolve_seed_bundle(seed=3)
    assert legacy.legacy_compatible
    assert (legacy.split_seed, legacy.sampling_seed, legacy.train_seed, legacy.loader_seed) == (3, 3, 3, 3)

    explicit = resolve_seed_bundle(seed=3, split_seed=1, train_seed=9)
    assert not explicit.legacy_compatible
    assert explicit.split_seed == 1
    assert explicit.sampling_seed == 1
    assert explicit.train_seed == 9
    assert explicit.loader_seed == 9


def test_train_seed_does_not_change_data_state(mock_public_dataset_files: dict[str, Path]) -> None:
    first = _prepare(mock_public_dataset_files["precursor_path"], split_seed=0, sampling_seed=0, train_seed=1, loader_seed=1)
    second = _prepare(mock_public_dataset_files["precursor_path"], split_seed=0, sampling_seed=0, train_seed=4, loader_seed=4)
    for split in ("train", "val", "test"):
        assert np.array_equal(first.uncapped_splits[split], second.uncapped_splits[split])
        assert np.array_equal(first.base_splits[split], second.base_splits[split])
    assert first.step_thresholds == second.step_thresholds
    assert first.outcome_thresholds == second.outcome_thresholds
    assert first.normalization_mean.tobytes() == second.normalization_mean.tobytes()
    assert first.normalization_std.tobytes() == second.normalization_std.tobytes()


def test_legacy_and_explicit_compatible_data_indices_match(mock_public_dataset_files: dict[str, Path]) -> None:
    legacy = _prepare(mock_public_dataset_files["precursor_path"])
    explicit = _prepare(
        mock_public_dataset_files["precursor_path"],
        split_seed=7,
        sampling_seed=7,
        train_seed=7,
        loader_seed=7,
    )
    for split in ("train", "val", "test"):
        assert np.array_equal(legacy.uncapped_splits[split], explicit.uncapped_splits[split])
        assert np.array_equal(legacy.base_splits[split], explicit.base_splits[split])
    assert legacy.step_thresholds == explicit.step_thresholds
    assert legacy.normalization_mean.tobytes() == explicit.normalization_mean.tobytes()


def test_public_dataset_exposes_episode_group_alias(mock_public_dataset_files: dict[str, Path]) -> None:
    import torch

    dataset = torch.load(mock_public_dataset_files["precursor_path"], map_location="cpu", weights_only=False)
    assert np.array_equal(dataset["episode_group_index"], dataset["run_group_index"])
    assert dataset["metadata"]["run_group_index_semantics"] == "legacy_alias_for_reconstructed_episode_group_not_logging_run"
    summary = summarize_public_precursor_dataset(dataset)
    assert summary["num_episode_groups"] == summary["num_run_groups"]


def test_seeded_output_paths_and_guards(tmp_path: Path) -> None:
    seeds1 = resolve_seed_bundle(seed=0, split_seed=0, train_seed=1)
    seeds2 = resolve_seed_bundle(seed=0, split_seed=0, train_seed=2)
    assert seeded_output_path(tmp_path, seeds1) != seeded_output_path(tmp_path, seeds2)
    protected = Path(__file__).resolve().parents[1] / "pilot_artifacts" / "Unified38_fresh_rerun_active" / "x"
    with pytest.raises(PermissionError):
        guard_revision_output_dir(protected, repo_root=Path(__file__).resolve().parents[1])
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        guard_revision_output_dir(occupied, repo_root=Path(__file__).resolve().parents[1])


def test_disk_guard_rejects_less_than_ten_gib(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        require_free_space(tmp_path, minimum_gib=10.0, observed_free_bytes=10 * GIB - 1)
    assert require_free_space(tmp_path, minimum_gib=10.0, observed_free_bytes=10 * GIB) == 10 * GIB


def test_compact_reproducibility_export_roundtrips(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    prepared = _prepare(mock_public_dataset_files["precursor_path"], split_seed=0, train_seed=1)
    record = export_reproducibility_artifacts(
        output_dir=tmp_path / "repro",
        dataset=prepared.dataset,
        uncapped_splits=prepared.uncapped_splits,
        capped_splits=prepared.base_splits,
        step_thresholds=prepared.step_thresholds,
        outcome_thresholds=prepared.outcome_thresholds,
        normalization_mean=prepared.normalization_mean,
        normalization_std=prepared.normalization_std,
    )
    with np.load(tmp_path / "repro" / "uncapped_split_indices.npz", allow_pickle=False) as indices:
        assert np.array_equal(indices["train"], prepared.uncapped_splits["train"])
        assert np.array_equal(indices["validation"], prepared.uncapped_splits["val"])
        assert np.array_equal(indices["test"], prepared.uncapped_splits["test"])
    assert record["uncapped"]["test"]["count"] == len(prepared.uncapped_splits["test"])
    assert (tmp_path / "repro" / "episode_group_assignment.csv.gz").is_file()
    assert (tmp_path / "repro" / "thresholds.json").stat().st_size < 100_000


def test_target_sidecars_bind_base_and_do_not_copy_inputs(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    import torch

    base_path = mock_public_dataset_files["precursor_path"]
    dataset = torch.load(base_path, map_location="cpu", weights_only=False)
    prepared = _prepare(base_path, split_seed=0, sampling_seed=0, train_seed=0)
    future_path = tmp_path / "future_g099.npz"
    future_manifest = build_label_sidecar(
        dataset=dataset,
        base_dataset_path=base_path,
        train_indices=prepared.base_splits["train"],
        output_path=future_path,
        target_variant="future_only",
        gamma=0.99,
        quantile_pair=(0.75, 0.90),
    )
    gamma_path = tmp_path / "future_g097.npz"
    gamma_manifest = build_label_sidecar(
        dataset=dataset,
        base_dataset_path=base_path,
        train_indices=prepared.base_splits["train"],
        output_path=gamma_path,
        target_variant="future_only",
        gamma=0.97,
        quantile_pair=(0.75, 0.90),
    )
    current_path = tmp_path / "current_g099.npz"
    current_manifest = build_label_sidecar(
        dataset=dataset,
        base_dataset_path=base_path,
        train_indices=prepared.base_splits["train"],
        output_path=current_path,
        target_variant="current_inclusive",
        gamma=0.99,
        quantile_pair=(0.75, 0.90),
    )
    assert future_manifest["base_dataset_sha256"] == gamma_manifest["base_dataset_sha256"]
    assert future_manifest["base_arrays_sha256"] == gamma_manifest["base_arrays_sha256"]
    assert future_manifest["base_arrays_sha256"] == current_manifest["base_arrays_sha256"]
    assert future_manifest["sidecar_sha256"] != gamma_manifest["sidecar_sha256"]
    assert future_manifest["sidecar_sha256"] != current_manifest["sidecar_sha256"]
    assert not future_manifest["contains_base_observation_or_action_arrays"]
    with np.load(future_path, allow_pickle=False) as sidecar:
        assert set(sidecar.files) == {"target_values", "labels"}
        values = sidecar["target_values"]
    episodes = np.asarray(dataset["episode_uid"], dtype=np.int64)
    costs = np.asarray(dataset["costs"], dtype=np.float32)
    for episode in np.unique(episodes):
        idx = np.flatnonzero(episodes == episode)
        assert values[idx[-1]] == pytest.approx(0.0)
        if len(idx) > 1:
            assert values[idx[0]] == pytest.approx(float(costs[idx[1]]) + 0.99 * float(values[idx[1]]), abs=1e-6)


def test_future_only_derivation_never_crosses_episode_boundary() -> None:
    costs = np.asarray([1.0, 2.0, 10.0, 20.0], dtype=np.float32)
    episodes = np.asarray([0, 0, 1, 1], dtype=np.int64)
    result = derive_discounted_target(costs, episodes, gamma=0.5, target_variant="future_only")
    assert result.tolist() == pytest.approx([2.0, 0.0, 20.0, 0.0])


def test_manifest_identity_separates_protocol_dimensions() -> None:
    seeds1 = resolve_seed_bundle(seed=0, split_seed=0, train_seed=1)
    seeds2 = resolve_seed_bundle(seed=0, split_seed=0, train_seed=2)
    p1 = protocol_identity(
        target_variant="current_inclusive",
        gamma=0.99,
        quantile_pair=(0.75, 0.90),
        split_family="episode_group",
        seeds=seeds1,
        feature_profile="full",
        model_name="gru",
    )
    p2 = protocol_identity(
        target_variant="current_inclusive",
        gamma=0.99,
        quantile_pair=(0.75, 0.90),
        split_family="episode_group",
        seeds=seeds2,
        feature_profile="full",
        model_name="gru",
    )
    p3 = protocol_identity(
        target_variant="current_inclusive",
        gamma=0.99,
        quantile_pair=(0.75, 0.90),
        outcome_quantile_pair=(0.80, 0.95),
        split_family="episode_group",
        seeds=seeds1,
        feature_profile="full",
        model_name="gru",
    )
    assert aggregation_identity(p1) != aggregation_identity(p2)
    assert aggregation_identity(p1) != aggregation_identity(p3)
    manifest = {
        "schema_version": "ottp_revision_run_manifest_v1",
        "freeze_id": "OTTP-R26-G1-v1",
        "experiment_id": "example",
        "attempt_id": "attempt-001",
        "work_package": "test",
        "source_snapshot_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "dataset": {},
        "protocol": p1,
        "indices": {"uncapped": {}, "capped": {}},
        "artifacts": {},
    }
    assert_valid_run_manifest(manifest)


def test_reference_run_writes_valid_manifest(mock_public_dataset_files: dict[str, Path], tmp_path: Path) -> None:
    prepared = _prepare(mock_public_dataset_files["precursor_path"], split_seed=0, sampling_seed=0, train_seed=2, loader_seed=3)
    output = seeded_output_path(tmp_path / "runs", prepared.seeds) / "prevalence"
    metrics = run_precursor_model(
        prepared=prepared,
        model_name="prevalence",
        output_dir=output,
        device="cpu",
        window_size=32,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        cnn_kernel_size=3,
        mlp_depth=1,
        batch_size=16,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-5,
        num_workers=0,
        critic_baseline_mode="disabled",
        minimum_free_space_gib=0.0,
    )
    payload = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert_valid_run_manifest(payload)
    assert metrics["split_seed"] == 0
    assert metrics["train_seed"] == 2
    assert payload["protocol"]["loader_seed"] == 3


