from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ctsc.scripts.analyze_revision_stage4 import _endpoint_alignment_audit

from ctsc.precursor.data import (
    FeatureMatrixSequenceDataset,
    build_feature_profile_matrix,
    feature_profile_current_cost_index,
    load_precursor_dataset,
)
from ctsc.precursor.stage4 import (
    ALL_CONDITIONS,
    CONDITIONS_BY_ID,
    CORE_CONDITIONS,
    MODELS,
    PANEL_TASKS,
    SENSITIVITY_CONDITIONS,
    build_frozen_statistics,
    cell_summary_rows,
    expected_new_jobs,
    frozen_matrix_manifest,
    platform_equal_weighted_rows,
    sidecar_name,
    _metric_row,
)


def _tasks() -> list[str]:
    return [*PANEL_TASKS, *[f"SyntheticTask{index:02d}" for index in range(30)]]


def test_frozen_stage4_matrix_counts_and_no_cartesian_expansion() -> None:
    tasks = _tasks()
    manifest = frozen_matrix_manifest(tasks)
    assert manifest["expected"] == {
        "core_model_results": 912,
        "core_reused_g0_model_results": 228,
        "core_new_model_results": 684,
        "sensitivity_model_results": 240,
        "sensitivity_reused_g0_model_results": 48,
        "sensitivity_new_model_results": 192,
        "new_suite_jobs": 438,
        "new_model_results": 876,
    }
    assert len(expected_new_jobs(tasks)) == 438
    assert len(CORE_CONDITIONS) == 4
    assert len(SENSITIVITY_CONDITIONS) == 5
    assert "gamma-by-quantile Cartesian grid" in manifest["excluded"]


def test_sidecar_routing_is_target_and_gamma_specific() -> None:
    names = {condition.condition_id: sidecar_name(condition) for condition in ALL_CONDITIONS}
    assert names["core_current_inclusive_full"] is None
    assert names["core_current_inclusive_masked"] is None
    assert names["core_future_only_full"] == names["core_future_only_masked"]
    assert names["sens_g095_q75_q90"] != names["sens_g097_q75_q90"]
    assert names["sens_g099_q70_q85"] is None


def _minimal_metrics(*, include_explicit_legacy_fields: bool) -> dict:
    payload = {
        "model_name": "gru",
        "feature_profile": "full",
        "label_protocol": {
            "step_q_low": 0.75,
            "step_q_high": 0.90,
            "outcome_q_low": 0.75,
            "outcome_q_high": 0.90,
        },
        "label_step": {
            "test": {
                "overall": {
                    "catastrophic_auprc": 0.5,
                    "catastrophic_prevalence": 0.1,
                    "catastrophic_ece": 0.2,
                    "lead_time_median_at_0.5": 3.0,
                }
            }
        },
        "label_outcome": {
            "test": {"overall": {"catastrophic_auprc": 0.4, "catastrophic_prevalence": 0.08}}
        },
    }
    if include_explicit_legacy_fields:
        payload["label_protocol"].update(
            {"target_variant": "current_inclusive", "gamma": 0.99, "split_seed": 0}
        )
    return payload


def test_g0_schema_compatibility_is_limited_to_reused_canonical_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_minimal_metrics(include_explicit_legacy_fields=False)), encoding="utf-8")
    condition = CONDITIONS_BY_ID["core_current_inclusive_full"]

    row = _metric_row(
        metrics_path=metrics_path,
        task_name="SyntheticTask",
        source_platform="synthetic",
        split_seed=0,
        condition=condition,
        reused_g0=True,
    )
    assert row["target_variant"] == "current_inclusive"
    assert row["gamma"] == 0.99

    with np.testing.assert_raises_regex(ValueError, "target_variant=None"):
        _metric_row(
            metrics_path=metrics_path,
            task_name="SyntheticTask",
            source_platform="synthetic",
            split_seed=0,
            condition=condition,
            reused_g0=False,
        )


def test_g0_schema_compatibility_does_not_accept_present_mismatch(tmp_path: Path) -> None:
    payload = _minimal_metrics(include_explicit_legacy_fields=True)
    payload["label_protocol"]["gamma"] = 0.95
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "gamma=0.95"):
        _metric_row(
            metrics_path=metrics_path,
            task_name="SyntheticTask",
            source_platform="synthetic",
            split_seed=0,
            condition=CONDITIONS_BY_ID["core_current_inclusive_full"],
            reused_g0=True,
        )


def test_endpoint_alignment_audit_is_machine_readable_and_rejects_missing_pair() -> None:
    row = {
        "step_auprc": 0.5,
        "matched_step_prevalence": 0.1,
        "outcome_auprc": 0.4,
        "matched_outcome_prevalence": 0.08,
    }
    audit = _endpoint_alignment_audit([row])
    assert audit["status"] == "PASS"
    assert audit["row_count"] == 1
    assert audit["cross_endpoint_pairing"] is False

    incomplete = dict(row)
    incomplete.pop("matched_outcome_prevalence")
    with np.testing.assert_raises_regex(ValueError, "Missing outcome endpoint field"):
        _endpoint_alignment_audit([incomplete])


def test_history_without_current_cost_changes_only_current_cost_cell(
    mock_public_dataset_files,
) -> None:
    dataset = load_precursor_dataset(mock_public_dataset_files["precursor_path"])
    features = build_feature_profile_matrix(dataset, "full")
    cost_index = feature_profile_current_cost_index(dataset, "full")
    positions = np.asarray(dataset["episode_position"], dtype=np.int64)
    candidates = np.flatnonzero(positions >= 1)
    sample_index = int(candidates[0])
    features[sample_index, int(cost_index)] = 3.0
    labels = np.zeros(features.shape[0], dtype=np.int64)
    common = {
        "features": features,
        "dataset": dataset,
        "indices": np.asarray([sample_index], dtype=np.int64),
        "labels": labels,
        "window_size": 4,
        "shuffle_seed": 0,
        "current_cost_index": cost_index,
    }
    full = FeatureMatrixSequenceDataset(sequence_view="full", **common)[0]["sequence"].numpy()
    masked = FeatureMatrixSequenceDataset(sequence_view="history_without_current_cost", **common)[0]["sequence"].numpy()
    changed = np.argwhere(full != masked)
    assert changed.tolist() == [[3, int(cost_index)]]
    assert masked[3, int(cost_index)] == 0.0
    assert np.array_equal(np.delete(full, int(cost_index), axis=1), np.delete(masked, int(cost_index), axis=1))


def test_frozen_f4_f5_statistics_use_task_pairs_and_holm_families() -> None:
    tasks = _tasks()
    rows = []
    condition_values = {
        "core_current_inclusive_full": (0.80, 0.10),
        "core_current_inclusive_masked": (0.70, 0.10),
        "core_future_only_full": (0.60, 0.20),
        "core_future_only_masked": (0.50, 0.20),
        "sens_g099_q75_q90": (0.80, 0.10),
        "sens_g095_q75_q90": (0.75, 0.10),
        "sens_g097_q75_q90": (0.77, 0.10),
        "sens_g099_q70_q85": (0.72, 0.15),
        "sens_g099_q80_q95": (0.84, 0.05),
    }
    for condition in ALL_CONDITIONS:
        selected_tasks = tasks if condition.matrix == "core" else list(PANEL_TASKS)
        for task in selected_tasks:
            for model in MODELS:
                auprc, prevalence = condition_values[condition.condition_id]
                rows.append(
                    {
                        "matrix": condition.matrix,
                        "condition_id": condition.condition_id,
                        "task_name": task,
                        "model_name": model,
                        "source_platform": "synthetic",
                        "step_auprc": auprc,
                        "matched_step_prevalence": prevalence,
                        "step_auprc_minus_prevalence": auprc - prevalence,
                    }
                )
    paired, stats = build_frozen_statistics(
        rows,
        bootstrap_rounds=100,
        bootstrap_seed=260922,
        permutation_rounds=1000,
        permutation_seed=260921,
    )
    f4 = [row for row in stats if row["family"] == "F4_anti_shortcut"]
    f5 = [row for row in stats if row["family"] == "F5_parameter_sensitivity"]
    assert len(f4) == 6 and all(row["n"] == 38 and row["holm_family_size"] == 6 for row in f4)
    assert len(f5) == 8 and all(row["n"] == 8 and row["holm_family_size"] == 8 for row in f5)
    assert all(row["permutation_mode"] == "monte_carlo_add_one" for row in f4)
    assert all(row["permutation_mode"] == "exact_all_sign_patterns" for row in f5)
    assert all(row["permutation_effective_draws"] == 256 for row in f5)
    assert len(paired) == 6 * 38 + 8 * 8
    assert not any(
        "current_inclusive" in row["left_condition_id"] and "future_only" in row["right_condition_id"]
        for row in paired
    )

    cell_rows = cell_summary_rows(rows)
    assert len(cell_rows) == 4 * 2 + 5 * 2
    canonical_gru = next(
        row
        for row in cell_rows
        if row["condition_id"] == "core_current_inclusive_full" and row["model_name"] == "gru"
    )
    assert canonical_gru["task_count"] == 38
    assert np.isclose(canonical_gru["step_auprc_mean"], 0.80)
    assert np.isclose(canonical_gru["baseline_adjusted_lift_mean"], 0.70)

    platform_rows, platform_summaries = platform_equal_weighted_rows(paired)
    assert platform_rows
    assert len(platform_summaries) == 14
    assert all(row["role"] == "descriptive_only_not_a_hypothesis_family" for row in platform_summaries)
