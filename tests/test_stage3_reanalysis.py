from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ctsc.precursor.stage3 import (
    RunRecord,
    audit_histories,
    bootstrap_mean_ci,
    build_carcircle_trace,
    build_complexity_rows,
    build_primary_rows,
    collect_run_records,
    holm_adjust,
    paired_dz,
    paired_sign_flip_pvalue,
    plot_best_epoch,
    plot_convergence,
    guard_stage3_output,
    task_model_means,
    validate_endpoint_pair,
    validate_grid,
    validate_matched_prevalence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_TASK_ROOT = REPO_ROOT / "pilot_artifacts/Unified38_fresh_rerun_active/task_runs"
CURRENT_TASK_ROOT = REPO_ROOT / "pilot_artifacts/Unified38_current_only_matrix/task_runs"


@pytest.fixture(scope="module")
def frozen_records() -> tuple[list[RunRecord], list[RunRecord]]:
    full = collect_run_records(FULL_TASK_ROOT, condition="full_prefix", repo_root=REPO_ROOT, hash_metrics=False)
    current = collect_run_records(
        CURRENT_TASK_ROOT,
        condition="current_only",
        repo_root=REPO_ROOT,
        hash_metrics=False,
    )
    return full, current










def test_endpoint_alignment_is_strict() -> None:
    validate_endpoint_pair("step", "step")
    validate_endpoint_pair("outcome", "outcome")
    with pytest.raises(ValueError, match="Cross-endpoint"):
        validate_endpoint_pair("step", "outcome")
    with pytest.raises(ValueError, match="Cross-endpoint"):
        validate_endpoint_pair("outcome", "step")






def test_statistical_helpers_are_deterministic_and_known() -> None:
    values = [float(index - 10) / 10.0 for index in range(21)]
    assert bootstrap_mean_ci(values, rounds=500, seed=260922) == bootstrap_mean_ci(
        values, rounds=500, seed=260922
    )
    pvalue = paired_sign_flip_pvalue(values, rounds=1000, seed=260921)
    assert pvalue == paired_sign_flip_pvalue(values, rounds=1000, seed=260921)
    assert 1.0 / 1001.0 <= pvalue <= 1.0
    assert paired_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )


def test_parameter_inconsistency_is_rejected() -> None:
    placeholder = Path("placeholder")
    base = RunRecord(
        condition="full_prefix",
        task_name="Synthetic",
        slice_name="canonical",
        split_seed=0,
        model_name="gru",
        source_platform="synthetic",
        step_auprc=0.5,
        outcome_auprc=0.5,
        step_ece=0.1,
        step_lead=1.0,
        step_prevalence=0.1,
        outcome_prevalence=0.1,
        training={},
        protocol_fields={},
        metrics_path=placeholder,
        metrics_sha256="",
        bundle_path=placeholder,
        parameter_count=10,
        model_config={"input_dim": 2, "window_size": 3},
        state_shape_signature="one",
    )
    changed = deepcopy(base)
    changed.split_seed = 1
    changed.parameter_count = 11
    changed.state_shape_signature = "two"
    with pytest.raises(ValueError, match="Parameter/state shape inconsistency"):
        build_complexity_rows([base, changed])


def test_output_guard_rejects_authority_and_nonempty_directory(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        guard_stage3_output(
            REPO_ROOT / "pilot_artifacts/Unified38_fresh_rerun_active/stage3_bad",
            repo_root=REPO_ROOT,
        )
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        guard_stage3_output(nonempty, repo_root=REPO_ROOT)


def test_final_figure_formats_and_dimensions(tmp_path: Path) -> None:
    curve_rows = []
    best_rows = []
    models = ("gru", "temporal_cnn", "flat_mlp", "linear")
    for slice_name in ("canonical", "sensitivity"):
        for model_index, model in enumerate(models):
            for epoch in range(12):
                curve_rows.append(
                    {
                        "condition": "full_prefix",
                        "slice_name": slice_name,
                        "model_name": model,
                        "epoch": epoch,
                        "task_mean_validation_step_auprc": 0.4 + 0.01 * epoch + 0.01 * model_index,
                        "task_bootstrap_ci95_low": 0.38 + 0.01 * epoch + 0.01 * model_index,
                        "task_bootstrap_ci95_high": 0.42 + 0.01 * epoch + 0.01 * model_index,
                    }
                )
                best_rows.append(
                    {
                        "condition": "full_prefix",
                        "slice_name": slice_name,
                        "model_name": model,
                        "best_epoch": epoch,
                        "run_proportion": 1.0 / 12.0,
                    }
                )
    outputs = [
        *plot_convergence(curve_rows, condition="full_prefix", output_dir=tmp_path),
        *plot_best_epoch(best_rows, condition="full_prefix", output_dir=tmp_path),
    ]
    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
    for path in outputs:
        assert path.is_file() and path.stat().st_size > 0
    for png in tmp_path.glob("*.png"):
        image = __import__("matplotlib.image").image.imread(png)
        assert image.shape[1] >= 1800
        assert image.shape[0] >= 800
