from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ctsc.precursor.public_paths import raw_dataset_path
from ctsc.precursor.revision import canonical_json_sha256, file_sha256
from ctsc.scripts.analyze_benchmark import collect_results
from ctsc.scripts.run_benchmark import ROOT, output_directory
from ctsc.scripts import run_Unified38_target_definition_ablation as target


@pytest.mark.parametrize("response", ["missing", "failed", "valid"])
def test_gpu_probe_supports_cpu_hosts(monkeypatch, response):
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        if response == "missing":
            raise FileNotFoundError("nvidia-smi")
        return subprocess.CompletedProcess(command, 1 if response == "failed" else 0,
                                           stdout="GPU, 8192, 1024, 7168, 50, 60", stderr="")
    monkeypatch.setattr(target.subprocess, "run", run)
    snapshot = target._gpu_snapshot()
    assert (snapshot is not None) == (response == "valid")
    if snapshot:
        assert snapshot["memory_total_mb"] == 8192


@pytest.mark.parametrize("script", ["run_Unified38_target_definition_ablation", "run_Unified38_split_family_counterfactual",
                                   "run_Unified38_actionability_offline_utility", "audit_Unified38_current_rerun"])
def test_auxiliary_registry_defaults_exist(script):
    module = importlib.import_module("ctsc.scripts." + script)
    path = getattr(module, "REGISTRY_PATH", None) or module.DEFAULT_REGISTRY
    assert path == ROOT / "manifests/experiment_registry.json"
    assert path.is_file()


@pytest.mark.parametrize("script", ["run_Unified38_target_definition_ablation", "run_Unified38_gru_profile_matrix",
                                   "run_Unified38_gru_window_sweep", "run_Unified38_actionability_offline_utility"])
def test_auxiliary_precursor_defaults_match_build_directory(script):
    assert importlib.import_module("ctsc.scripts." + script).DEFAULT_PRECURSOR_SOURCE_DIR == ROOT / "work/precursors"


@pytest.mark.parametrize("download_log", [False, True])
def test_raw_paths_accept_source_catalog_and_download_log(tmp_path, download_log):
    raw = tmp_path / "task.hdf5"
    raw.write_bytes(b"dataset fixture")
    row = {"task_name": "BallRun", "file_name": raw.name, "bytes": raw.stat().st_size, "sha256": file_sha256(raw)}
    if download_log:
        row["output_path"] = str(raw)
    assert raw_dataset_path("BallRun", {"tasks": [row]}, tmp_path, root=ROOT) == raw
    from ctsc.scripts.run_Unified38_anti_shortcut_suite import _task_dataset_path
    assert Path(_task_dataset_path("BallRun", {"tasks": [row]}, tmp_path)) == raw
    raw.write_bytes(b"dataset changed")
    with pytest.raises(ValueError, match="SHA256"):
        raw_dataset_path("BallRun", {"tasks": [row]}, tmp_path, root=ROOT)


def test_split_family_reuses_precursors_without_raw_files(monkeypatch, tmp_path):
    from ctsc.scripts import run_Unified38_split_family_counterfactual as split
    precursors = tmp_path / "precursors"
    precursors.mkdir()
    (precursors / "BallRun.pt").write_bytes(b"path fixture")
    calls = []
    monkeypatch.setattr(sys, "argv", ["split", "--tasks", "BallRun", "--split-seeds", "0", "--split-families", "episode_group",
                                    "--precursor-source-dir", str(precursors), "--raw-dir", str(tmp_path / "missing"),
                                    "--output-dir", str(tmp_path / "output"), "--device", "cpu"])
    monkeypatch.setattr(split, "_run", lambda command, **kwargs: calls.append(command))
    split.main()
    assert len(calls) == 4
    assert all("build_public_precursor_dataset.py" not in " ".join(c) for c in calls)


@pytest.fixture
def small_matrix(tmp_path):
    experiment = tmp_path / "experiment"
    relative = Path("cell_runs/canonical_full/BallRun/split_seed_0")
    source, cell = ROOT / "runs/main" / relative, experiment / relative
    shutil.copytree(source, cell)
    for model in ("gru", "flat_mlp", "transformer"):
        attempt = cell / "models" / model / "attempt-001"
        bundle = attempt / "precursor_model.pt"
        bundle.write_bytes(b"test fixture only")
        path = attempt / "run_manifest.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["artifacts"]["precursor_model.pt"] = {"sha256": file_sha256(bundle), "bytes": bundle.stat().st_size}
        path.write_text(json.dumps(record), encoding="utf-8")
    (experiment / "matrix.json").write_text(json.dumps({"cells": [{"condition_id": "canonical_full", "task_name": "BallRun",
                                                                "split_seed": 0, "models": ["gru", "flat_mlp", "transformer"]}],
                                                       "smoke": False}), encoding="utf-8")
    return experiment


def test_partial_analysis_has_seed_metrics_but_no_full_statistics(small_matrix, tmp_path, monkeypatch):
    from ctsc.scripts import analyze_benchmark as analysis
    assert len(collect_results(small_matrix, partial=True)) == 3
    with pytest.raises(ValueError, match="Incomplete 38-task"):
        collect_results(small_matrix)
    output = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["analyze", "--experiment-root", str(small_matrix), "--output-dir", str(output), "--partial"])
    analysis.main()
    assert (output / "seed_level_all_results.csv").is_file()
    assert not (output / "f1_f2_f3_statistics.csv").exists()
    with pytest.raises(FileExistsError):
        analysis.main()


@pytest.mark.parametrize("damage", ["metrics", "preprocessing", "plan", "weights", "history"])
def test_analysis_rejects_inconsistent_runs(small_matrix, damage):
    cell = small_matrix / "cell_runs/canonical_full/BallRun/split_seed_0"
    attempt = cell / "models/gru/attempt-001"
    if damage == "metrics":
        with (attempt / "metrics.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
    elif damage == "preprocessing":
        (cell / "reproducibility/normalization_stats.npz").write_bytes(b"corrupted fixture")
    elif damage == "weights":
        (attempt / "precursor_model.pt").unlink()
    elif damage == "history":
        metrics = json.loads((attempt / "metrics.json").read_text(encoding="utf-8"))
        metrics["training"]["best_epoch"] = -1
        (attempt / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        record = json.loads((attempt / "run_manifest.json").read_text(encoding="utf-8"))
        record["artifacts"]["metrics.json"]["sha256"] = file_sha256(attempt / "metrics.json")
        (attempt / "run_manifest.json").write_text(json.dumps(record), encoding="utf-8")
    else:
        plan = json.loads((small_matrix / "matrix.json").read_text(encoding="utf-8"))
        plan["cells"][0]["models"] = ["gru"]
        (small_matrix / "matrix.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        collect_results(small_matrix, partial=True)


def test_analysis_refuses_empty_input_and_protected_output(tmp_path):
    with pytest.raises(ValueError, match="No main-matrix runs"):
        collect_results(tmp_path, partial=True)
    with pytest.raises(ValueError, match="outside archived assets"):
        output_directory(ROOT / "provenance/output")


def test_legacy_aggregation_refuses_new_matrix_layout(tmp_path, monkeypatch):
    from ctsc.scripts import aggregate_Unified38_current_runs as aggregation
    monkeypatch.setattr(sys, "argv", ["aggregate", "--input-root", str(tmp_path), "--output-dir", str(tmp_path / "output"),
                                    "--task-names", "BallRun"])
    with pytest.raises(ValueError, match="analyze_benchmark.py"):
        aggregation.main()


def test_ablation_reference_preserves_legacy_seed_and_fixed_lr(tmp_path, monkeypatch):
    from ctsc.scripts import run_ablation_reference as reference
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "BallRun.pt").write_bytes(b"dataset path fixture")
    calls = []
    monkeypatch.setattr(reference.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(sys, "argv", ["reference", "--dataset-dir", str(datasets), "--output-dir", str(tmp_path / "reference"),
                                    "--tasks", "BallRun", "--split-seeds", "0", "1"])
    reference.main()
    assert len(calls) == 2
    for seed, command in enumerate(calls):
        assert command[command.index("--seed") + 1] == str(seed)
        assert command[command.index("--lr") + 1] == "0.001"
        assert command[command.index("--epochs") + 1] == "12"
        assert "--split-seed" not in command and "--train-seed" not in command
        assert command[command.index("--output-dir") + 1].endswith(f"seed_{seed}")
    with pytest.raises(FileExistsError):
        reference.main()


def test_ablation_reference_resume_uses_normalized_model_paths(tmp_path, monkeypatch):
    from ctsc.scripts import run_ablation_reference as reference
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    dataset = datasets / "BallRun.pt"
    dataset.write_bytes(b"dataset fixture")
    output = tmp_path / "reference"
    arguments = ["reference", "--dataset-dir", str(datasets), "--output-dir", str(output), "--tasks", "BallRun", "--split-seeds", "0"]
    monkeypatch.setattr(sys, "argv", arguments + ["--prepare-only"])
    reference.main()
    destination = output / "task_runs/BallRun/canonical/seed_0"
    destination.mkdir(parents=True)
    (destination / "suite_summary.json").write_text(json.dumps({"models": {"gru": {}, "flat-mlp": {}}}), encoding="utf-8")
    source = ROOT / "runs/main/cell_runs/canonical_full/BallRun/split_seed_0/models"
    for model in ("gru", "flat_mlp"):
        model_dir = destination / model
        model_dir.mkdir()
        shutil.copyfile(source / model / "attempt-001/metrics.json", model_dir / "metrics.json")
        bundle = model_dir / "precursor_model.pt"
        bundle.write_bytes(b"weights fixture")
        record = json.loads((source / model / "attempt-001/run_manifest.json").read_text(encoding="utf-8"))
        record["training"]["learning_rate"] = 1e-3
        record["protocol"]["legacy_compatible"] = True
        record["protocol"].pop("protocol_hash")
        record["protocol"]["protocol_hash"] = canonical_json_sha256(record["protocol"])
        record["dataset"]["sha256"] = file_sha256(dataset)
        record["artifacts"]["precursor_model.pt"]["sha256"] = file_sha256(bundle)
        record["artifacts"]["precursor_model.pt"]["bytes"] = bundle.stat().st_size
        (model_dir / "run_manifest.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    monkeypatch.setattr(reference.subprocess, "run", lambda *a, **k: pytest.fail("Resume must reuse verified models"))
    reference.main()
    record_path = destination / "flat_mlp/run_manifest.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["training"]["learning_rate"] = 3e-3
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Reference protocol differs"):
        reference.main()
