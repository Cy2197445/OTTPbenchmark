from pathlib import Path
import json
import subprocess
import sys

import pytest
import yaml

from ctsc.scripts.run_benchmark import output_directory, selected_plan
from release_tools.verify_release import compare

ROOT = Path(__file__).resolve().parents[1]


def test_main_matrix_matches_recorded_protocol():
    defaults, cells = selected_plan()
    assert len(cells) == 342
    assert sum(len(c["models"]) for c in cells) == 1026
    assert defaults["epochs"] == 12
    assert defaults["window_size"] == 32
    assert defaults["max_train_samples"] == 100000
    assert all(c["model_learning_rates"] == {"gru": 3e-4, "flat_mlp": 3e-3, "transformer": 3e-3} for c in cells)


def test_matrix_selection_and_protected_output(tmp_path):
    _, cells = selected_plan(["BallRun"], [0], ["canonical_full"])
    assert len(cells) == 1
    assert output_directory(tmp_path / "output") == tmp_path / "output"
    for path in (ROOT, ROOT.parent, ROOT / "runs/main", ROOT / "results/main"):
        with pytest.raises(ValueError):
            output_directory(path)
    for selection in (["unknown"], []):
        with pytest.raises(ValueError):
            selected_plan(selection)


def test_citation_and_dataset_metadata():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["license"] == "MIT"
    assert citation["version"] == "1.1.0"
    assert [r["family-names"] for r in citation["authors"]] == ["Chai", "Liu", "Luo"]
    catalog = json.loads((ROOT / "manifests/datasets.json").read_text(encoding="utf-8"))
    assert len(catalog["tasks"]) == 38
    assert all(len(r["sha256"]) == 64 and r["bytes"] > 0 for r in catalog["tasks"])


def test_numeric_comparison_rejects_changed_results():
    compare([{"task": "x", "score": 0.5}], [{"task": "x", "score": "0.5"}], ("task",))
    with pytest.raises(ValueError):
        compare([{"task": "x", "score": 0.6}], [{"task": "x", "score": "0.5"}], ("task",))
    with pytest.raises(ValueError, match="Duplicate"):
        compare([{"task": "x", "score": 0.5}] * 2, [{"task": "x", "score": "0.5"}], ("task",))
    with pytest.raises(ValueError, match="Missing numeric"):
        compare([{"task": "x", "score": 0.5}], [{"task": "x"}], ("task",))


@pytest.mark.parametrize("script", ["ctsc/scripts/run_benchmark.py", "ctsc/scripts/analyze_benchmark.py", "ctsc/scripts/run_ablation_reference.py", "ctsc/scripts/run_additional.py", "ctsc/scripts/download_unified38_public_data.py", "ctsc/scripts/build_public_precursor_dataset.py", "ctsc/scripts/build_benchmark_datasets.py", "ctsc/scripts/train_precursor_model.py", "ctsc/scripts/fetch_revision_stage8a_point_robot.py", "ctsc/scripts/build_revision_stage8a_point_robot.py", "ctsc/scripts/run_revision_stage9_utility.py", "ctsc/scripts/run_Unified38_target_definition_ablation.py", "ctsc/scripts/run_Unified38_split_family_counterfactual.py", "ctsc/scripts/run_Unified38_anti_shortcut_suite.py", "ctsc/scripts/run_Unified38_gru_profile_matrix.py", "ctsc/scripts/run_Unified38_gru_window_sweep.py", "release_tools/verify_release.py"])
def test_public_command_help(script):
    result = subprocess.run([sys.executable, "-B", script, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
