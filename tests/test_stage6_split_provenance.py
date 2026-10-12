import csv
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ctsc.precursor.revision import array_sha256, canonical_json_sha256, file_sha256
from ctsc.precursor.stage6 import (
    assert_branch_b_safe, audit_raw_hdf5, audit_split_cell,
    reconstructed_episode_ids, safe_output_dir,
)


def _raw(path: Path, *, candidate: str | None = None, attribute: str | None = None) -> None:
    with h5py.File(path, "w") as handle:
        for key in ("actions", "costs", "next_observations", "observations", "rewards", "terminals", "timeouts"):
            handle.create_dataset(key, data=np.array([0, 1, 0, 1, 0, 1], dtype=np.int64) if key == "terminals" else np.zeros(6))
        if candidate:
            handle.create_dataset(candidate, data=np.zeros(6))
        if attribute:
            handle.attrs[attribute] = "ambiguous"


def test_no_provenance_and_placeholders_are_not_raw_evidence(tmp_path):
    path = tmp_path / "raw.hdf5"
    _raw(path)
    audit = audit_raw_hdf5(path, tmp_path)
    assert audit["provenance_candidates"] == []
    assert audit["root_attribute_names"] == []
    assert audit["sha256"] == file_sha256(path)
    assert_branch_b_safe([audit])
    assert reconstructed_episode_ids(path).tolist() == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("field,attribute", [("collection_run_id", None), (None, "policy_seed"), ("source_shard", None)])
def test_candidate_requires_manual_episode_mapping(tmp_path, field, attribute):
    path = tmp_path / "raw.hdf5"
    _raw(path, candidate=field, attribute=attribute)
    audit = audit_raw_hdf5(path, tmp_path)
    assert audit["provenance_candidates"]
    with pytest.raises(ValueError, match="manual per-episode"):
        assert_branch_b_safe([audit])


def _cell(tmp_path: Path, overlap=False, corrupt=False) -> tuple[Path, np.ndarray]:
    cell = tmp_path / "cell"
    folder = cell / "reproducibility"
    folder.mkdir(parents=True)
    values = {
        "train": np.array([0, 1], dtype=np.int64),
        "validation": np.array([2, 3], dtype=np.int64),
        "test": np.array([0, 5] if overlap else [4, 5], dtype=np.int64),
    }
    np.savez_compressed(folder / "uncapped_split_indices.npz", **values)
    with gzip.open(folder / "episode_group_assignment.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_group_index", "episode_uid", "split"])
        for group, split in enumerate(("train", "val", "test")):
            writer.writerow([group, group, split])
    files = {path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size} for path in folder.iterdir()}
    record = {
        "files": files,
        "uncapped": {name: {"count": arr.size, "sha256": array_sha256(arr)} for name, arr in
                     {"train": values["train"], "val": values["validation"], "test": values["test"]}.items()},
    }
    (cell / "reproducibility_record.json").write_text(json.dumps(record), encoding="utf-8")
    manifest = {
        "task_name": "Fake", "split_seed": 0, "sampling_seed": 0, "train_seed": 0, "loader_seed": 0,
        "condition_id": "canonical_full", "seed_protocol": "explicit_four_stream",
        "cell_sha256": "abc", "shared_reproducibility_record_sha256": canonical_json_sha256(record),
    }
    (cell / "cell_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (cell / "cell_summary.json").write_text(json.dumps({"status": "PASS", "cell_sha256": "abc"}), encoding="utf-8")
    if corrupt:
        (folder / "episode_group_assignment.csv.gz").write_bytes(b"changed")
    return cell, np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)


def test_episode_split_disjoint_and_hashes(tmp_path):
    cell, episodes = _cell(tmp_path)
    result = audit_split_cell(cell, "Fake", 0, episodes, tmp_path)
    assert result["episode_overlap"] == 0
    assert result["conclusion"] == "EPISODE_DISJOINT_ONLY"


@pytest.mark.parametrize("overlap,corrupt,pattern", [(True, False, "overlap"), (False, True, "hash/size")])
def test_overlap_and_hash_mismatch_block(tmp_path, overlap, corrupt, pattern):
    cell, episodes = _cell(tmp_path, overlap=overlap, corrupt=corrupt)
    with pytest.raises(ValueError, match=pattern):
        audit_split_cell(cell, "Fake", 0, episodes, tmp_path)


def test_missing_cell_and_output_guard(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_split_cell(tmp_path / "missing", "Fake", 0, np.array([0]), tmp_path)
    root = tmp_path / "repo"
    output = root / "results/split_audit"
    safe_output_dir(output, root)
    output.mkdir(parents=True)
    (output / "existing.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        safe_output_dir(output, root)
    with pytest.raises(ValueError):
        safe_output_dir(root / "pilot_artifacts", root)


