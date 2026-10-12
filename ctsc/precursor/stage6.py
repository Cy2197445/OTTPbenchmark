'Read-only audit of public-data provenance and episode splits.'

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ctsc.precursor.revision import array_sha256, canonical_json_sha256, file_sha256


CORE_FIELDS = frozenset({
    "actions", "costs", "next_observations", "observations", "rewards",
    "terminals", "timeouts",
})
PROVENANCE_PATTERN = re.compile(
    r"(?:^|[/_\-])(run|policy|seed|checkpoint|ckpt|shard|collector|collection|"
    r"trajectory_source|source_file|batch_id)(?:$|[/_\-])", re.I
)
PLACEHOLDER_FIELDS = {
    "policy_index": "constant zero added by precursor; not a behavior-policy ID",
    "checkpoint_index": "constant zero added by precursor; not a checkpoint ID",
    "run_group_index": "legacy alias for reconstructed episode group; not a logging-run ID",
    "env_index": "constant zero added by precursor; not provenance",
}


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def audit_raw_hdf5(path: Path, root: Path) -> dict[str, Any]:
    'Inventory *all* datasets/attributes without interpreting candidate IDs.'
    if not path.is_file():
        raise FileNotFoundError(path)
    objects: list[dict[str, Any]] = []
    candidates: list[str] = []
    with h5py.File(path, "r") as handle:
        def record(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            item: dict[str, Any] = {
                "path": "/" + name,
                "kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                "attribute_names": sorted(str(key) for key in obj.attrs.keys()),
            }
            if isinstance(obj, h5py.Dataset):
                item.update(shape=list(obj.shape), dtype=str(obj.dtype))
            objects.append(item)
            for term in [name, *(str(key) for key in obj.attrs.keys())]:
                if PROVENANCE_PATTERN.search(term):
                    candidates.append("/" + name + ":" + term)

        root_attrs = sorted(str(key) for key in handle.attrs.keys())
        for name in root_attrs:
            if PROVENANCE_PATTERN.search(name):
                candidates.append("/:" + name)
        handle.visititems(record)
        top_fields = {str(key) for key in handle.keys() if isinstance(handle[key], h5py.Dataset)}
        if not CORE_FIELDS.issubset(top_fields):
            raise ValueError(f"Missing core fields in {path}: {sorted(CORE_FIELDS - top_fields)}")
        n = int(handle["terminals"].shape[0])
        if int(handle["timeouts"].shape[0]) != n:
            raise ValueError(f"Terminal/timeout length mismatch: {path}")
    return {
        "path": _rel(path, root), "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size), "raw_transitions": n,
        "root_attribute_names": root_attrs, "objects": objects,
        "provenance_candidates": sorted(set(candidates)),
        "mapping_status": "REQUIRES_MANUAL_EPISODE_MAPPING" if candidates else "NO_CANDIDATE_FIELD",
    }


def reconstructed_episode_ids(path: Path) -> np.ndarray:
    'Reproduce public_offline.py boundary and incomplete-tail semantics.'
    with h5py.File(path, "r") as handle:
        terminal = np.asarray(handle["terminals"][:]).reshape(-1) > 0
        timeout = np.asarray(handle["timeouts"][:]).reshape(-1) > 0
    ends = terminal | timeout
    if not ends.size:
        return np.empty(0, dtype=np.int64)
    starts = np.empty(ends.size, dtype=bool)
    starts[0] = True
    starts[1:] = ends[:-1]
    ids = np.cumsum(starts.astype(np.int64)) - 1
    if not ends[-1]:
        ids = ids[ids != ids[-1]]
    return ids


def audit_split_cell(cell: Path, task: str, seed: int, episode_ids: np.ndarray, root: Path) -> dict[str, Any]:
    manifest_path = cell / "cell_manifest.json"
    record_path = cell / "reproducibility_record.json"
    summary_path = cell / "cell_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (manifest.get("task_name"), manifest.get("split_seed"), manifest.get("condition_id")) != (task, seed, "canonical_full"):
        raise ValueError(f"Cell identity mismatch: {cell}")
    if any(manifest.get(f"{key}_seed") != seed for key in ("split", "sampling", "train", "loader")):
        raise ValueError(f"Four-stream seed mismatch: {cell}")
    if manifest.get("seed_protocol") != "explicit_four_stream" or summary.get("status") != "PASS":
        raise ValueError(f"Protocol/status mismatch: {cell}")
    if summary.get("cell_sha256") != manifest.get("cell_sha256"):
        raise ValueError(f"Cell hash mismatch: {cell}")
    if canonical_json_sha256(record) != manifest.get("shared_reproducibility_record_sha256"):
        raise ValueError(f"Reproducibility record hash mismatch: {cell}")
    for filename, value in record["files"].items():
        artifact = cell / "reproducibility" / filename
        if file_sha256(artifact) != value["sha256"] or artifact.stat().st_size != value["bytes"]:
            raise ValueError(f"Split artifact hash/size mismatch: {artifact}")

    split_file = cell / "reproducibility" / "uncapped_split_indices.npz"
    with np.load(split_file) as split_npz:
        indices = {name: np.asarray(split_npz[name], dtype=np.int64) for name in ("train", "validation", "test")}
    labels = np.full(episode_ids.shape[0], -1, dtype=np.int8)
    episode_sets: dict[str, set[int]] = {}
    for number, name in enumerate(("train", "validation", "test")):
        values = indices[name]
        stored = record["uncapped"]["val" if name == "validation" else name]
        if int(stored["count"]) != values.size or array_sha256(values) != stored["sha256"]:
            raise ValueError(f"Split index record mismatch: {cell}/{name}")
        if values.size == 0 or values.min() < 0 or values.max() >= episode_ids.size:
            raise ValueError(f"Empty/out-of-bounds split: {cell}/{name}")
        if np.unique(values).size != values.size or np.any(labels[values] != -1):
            raise ValueError(f"Transition overlap/duplication: {cell}/{name}")
        labels[values] = number
        episode_sets[name] = set(np.unique(episode_ids[values]).tolist())
    if np.any(labels == -1):
        raise ValueError(f"Split does not cover all retained transitions: {cell}")
    names = list(episode_sets)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if episode_sets[first] & episode_sets[second]:
                raise ValueError(f"Episode overlap: {cell}/{first}/{second}")
    assignment_path = cell / "reproducibility" / "episode_group_assignment.csv.gz"
    seen: set[int] = set()
    with gzip.open(assignment_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["episode_group_index", "episode_uid", "split"]:
            raise ValueError(f"Unexpected assignment schema: {cell}")
        for row in reader:
            group, episode = int(row["episode_group_index"]), int(row["episode_uid"])
            name = "validation" if row["split"] == "val" else row["split"]
            if group != episode or episode in seen or name not in episode_sets or episode not in episode_sets[name]:
                raise ValueError(f"Episode assignment mismatch: {cell}/{episode}")
            seen.add(episode)
    if seen != set(np.unique(episode_ids).tolist()):
        raise ValueError(f"Missing episode assignment: {cell}")
    return {
        "task": task, "split_seed": seed, "cell_path": _rel(cell, root),
        "cell_manifest_sha256": file_sha256(manifest_path),
        "reproducibility_record_sha256": file_sha256(record_path),
        "uncapped_split_sha256": file_sha256(split_file),
        "assignment_sha256": file_sha256(assignment_path),
        "retained_transitions": int(episode_ids.size),
        "transition_counts": {name: int(values.size) for name, values in indices.items()},
        "episode_counts": {name: len(values) for name, values in episode_sets.items()},
        "episode_overlap": 0, "conclusion": "EPISODE_DISJOINT_ONLY",
    }


def assert_branch_b_safe(raw_audits: list[dict[str, Any]]) -> None:
    unresolved = [row["path"] for row in raw_audits if row["provenance_candidates"]]
    if unresolved:
        raise ValueError(f"Provenance candidates require manual per-episode adjudication: {unresolved}")


def safe_output_dir(output: Path, root: Path) -> None:
    resolved = output.resolve()
    base = (root / "results/split_audit").resolve()
    if resolved != base:
        raise ValueError(f"Split-audit output must be isolated at {base}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing nonempty Stage 6 output: {resolved}")
