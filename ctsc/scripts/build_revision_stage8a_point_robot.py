'Audit and adapt the locked FISOR Point Robot file for the unchanged OTTP protocol.'

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctsc.precursor.data import save_precursor_dataset
from ctsc.precursor.public_offline import (
    PublicOfflineProtocolConfig,
    audit_public_task_protocol,
    build_public_precursor_dataset,
)


RAW_SHA256 = "20773a5530cbda57644adac7226812cf492593a617735b4685a969131d37c470"
RAW_NAME = "point_robot-expert-random-100k.hdf5"
TASK_NAME = "FISOR_PointRobot"
SOURCE_COMMIT = "2097e8bbc96ef49183f22cb2a1b05948b5859ec0"
FLOOR_BYTES = 10 * 1024**3
BUILD_RESERVE_BYTES = 256 * 1024**2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_raw(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if path.name != RAW_NAME or file_sha256(path) != RAW_SHA256:
        raise ValueError("Source file or SHA256 differs from locked FISOR Point Robot file")
    expected = {"state", "next_state", "action", "reward", "h", "cost", "done"}
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != expected:
            raise ValueError(f"Unexpected HDF5 fields: {sorted(handle.keys())}")
        arrays = {key: np.asarray(handle[key][()]) for key in expected}
    n = 100_000
    if arrays["state"].shape != (n, 11) or arrays["next_state"].shape != (n, 11):
        raise ValueError("Unexpected observation shape")
    if arrays["action"].shape != (n, 2):
        raise ValueError("Unexpected action shape")
    if any(arr.shape[0] != n or not np.isfinite(arr).all() for arr in arrays.values()):
        raise ValueError("Incomplete or non-finite HDF5 field")
    done = arrays["done"].astype(bool)
    h = arrays["h"]
    violation = arrays["cost"]
    if not np.array_equal(violation, (h > 0).astype(violation.dtype)):
        raise ValueError("Author-provided violation flag is inconsistent with signed h")
    done_indices = np.flatnonzero(done)
    if done_indices.size < 3:
        raise ValueError("Insufficient complete episodes")
    incomplete_tail = n - int(done_indices[-1]) - 1
    if incomplete_tail < 0:
        raise ValueError("Invalid episode boundaries")
    discontinuity = np.max(np.abs(arrays["next_state"][:-1] - arrays["state"][1:]), axis=1)
    if np.any(discontinuity[~done[:-1]] > 1e-5):
        raise ValueError("Unmarked episode discontinuity in nonterminal transitions")
    raw = {
        "observations": arrays["state"],
        "next_observations": arrays["next_state"],
        "actions": arrays["action"],
        "rewards": arrays["reward"],
        "costs": h,  # author's dataset loader maps h -> costs
        "terminals": done,
        "timeouts": np.zeros(n, dtype=np.float32),  # source supplies only done
    }
    audit: dict[str, object] = {
        "source_sha256": RAW_SHA256,
        "source_bytes": path.stat().st_size,
        "source_commit": SOURCE_COMMIT,
        "raw_field_map": {"state": "observations", "action": "actions", "next_state": "next_observations", "reward": "rewards", "h": "costs", "done": "terminals"},
        "cost_semantics": "signed hazard constraint h; positive denotes a violation; author-provided binary cost equals (h > 0)",
        "positive_h_count": int((h > 0).sum()),
        "complete_episode_count": int(done_indices.size),
        "incomplete_tail_dropped": incomplete_tail,
        "unmarked_nonterminal_discontinuities": 0,
        "timeout_field_supplied": False,
        "license_note": "Author officially distributes data for training; no separate redistribution license found; raw file is local-only.",
    }
    return raw, audit


def build(raw_path: Path, output: Path, audit_path: Path) -> dict[str, object]:
    raw_path = raw_path.resolve()
    output = output.resolve()
    audit_path = audit_path.resolve()
    if output.exists() or audit_path.exists():
        raise FileExistsError("Refusing to overwrite an existing G8A asset")
    free_before = shutil.disk_usage(output.parent if output.parent.exists() else raw_path.parent).free
    if free_before < FLOOR_BYTES + BUILD_RESERVE_BYTES:
        raise RuntimeError("Insufficient disk space for precursor build and future results")
    raw, audit = audit_raw(raw_path)
    protocol = PublicOfflineProtocolConfig()
    dataset = build_public_precursor_dataset(raw_dataset=raw, task_name=TASK_NAME, protocol=protocol)
    dataset["metadata"]["external_source"] = {
        "source": "FISOR Point Robot author release, not a DSRL registry dataset",
        "raw_sha256": RAW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "cost_field": "h",
        "episode_boundary_field": "done",
        "incomplete_tail_dropped": audit["incomplete_tail_dropped"],
    }
    protocol_audit = audit_public_task_protocol(dataset, split_seeds=[0, 1, 2], protocol=protocol)
    if not protocol_audit["stable_task"]:
        raise RuntimeError(f"External task fails frozen split/label audit: {protocol_audit['reasons']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    save_precursor_dataset(dataset, output)
    if shutil.disk_usage(output.parent).free < FLOOR_BYTES:
        raise RuntimeError("disk safety floor breached after precursor build")
    audit["precursor_relative_path"] = str(output)
    audit["precursor_sha256"] = file_sha256(output)
    audit["precursor_samples"] = int(len(dataset["costs"]))
    audit["protocol"] = protocol.to_dict()
    audit["split_audit"] = [
        {
            "split_seed": row["split_seed"],
            "pass": row["gate_b_split_pass"],
            "group_leakage_detected": row["group_leakage_detected"],
            "split_sizes": row["split_sizes"],
            "step_thresholds": row["step_thresholds"],
            "outcome_thresholds": row["outcome_thresholds"],
            "labels": row["labels"],
        }
        for row in protocol_audit["split_results"]
    ]
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = build(args.raw, args.output, args.audit)
    print(json.dumps({key: audit[key] for key in ("source_sha256", "precursor_sha256", "precursor_samples", "complete_episode_count", "incomplete_tail_dropped")}, indent=2))


if __name__ == "__main__":
    main()
