'protocol, provenance, and release helpers.'

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import csv
import gzip
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ctsc.precursor.data import assign_3way_labels, compute_label_thresholds, resolve_episode_group_index


REVISION_SCHEMA_VERSION = "ottp_revision_run_manifest_v1"
FREEZE_ID = "OTTP-R26-G1-v1"
MIN_FREE_SPACE_GIB = 10.0
GIB = 1024**3
AUTHORITATIVE_RESULT_ROOTS = (
    Path("pilot_artifacts/Unified38_fresh_rerun_active"),
    Path("pilot_artifacts/Unified38_current_only_matrix"),
    Path("runs"),
    Path("results"),
    Path("manifests"),
    Path("provenance"),
    Path("ctsc"),
    Path("tests"),
    Path("docs"),
    Path("release_tools"),
)
PROTOCOL_IDENTITY_FIELDS = (
    "target_variant",
    "gamma",
    "quantile_pair",
    "outcome_quantile_pair",
    "split_family",
    "split_seed",
    "sampling_seed",
    "train_seed",
    "loader_seed",
    "feature_profile",
    "model_name",
)


@dataclass(frozen=True)
class SeedBundle:
    'Four independent random streams plus the legacy compatibility mode.'

    split_seed: int
    sampling_seed: int
    train_seed: int
    loader_seed: int
    legacy_seed: int
    legacy_compatible: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def resolve_seed_bundle(
    *,
    seed: int = 0,
    split_seed: int | None = None,
    sampling_seed: int | None = None,
    train_seed: int | None = None,
    loader_seed: int | None = None,
) -> SeedBundle:
    'Resolve explicit seeds while preserving the one-seed legacy contract.'
    legacy = int(seed)
    explicit = any(value is not None for value in (split_seed, sampling_seed, train_seed, loader_seed))
    resolved_split = legacy if split_seed is None else int(split_seed)
    resolved_sampling = resolved_split if sampling_seed is None else int(sampling_seed)
    resolved_train = legacy if train_seed is None else int(train_seed)
    resolved_loader = resolved_train if loader_seed is None else int(loader_seed)
    return SeedBundle(
        split_seed=resolved_split,
        sampling_seed=resolved_sampling,
        train_seed=resolved_train,
        loader_seed=resolved_loader,
        legacy_seed=legacy,
        legacy_compatible=not explicit,
    )


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    values = np.ascontiguousarray(np.asarray(array))
    digest = sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(canonical_json_bytes({"shape": list(values.shape)}))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def arrays_sha256(dataset: Mapping[str, Any], fields: Sequence[str]) -> str:
    digest = sha256()
    for field in fields:
        digest.update(str(field).encode("utf-8"))
        digest.update(array_sha256(dataset[field]).encode("ascii"))
    return digest.hexdigest()


def protocol_identity(
    *,
    target_variant: str,
    gamma: float,
    quantile_pair: Sequence[float],
    outcome_quantile_pair: Sequence[float] | None = None,
    split_family: str,
    seeds: SeedBundle,
    feature_profile: str,
    model_name: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if target_variant not in {"current_inclusive", "future_only"}:
        raise ValueError(f"Unsupported target_variant={target_variant!r}.")
    if len(tuple(quantile_pair)) != 2:
        raise ValueError("quantile_pair must contain exactly two values.")
    low, high = (float(value) for value in quantile_pair)
    if not 0.0 < low < high < 1.0:
        raise ValueError(f"Invalid quantile_pair={[low, high]}.")
    resolved_outcome_pair = quantile_pair if outcome_quantile_pair is None else outcome_quantile_pair
    if len(tuple(resolved_outcome_pair)) != 2:
        raise ValueError("outcome_quantile_pair must contain exactly two values.")
    outcome_low, outcome_high = (float(value) for value in resolved_outcome_pair)
    if not 0.0 < outcome_low < outcome_high < 1.0:
        raise ValueError(f"Invalid outcome_quantile_pair={[outcome_low, outcome_high]}.")
    payload: dict[str, Any] = {
        "freeze_id": FREEZE_ID,
        "target_variant": str(target_variant),
        "gamma": float(gamma),
        "quantile_pair": [low, high],
        "outcome_quantile_pair": [outcome_low, outcome_high],
        "split_family": str(split_family),
        **seeds.to_dict(),
        "feature_profile": str(feature_profile),
        "model_name": str(model_name),
    }
    if extra:
        payload["extra"] = dict(extra)
    payload["protocol_hash"] = canonical_json_sha256(payload)
    return payload


def aggregation_identity(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    'Return a collision-resistant aggregation key for revision results.'
    return tuple(
        tuple(payload[field]) if field in {"quantile_pair", "outcome_quantile_pair"} else payload[field]
        for field in PROTOCOL_IDENTITY_FIELDS
    )


def validate_run_manifest(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "freeze_id",
        "experiment_id",
        "attempt_id",
        "work_package",
        "source_snapshot_hash",
        "dataset",
        "protocol",
        "indices",
        "artifacts",
    }
    missing = sorted(required - set(payload))
    if missing:
        errors.append(f"missing required keys: {missing}")
        return errors
    if payload.get("schema_version") != REVISION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {REVISION_SCHEMA_VERSION!r}")
    if payload.get("freeze_id") != FREEZE_ID:
        errors.append(f"freeze_id must be {FREEZE_ID!r}")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        errors.append("protocol must be an object")
    else:
        for field in PROTOCOL_IDENTITY_FIELDS:
            if field not in protocol:
                errors.append(f"protocol missing {field}")
        if "protocol_hash" in protocol:
            without_hash = dict(protocol)
            reported = str(without_hash.pop("protocol_hash"))
            expected = canonical_json_sha256(without_hash)
            if reported != expected:
                errors.append("protocol_hash mismatch")
    indices = payload.get("indices")
    if not isinstance(indices, Mapping) or not {"uncapped", "capped"}.issubset(indices):
        errors.append("indices must contain uncapped and capped records")
    return errors


def assert_valid_run_manifest(payload: Mapping[str, Any]) -> None:
    errors = validate_run_manifest(payload)
    if errors:
        raise ValueError("Invalid revision run manifest: " + "; ".join(errors))


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def guard_revision_output_dir(
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    allow_existing: bool = False,
) -> Path:
    'Reject authoritative or already-populated output targets.'
    root = _resolved(repo_root)
    output = _resolved(output_dir)
    for relative in AUTHORITATIVE_RESULT_ROOTS:
        authoritative = _resolved(root / relative)
        if is_within(output, authoritative):
            raise PermissionError(f"Revision output may not write into authoritative results: {authoritative}")
    if output.exists() and not allow_existing:
        protected_names = {"metrics.json", "precursor_model.pt", "run_manifest.json", "suite_summary.json"}
        collisions = sorted(path.name for path in output.iterdir() if path.name in protected_names)
        if collisions:
            raise FileExistsError(f"Refusing to overwrite existing run artifacts in {output}: {collisions}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def seeded_output_path(base_dir: str | Path, seeds: SeedBundle) -> Path:
    'Return the mandatory collision-free path for an explicit-seed run.'
    base = Path(base_dir)
    expected_tail = (f"split_seed_{seeds.split_seed}", f"train_seed_{seeds.train_seed}")
    if len(base.parts) >= 2 and tuple(base.parts[-2:]) == expected_tail:
        return base
    return base / expected_tail[0] / expected_tail[1]


def source_snapshot_sha256(repo_root: str | Path, relative_paths: Iterable[str]) -> str:
    'Hash an explicit active-code allowlist when the workspace has no Git commit.'
    root = _resolved(repo_root)
    records: list[dict[str, str]] = []
    for relative_text in sorted(str(value).replace("\\", "/") for value in relative_paths):
        path = _resolved(root / relative_text)
        if not is_within(path, root) or not path.is_file():
            raise FileNotFoundError(f"Invalid source snapshot member: {relative_text}")
        records.append({"path": relative_text, "sha256": file_sha256(path)})
    return canonical_json_sha256({"files": records})


def require_free_space(
    path: str | Path,
    *,
    minimum_gib: float = MIN_FREE_SPACE_GIB,
    observed_free_bytes: int | None = None,
) -> int:
    'Stop a job before it can violate the frozen disk-space floor.'
    target = Path(path).resolve()
    probe = target if target.exists() else target.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = int(observed_free_bytes) if observed_free_bytes is not None else int(shutil.disk_usage(probe).free)
    threshold = int(float(minimum_gib) * GIB)
    if free_bytes < threshold:
        raise OSError(
            f"Insufficient free space for revision job: {free_bytes / GIB:.2f} GiB available; "
            f"minimum is {float(minimum_gib):.2f} GiB."
        )
    return free_bytes


def _jsonable_thresholds(payload: Mapping[int, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    return {
        str(int(env_id)): {str(key): float(value) for key, value in values.items()}
        for env_id, values in payload.items()
    }


def _index_record(indices: Any) -> dict[str, Any]:
    values = np.asarray(indices, dtype=np.int64)
    return {"count": int(values.size), "sha256": array_sha256(values)}


def export_reproducibility_artifacts(
    *,
    output_dir: str | Path,
    dataset: Mapping[str, Any],
    uncapped_splits: Mapping[str, Any],
    capped_splits: Mapping[str, Any],
    step_thresholds: Mapping[int, Mapping[str, float]],
    outcome_thresholds: Mapping[int, Mapping[str, float]],
    normalization_mean: Any,
    normalization_std: Any,
) -> dict[str, Any]:
    'Export compact exact indices and preprocessing state without JSON bloat.'
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    uncapped_path = target / "uncapped_split_indices.npz"
    capped_path = target / "capped_train_validation_indices.npz"
    normalization_path = target / "normalization_stats.npz"
    group_path = target / "episode_group_assignment.csv.gz"
    thresholds_path = target / "thresholds.json"

    np.savez_compressed(
        uncapped_path,
        train=np.asarray(uncapped_splits["train"], dtype=np.int64),
        validation=np.asarray(uncapped_splits["val"], dtype=np.int64),
        test=np.asarray(uncapped_splits["test"], dtype=np.int64),
    )
    np.savez_compressed(
        capped_path,
        train=np.asarray(capped_splits["train"], dtype=np.int64),
        validation=np.asarray(capped_splits["val"], dtype=np.int64),
    )
    np.savez_compressed(
        normalization_path,
        mean=np.asarray(normalization_mean, dtype=np.float32),
        std=np.asarray(normalization_std, dtype=np.float32),
    )
    groups = resolve_episode_group_index(dict(dataset))
    episode_uids = np.asarray(dataset["episode_uid"], dtype=np.int64)
    group_to_split: dict[int, str] = {}
    for split_name in ("train", "val", "test"):
        for index in np.asarray(uncapped_splits[split_name], dtype=np.int64):
            group_to_split[int(groups[int(index)])] = split_name
    with gzip.open(group_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_group_index", "episode_uid", "split"])
        for group_id in sorted(np.unique(groups).astype(np.int64).tolist()):
            members = np.flatnonzero(groups == int(group_id))
            episode_id = int(episode_uids[int(members[0])]) if members.size else -1
            writer.writerow([int(group_id), episode_id, group_to_split.get(int(group_id), "")])
    thresholds_payload = {
        "step": _jsonable_thresholds(step_thresholds),
        "outcome": _jsonable_thresholds(outcome_thresholds),
    }
    thresholds_path.write_text(json.dumps(thresholds_payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "root": str(target),
        "uncapped": {name: _index_record(uncapped_splits[name]) for name in ("train", "val", "test")},
        "capped": {name: _index_record(capped_splits[name]) for name in ("train", "val", "test")},
        "files": {
            path.name: {"sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
            for path in (uncapped_path, capped_path, normalization_path, group_path, thresholds_path)
        },
    }


def derive_discounted_target(
    costs: Any,
    episode_uids: Any,
    *,
    gamma: float,
    target_variant: str,
) -> np.ndarray:
    costs_array = np.asarray(costs, dtype=np.float32).reshape(-1)
    episodes = np.asarray(episode_uids, dtype=np.int64).reshape(-1)
    if costs_array.shape[0] != episodes.shape[0]:
        raise ValueError("costs and episode_uids must have equal length")
    if target_variant not in {"current_inclusive", "future_only"}:
        raise ValueError(f"Unsupported target_variant={target_variant!r}")
    current = np.zeros(costs_array.shape[0], dtype=np.float32)
    for episode_id in np.unique(episodes):
        indices = np.flatnonzero(episodes == int(episode_id))
        running = 0.0
        for index in indices[::-1]:
            running = float(costs_array[int(index)]) + float(gamma) * running
            current[int(index)] = running
    if target_variant == "current_inclusive":
        return current
    future = np.zeros_like(current)
    for episode_id in np.unique(episodes):
        indices = np.flatnonzero(episodes == int(episode_id))
        if indices.size > 1:
            future[indices[:-1]] = current[indices[1:]]
        if indices.size:
            future[int(indices[-1])] = 0.0
    return future


def build_label_sidecar(
    *,
    dataset: Mapping[str, Any],
    base_dataset_path: str | Path,
    train_indices: Any,
    output_path: str | Path,
    target_variant: str,
    gamma: float,
    quantile_pair: Sequence[float],
    precomputed_base_dataset_sha256: str | None = None,
    precomputed_base_arrays_sha256: str | None = None,
) -> dict[str, Any]:
    'Write only target/label arrays and provenance; never duplicate base inputs.'
    low, high = (float(value) for value in quantile_pair)
    target_values = derive_discounted_target(
        dataset["costs"],
        dataset["episode_uid"],
        gamma=float(gamma),
        target_variant=str(target_variant),
    )
    label_dataset = {
        "revision_target": target_values,
        "env_index": np.asarray(dataset["env_index"], dtype=np.int64),
    }
    thresholds = compute_label_thresholds(
        label_dataset,
        np.asarray(train_indices, dtype=np.int64),
        value_key="revision_target",
        lower_quantile=low,
        upper_quantile=high,
    )
    labels = assign_3way_labels(target_values, label_dataset["env_index"], thresholds)
    counts = np.bincount(labels, minlength=3).astype(np.int64)
    ratios = counts.astype(np.float64) / max(int(counts.sum()), 1)
    base_fields = ("obs", "actions", "costs", "episode_uid", "episode_position", "env_index")
    resolved_dataset_sha256 = str(precomputed_base_dataset_sha256 or file_sha256(base_dataset_path))
    resolved_arrays_sha256 = str(precomputed_base_arrays_sha256 or arrays_sha256(dataset, base_fields))
    for digest_name, digest_value in (
        ("base_dataset_sha256", resolved_dataset_sha256),
        ("base_arrays_sha256", resolved_arrays_sha256),
    ):
        if len(digest_value) != 64 or any(character not in "0123456789abcdef" for character in digest_value.lower()):
            raise ValueError(f"Invalid cached {digest_name}: {digest_value!r}")
    sidecar = Path(output_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar,
        target_values=target_values.astype(np.float32),
        labels=labels.astype(np.uint8),
    )
    manifest = {
        "schema_version": "ottp_label_sidecar_v1",
        "freeze_id": FREEZE_ID,
        "target_variant": str(target_variant),
        "gamma": float(gamma),
        "quantile_pair": [low, high],
        "train_indices_sha256": array_sha256(np.asarray(train_indices, dtype=np.int64)),
        "thresholds": _jsonable_thresholds(thresholds),
        "class_counts": counts.tolist(),
        "class_ratios": ratios.tolist(),
        "base_dataset_path": str(Path(base_dataset_path)),
        "base_dataset_sha256": resolved_dataset_sha256,
        "base_arrays_sha256": resolved_arrays_sha256,
        "base_array_fields": list(base_fields),
        "sidecar_path": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "sidecar_bytes": int(sidecar.stat().st_size),
        "contains_base_observation_or_action_arrays": False,
    }
    manifest_path = sidecar.with_suffix(sidecar.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


