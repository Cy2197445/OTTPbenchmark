'Public offline safe-RL dataset helpers for the Unified38 mainline.'

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ctsc.precursor.data import (
    assign_3way_labels,
    build_group_splits,
    compute_label_thresholds,
    resolve_episode_group_index,
)


_RAW_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "observations": ("observations", "obs"),
    "next_observations": ("next_observations", "next_obs"),
    "actions": ("actions",),
    "rewards": ("rewards",),
    "costs": ("costs", "cost"),
    "terminals": ("terminals", "terminated", "dones", "done"),
    "timeouts": ("timeouts", "truncated", "timeout"),
}


@dataclass(frozen=True)
class PublicOfflineProtocolConfig:
    'Fixed public protocol used across Unified38 audits and rebuilds.'

    gamma: float = 0.99
    group_size_episodes: int = 1
    step_q_low: float = 0.75
    step_q_high: float = 0.90
    outcome_q_low: float = 0.75
    outcome_q_high: float = 0.90
    min_class_ratio: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flatten_vector(array: Any, key: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 0:
        raise ValueError(f"Dataset field `{key}` must not be scalar.")
    return values.reshape(-1)


def _ensure_matrix(array: Any, key: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"Dataset field `{key}` must be 2D after normalization, got shape={values.shape}.")
    return values.astype(np.float32, copy=False)


def _extract_raw_field(raw_dataset: dict[str, Any], canonical_key: str) -> Any:
    aliases = _RAW_KEY_ALIASES[canonical_key]
    for alias in aliases:
        if alias in raw_dataset:
            return raw_dataset[alias]
    raise KeyError(f"Required dataset field missing: `{canonical_key}`. Supported aliases={aliases}.")


def _unwrap_dataset_container(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "dataset" in payload and isinstance(payload["dataset"], dict):
        return payload["dataset"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported dataset container type: {type(payload)!r}")
    return payload


def _load_hdf5_dataset(path: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "Reading .h5/.hdf5 public datasets requires `h5py`. "
            "Install it via `pip install -r requirements_public_offline_release.txt`."
        ) from exc

    payload: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            item = handle[key]
            if isinstance(item, h5py.Dataset):
                payload[str(key)] = item[()]
            elif isinstance(item, h5py.Group):
                payload[str(key)] = {sub_key: item[sub_key][()] for sub_key in item.keys()}
    return payload


def _load_local_raw_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as handle:
            payload = {key: handle[key] for key in handle.files}
    elif suffix in {".h5", ".hdf5"}:
        payload = _load_hdf5_dataset(path)
    else:
        raise ValueError(f"Unsupported dataset file format: {path}")
    return _unwrap_dataset_container(payload)


def load_public_offline_raw_dataset(
    *,
    env_id: str | None = None,
    dataset_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    'Load a D4RL/DSRL-style raw dataset from a local file or DSRL env id.'
    if env_id is None and dataset_path is None:
        raise ValueError("Either `env_id` or `dataset_path` must be provided.")

    if env_id is not None:
        import gymnasium as gym

        try:
            import dsrl  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise ImportError(
                "Loading datasets via `env_id` requires the optional `dsrl` package. "
                "Install it via `pip install -r requirements_public_offline_release.txt`."
            ) from exc

        env = gym.make(str(env_id))
        try:
            if dataset_path is not None:
                try:
                    payload = env.get_dataset(str(Path(dataset_path)))
                except TypeError:
                    payload = env.get_dataset(filepath=str(Path(dataset_path)))
            else:
                payload = env.get_dataset()
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        return _unwrap_dataset_container(payload), {
            "source_type": "dsrl_env",
            "source_name": str(env_id),
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
        }

    payload = _load_local_raw_dataset(Path(dataset_path))
    return payload, {
        "source_type": "local_file",
        "source_name": str(Path(dataset_path).name),
        "dataset_path": str(Path(dataset_path)),
    }


def _build_episode_ids(terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
    ends = np.logical_or(terminated.reshape(-1) > 0, truncated.reshape(-1) > 0)
    start_flags = np.concatenate(([True], ends[:-1]))
    return np.cumsum(start_flags.astype(np.int64)) - 1


def _derive_episode_scalars(costs: np.ndarray, episode_ids: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
    size = int(costs.shape[0])
    if size <= 0:
        return {
            "cost_to_go_disc": np.zeros((0,), dtype=np.float32),
            "episode_cost_disc": np.zeros((0,), dtype=np.float32),
            "env_step": np.zeros((0,), dtype=np.float32),
        }

    cost_to_go_disc = np.zeros(size, dtype=np.float32)
    episode_cost_disc = np.zeros(size, dtype=np.float32)
    env_step = np.zeros(size, dtype=np.float32)
    num_episodes = int(episode_ids.max()) + 1 if episode_ids.size else 0
    for episode_id in range(num_episodes):
        indices = np.flatnonzero(episode_ids == episode_id)
        if indices.size <= 0:
            continue
        env_step[indices] = np.arange(1, indices.size + 1, dtype=np.float32)
        episode_cost = float(np.sum(costs[indices], dtype=np.float64))
        episode_cost_disc[indices] = episode_cost
        running = 0.0
        for index in indices[::-1]:
            running = float(costs[index]) + float(gamma) * running
            cost_to_go_disc[index] = running
    return {
        "cost_to_go_disc": cost_to_go_disc,
        "episode_cost_disc": episode_cost_disc,
        "env_step": env_step,
    }


def _trim_incomplete_tail_episode(
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    rewards: np.ndarray,
    costs: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    episode_ids = _build_episode_ids(terminated, truncated)
    keep_mask = np.ones(obs.shape[0], dtype=bool)
    if episode_ids.size > 0 and not bool(terminated[-1] > 0 or truncated[-1] > 0):
        keep_mask = episode_ids != int(episode_ids[-1])

    payload = {
        "obs": obs[keep_mask],
        "actions": actions[keep_mask],
        "next_obs": next_obs[keep_mask],
        "rewards": rewards[keep_mask],
        "costs": costs[keep_mask],
        "terminated": terminated[keep_mask],
        "truncated": truncated[keep_mask],
    }
    trimmed_episode_ids = _build_episode_ids(payload["terminated"], payload["truncated"])
    payload.update(_derive_episode_scalars(payload["costs"], trimmed_episode_ids, gamma=float(gamma)))
    return payload, trimmed_episode_ids


def _episode_positions_and_lengths(episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    episode_count = int(episode_ids.max()) + 1 if episode_ids.size else 0
    lengths = np.bincount(episode_ids, minlength=episode_count).astype(np.int64)
    positions = np.zeros_like(episode_ids, dtype=np.int64)
    for episode_id in range(episode_count):
        mask = episode_ids == episode_id
        positions[mask] = np.arange(int(mask.sum()), dtype=np.int64)
    return positions, lengths


def build_public_precursor_dataset(
    *,
    raw_dataset: dict[str, Any] | None = None,
    env_id: str | None = None,
    dataset_path: str | Path | None = None,
    task_name: str | None = None,
    protocol: PublicOfflineProtocolConfig | None = None,
) -> dict[str, Any]:
    'Convert a public offline safe-RL dataset into the precursor dataset schema.'
    protocol = protocol or PublicOfflineProtocolConfig()
    if int(protocol.group_size_episodes) <= 0:
        raise ValueError(f"group_size_episodes must be positive, got {protocol.group_size_episodes}")

    if raw_dataset is None:
        raw_dataset, source_meta = load_public_offline_raw_dataset(env_id=env_id, dataset_path=dataset_path)
    else:
        raw_dataset = _unwrap_dataset_container(raw_dataset)
        source_meta = {
            "source_type": "in_memory",
            "source_name": str(task_name or env_id or "in_memory_public_dataset"),
            "dataset_path": None,
        }

    task_label = str(task_name or env_id or (Path(dataset_path).stem if dataset_path is not None else "public_offline_task"))

    obs = _ensure_matrix(_extract_raw_field(raw_dataset, "observations"), "observations")
    next_obs = _ensure_matrix(_extract_raw_field(raw_dataset, "next_observations"), "next_observations")
    actions = _ensure_matrix(_extract_raw_field(raw_dataset, "actions"), "actions")
    rewards = _flatten_vector(_extract_raw_field(raw_dataset, "rewards"), "rewards").astype(np.float32, copy=False)
    costs = _flatten_vector(_extract_raw_field(raw_dataset, "costs"), "costs").astype(np.float32, copy=False)
    terminated = _flatten_vector(_extract_raw_field(raw_dataset, "terminals"), "terminals").astype(np.float32, copy=False)
    truncated = _flatten_vector(_extract_raw_field(raw_dataset, "timeouts"), "timeouts").astype(np.float32, copy=False)

    size = int(obs.shape[0])
    expected_lengths = {
        "next_observations": int(next_obs.shape[0]),
        "actions": int(actions.shape[0]),
        "rewards": int(rewards.shape[0]),
        "costs": int(costs.shape[0]),
        "terminals": int(terminated.shape[0]),
        "timeouts": int(truncated.shape[0]),
    }
    mismatched = {key: value for key, value in expected_lengths.items() if value != size}
    if mismatched:
        raise ValueError(f"Public dataset fields have inconsistent lengths: obs={size}, mismatched={mismatched}")

    trimmed_payload, episode_ids = _trim_incomplete_tail_episode(
        obs=obs,
        actions=actions,
        next_obs=next_obs,
        rewards=rewards,
        costs=costs,
        terminated=terminated,
        truncated=truncated,
        gamma=float(protocol.gamma),
    )
    positions, lengths = _episode_positions_and_lengths(episode_ids)
    episode_uids = episode_ids.astype(np.int64)
    steps_remaining = lengths[episode_ids] - positions - 1 if episode_ids.size else np.zeros((0,), dtype=np.int64)
    run_group_index = (episode_uids // int(protocol.group_size_episodes)).astype(np.int64) if episode_uids.size else np.zeros((0,), dtype=np.int64)
    unique_run_groups = np.unique(run_group_index).astype(np.int64).tolist()
    run_group_names = {int(index): f"{task_label}_episode_group_{int(index)}" for index in unique_run_groups}

    sample_count = int(trimmed_payload["obs"].shape[0])
    return {
        "obs": np.asarray(trimmed_payload["obs"], dtype=np.float32),
        "actions": np.asarray(trimmed_payload["actions"], dtype=np.float32),
        "next_obs": np.asarray(trimmed_payload["next_obs"], dtype=np.float32),
        "rewards": np.asarray(trimmed_payload["rewards"], dtype=np.float32),
        "costs": np.asarray(trimmed_payload["costs"], dtype=np.float32),
        "terminated": np.asarray(trimmed_payload["terminated"], dtype=np.float32),
        "truncated": np.asarray(trimmed_payload["truncated"], dtype=np.float32),
        "cost_to_go_disc": np.asarray(trimmed_payload["cost_to_go_disc"], dtype=np.float32),
        "episode_cost_disc": np.asarray(trimmed_payload["episode_cost_disc"], dtype=np.float32),
        "env_step": np.asarray(trimmed_payload["env_step"], dtype=np.float32),
        "episode_uid": episode_uids,
        "episode_position": positions.astype(np.int64),
        "steps_remaining": steps_remaining.astype(np.int64),
        "env_index": np.zeros(sample_count, dtype=np.int64),
        "policy_index": np.zeros(sample_count, dtype=np.int64),
        "episode_group_index": run_group_index,
        # Compatibility alias for precursor files created before OTTP-R26-G1-v1.
        # It denotes reconstructed episode groups, not logging collection runs.
        "run_group_index": run_group_index,
        "checkpoint_index": np.zeros(sample_count, dtype=np.int64),
        "critic_score": np.zeros(sample_count, dtype=np.float32),
        "rarecat_multiplier": np.ones(sample_count, dtype=np.float32),
        "catastrophic_flag": np.zeros(sample_count, dtype=np.int64),
        "metadata": {
            "selection_mode": "public_offline",
            "source_type": source_meta["source_type"],
            "source_name": source_meta["source_name"],
            "dataset_path": source_meta["dataset_path"],
            "task_name": task_label,
            "env_names": {0: task_label},
            "policy_names": {0: "public_offline"},
            "run_group_names": run_group_names,
            "episode_group_field": "episode_group_index",
            "run_group_index_semantics": "legacy_alias_for_reconstructed_episode_group_not_logging_run",
            "max_obs_dim": int(trimmed_payload["obs"].shape[1]),
            "max_act_dim": int(trimmed_payload["actions"].shape[1]),
            "skipped": [],
            "protocol": protocol.to_dict(),
            "raw_dataset_keys": sorted(str(key) for key in raw_dataset.keys()),
            "num_episodes": int(episode_uids.max()) + 1 if episode_uids.size else 0,
        },
    }


def summarize_public_precursor_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    episode_uids = np.asarray(dataset["episode_uid"], dtype=np.int64)
    run_group_indices = resolve_episode_group_index(dataset)
    metadata = dataset["metadata"]
    return {
        "selection_mode": str(metadata.get("selection_mode", "public_offline")),
        "source_type": str(metadata.get("source_type", "unknown")),
        "source_name": str(metadata.get("source_name", "unknown")),
        "dataset_path": metadata.get("dataset_path"),
        "task_name": str(metadata.get("task_name", "unknown")),
        "num_samples": int(np.asarray(dataset["obs"]).shape[0]),
        "num_episodes": int(episode_uids.max()) + 1 if episode_uids.size else 0,
        "num_run_groups": int(np.unique(run_group_indices).size),
        "num_episode_groups": int(np.unique(run_group_indices).size),
        "episode_group_field": "episode_group_index" if "episode_group_index" in dataset else "run_group_index",
        "run_group_index_semantics": "legacy_alias_for_reconstructed_episode_group_not_logging_run",
        "obs_dim": int(np.asarray(dataset["obs"]).shape[1]),
        "act_dim": int(np.asarray(dataset["actions"]).shape[1]),
        "protocol": metadata.get("protocol", {}),
        "env_names": metadata.get("env_names", {}),
        "run_group_names": metadata.get("run_group_names", {}),
    }


def _summarize_labels(labels: np.ndarray, min_class_ratio: float) -> dict[str, Any]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(np.int64)
    total = max(int(counts.sum()), 1)
    ratios = counts.astype(np.float64) / float(total)
    return {
        "counts": {
            "safe": int(counts[0]),
            "near_miss": int(counts[1]),
            "catastrophic": int(counts[2]),
        },
        "ratios": {
            "safe": float(ratios[0]),
            "near_miss": float(ratios[1]),
            "catastrophic": float(ratios[2]),
        },
        "all_classes_present": bool(np.all(counts > 0)),
        "extreme_class_imbalance": bool(np.any(counts == 0) or np.any(ratios < float(min_class_ratio))),
        "min_class_ratio": float(min_class_ratio),
    }


def _has_group_leakage(dataset: dict[str, Any], splits: dict[str, Any]) -> bool:
    run_group_index = resolve_episode_group_index(dataset)
    memberships = {
        split_name: set(np.unique(run_group_index[np.asarray(splits[split_name], dtype=np.int64)]).astype(np.int64).tolist())
        for split_name in ("train", "val", "test")
    }
    return bool(
        memberships["train"] & memberships["val"]
        or memberships["train"] & memberships["test"]
        or memberships["val"] & memberships["test"]
    )


def audit_public_task_protocol(
    dataset: dict[str, Any],
    *,
    split_seeds: list[int] | tuple[int, ...],
    protocol: PublicOfflineProtocolConfig | None = None,
) -> dict[str, Any]:
    'Audit whether a fixed public-offline protocol is stable for one task.'
    protocol = protocol or PublicOfflineProtocolConfig()
    summary = summarize_public_precursor_dataset(dataset)
    task_name = str(dataset["metadata"].get("task_name", summary["task_name"]))
    split_results: list[dict[str, Any]] = []
    stable_task = True
    reasons: list[str] = []

    for split_seed in split_seeds:
        try:
            splits = build_group_splits(dataset, seed=int(split_seed))
        except ValueError as exc:
            stable_task = False
            reasons.append(f"split_seed_{int(split_seed)}: {exc}")
            split_results.append({
                "split_seed": int(split_seed),
                "status": "error",
                "error": str(exc),
            })
            continue

        leakage = _has_group_leakage(dataset, splits)
        step_thresholds = compute_label_thresholds(
            dataset,
            splits["train"],
            value_key="cost_to_go_disc",
            lower_quantile=float(protocol.step_q_low),
            upper_quantile=float(protocol.step_q_high),
        )
        step_labels = assign_3way_labels(dataset["cost_to_go_disc"], dataset["env_index"], step_thresholds)
        outcome_thresholds = compute_label_thresholds(
            dataset,
            splits["train"],
            value_key="episode_cost_disc",
            lower_quantile=float(protocol.outcome_q_low),
            upper_quantile=float(protocol.outcome_q_high),
        )
        outcome_labels = assign_3way_labels(dataset["episode_cost_disc"], dataset["env_index"], outcome_thresholds)

        label_payload: dict[str, Any] = {}
        split_pass = not leakage
        for label_name, labels in (("step", step_labels), ("outcome", outcome_labels)):
            label_payload[label_name] = {}
            for split_name in ("train", "val", "test"):
                indices = np.asarray(splits[split_name], dtype=np.int64)
                stats = _summarize_labels(labels[indices], min_class_ratio=float(protocol.min_class_ratio))
                label_payload[label_name][split_name] = stats
                split_pass = split_pass and not bool(stats["extreme_class_imbalance"])

        if not split_pass:
            stable_task = False
            reasons.append(f"split_seed_{int(split_seed)} failed label stability or leakage checks")

        split_results.append({
            "split_seed": int(split_seed),
            "status": "ok",
            "group_leakage_detected": bool(leakage),
            "split_sizes": {name: int(len(splits[name])) for name in ("train", "val", "test")},
            "step_thresholds": {int(key): value for key, value in step_thresholds.items()},
            "outcome_thresholds": {int(key): value for key, value in outcome_thresholds.items()},
            "labels": label_payload,
            "gate_b_split_pass": bool(split_pass),
            "split_manifest": splits["manifest"],
        })

    return {
        "task_name": task_name,
        "dataset_summary": summary,
        "protocol": protocol.to_dict(),
        "task_specific_overrides_used": False,
        "protocol_mode": "global_fixed",
        "stable_task": bool(stable_task),
        "reasons": reasons,
        "split_results": split_results,
    }
