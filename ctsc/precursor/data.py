'ArchiveCurator and PrefixPrecursorLabeler utilities.'

from __future__ import annotations

import json
import math
import warnings
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # pragma: no cover - optional dependency for archive extraction
    import safety_gymnasium as safety_gym
except ImportError:  # pragma: no cover - optional dependency
    safety_gym = None


SelectionMode = Literal["latest", "latest+mid"]
SplitFamily = Literal["episode_group", "transition_random"]
_STEP_PATTERN = re.compile(r"step_(\d+)_periodic\.pt$")
SPLIT_FAMILY_REGISTRY: tuple[SplitFamily, ...] = ("episode_group", "transition_random")


def resolve_episode_group_index(dataset: dict[str, Any]) -> np.ndarray:
    """Return reconstructed episode groups.

    ``run_group_index`` is the historical compatibility alias. Its values
    identify episode groups; collection-run identity requires upstream metadata.
    """
    if "episode_group_index" in dataset:
        groups = np.asarray(dataset["episode_group_index"], dtype=np.int64)
        if "run_group_index" in dataset:
            legacy = np.asarray(dataset["run_group_index"], dtype=np.int64)
            if not np.array_equal(groups, legacy):
                raise ValueError("episode_group_index and legacy run_group_index disagree.")
        return groups
    if "run_group_index" not in dataset:
        raise KeyError("Dataset must provide episode_group_index or legacy run_group_index.")
    warnings.warn(
        "Using legacy run_group_index as an episode-group compatibility alias; "
        "it must not be interpreted as a true logging-run identifier.",
        FutureWarning,
        stacklevel=2,
    )
    return np.asarray(dataset["run_group_index"], dtype=np.int64)


@dataclass(frozen=True)
class SelectedCheckpoint:
    'One checkpoint selected for precursor extraction.'

    run_dir: Path
    run_name: str
    env_id: str
    agent_name: str
    checkpoint_path: Path
    checkpoint_step: int
    checkpoint_role: str
    latest_step: int


def _parse_step(path: Path) -> int:
    match = _STEP_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unsupported checkpoint filename: {path.name}")
    return int(match.group(1))


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def select_periodic_checkpoints(runs_root: str | Path, selection_mode: SelectionMode = "latest") -> list[SelectedCheckpoint]:
    'Select periodic checkpoints according to `latest` or `latest+mid` mode.'
    runs_root = Path(runs_root)
    selected: list[SelectedCheckpoint] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        checkpoints = sorted(run_dir.glob("checkpoints/*_periodic.pt"), key=_parse_step)
        if not checkpoints:
            continue
        config = _load_run_config(run_dir)
        env_id = str(config["env"]["id"])
        agent_name = str(config["agent"]["name"])
        latest = checkpoints[-1]
        latest_step = _parse_step(latest)
        picks: list[tuple[str, Path]] = [("latest", latest)]
        if selection_mode == "latest+mid" and len(checkpoints) > 1:
            target = latest_step / 2.0
            mid = min(checkpoints[:-1], key=lambda item: abs(_parse_step(item) - target))
            if mid != latest:
                picks.append(("mid", mid))
        seen: set[Path] = set()
        for role, checkpoint_path in sorted(picks, key=lambda item: _parse_step(item[1])):
            if checkpoint_path in seen:
                continue
            seen.add(checkpoint_path)
            selected.append(
                SelectedCheckpoint(
                    run_dir=run_dir,
                    run_name=run_dir.name,
                    env_id=env_id,
                    agent_name=agent_name,
                    checkpoint_path=checkpoint_path,
                    checkpoint_step=_parse_step(checkpoint_path),
                    checkpoint_role=role,
                    latest_step=latest_step,
                )
            )
    return selected


def _pad_width(array: np.ndarray, target_dim: int) -> np.ndarray:
    if array.shape[1] == target_dim:
        return array.astype(np.float32, copy=False)
    padded = np.zeros((array.shape[0], target_dim), dtype=np.float32)
    padded[:, : array.shape[1]] = array.astype(np.float32, copy=False)
    return padded


def _load_train_episode_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "diagnostics" / "train_episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Episode diagnostics not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _score_transitions(config: dict[str, Any], checkpoint: dict[str, Any], obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    if safety_gym is None:
        raise ImportError(
            "Private archive extraction requires `safety_gymnasium`, but it is not installed in this environment."
        )
    try:
        from ctsc.envs import register_rarecat_envs
        from ctsc.scripts.train import make_agent
    except ImportError as exc:
        raise ImportError(
            "Private archive extraction requires the full OTTP training stack "
            "(`ctsc.envs`, `ctsc.scripts.train`), but those modules are not present."
        ) from exc
    config = json.loads(json.dumps(config))
    config["device"] = "cpu"
    register_rarecat_envs()
    env = safety_gym.make(config["env"]["id"])
    try:
        agent = make_agent(
            config,
            obs_dim=int(env.observation_space.shape[0]),
            act_dim=int(env.action_space.shape[0]),
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        )
    finally:
        env.close()
    agent.load_state_dict(checkpoint["agent"])
    scores: list[np.ndarray] = []
    batch_size = 2048
    with torch.no_grad():
        for start in range(0, obs.shape[0], batch_size):
            end = min(start + batch_size, obs.shape[0])
            obs_t = torch.as_tensor(obs[start:end], dtype=torch.float32, device=agent.device)
            act_t = torch.as_tensor(actions[start:end], dtype=torch.float32, device=agent.device)
            tail = agent.cost_critic.tail_cvar_hat(obs_t, act_t, alpha=float(agent.risk_alpha))
            scores.append(tail.detach().cpu().numpy().reshape(-1))
    return np.concatenate(scores, axis=0).astype(np.float32, copy=False)


def _build_episode_ids(terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
    ends = np.logical_or(terminated.reshape(-1) > 0, truncated.reshape(-1) > 0)
    start_flags = np.concatenate(([True], ends[:-1]))
    return np.cumsum(start_flags.astype(np.int64)) - 1


def _derive_episode_scalars_from_replay(costs: np.ndarray, episode_ids: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
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
    calibration: dict[str, np.ndarray] | None,
    gamma: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    episode_ids = _build_episode_ids(terminated, truncated)
    keep_mask = np.ones(obs.shape[0], dtype=bool)
    if episode_ids.size > 0 and not bool(terminated[-1] > 0 or truncated[-1] > 0):
        last_episode = int(episode_ids[-1])
        keep_mask = episode_ids != last_episode

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

    if calibration is not None:
        payload["cost_to_go_disc"] = calibration["cost_to_go_disc"][: obs.shape[0]].reshape(-1)[keep_mask]
        payload["episode_cost_disc"] = calibration["episode_cost_disc"][: obs.shape[0]].reshape(-1)[keep_mask]
        payload["env_step"] = calibration["env_step"][: obs.shape[0]].reshape(-1)[keep_mask]
    else:
        derived = _derive_episode_scalars_from_replay(payload["costs"], trimmed_episode_ids, gamma=float(gamma))
        payload.update(derived)

    return payload, trimmed_episode_ids


def build_precursor_dataset(
    runs_root: str | Path,
    selection_mode: SelectionMode = "latest",
    include_critic_score: bool = True,
) -> dict[str, Any]:
    'Extract a precursor dataset from periodic checkpoints under `runs_root`.'
    try:
        from ctsc.utils.checkpoint import load_checkpoint
    except ImportError as exc:
        raise ImportError(
            "Private archive extraction requires `ctsc.utils.checkpoint`, but it is not present "
            "in this source-only workspace."
        ) from exc
    selections = select_periodic_checkpoints(runs_root, selection_mode=selection_mode)
    if not selections:
        raise ValueError("No periodic checkpoints found.")

    max_obs_dim = 0
    max_act_dim = 0
    for selection in selections:
        checkpoint = load_checkpoint(selection.checkpoint_path)
        replay = checkpoint.get("replay_buffer")
        if replay is None:
            continue
        max_obs_dim = max(max_obs_dim, int(replay["obs"].shape[1]))
        max_act_dim = max(max_act_dim, int(replay["actions"].shape[1]))
    if max_obs_dim <= 0 or max_act_dim <= 0:
        raise ValueError("Unable to infer observation/action dimensions from periodic checkpoints.")

    arrays: dict[str, list[np.ndarray]] = {key: [] for key in [
        "obs", "actions", "next_obs", "rewards", "costs", "terminated", "truncated",
        "cost_to_go_disc", "episode_cost_disc", "env_step", "episode_uid", "episode_position",
        "steps_remaining", "env_index", "policy_index", "run_group_index", "checkpoint_index",
        "critic_score", "rarecat_multiplier", "catastrophic_flag",
    ]}
    checkpoints_meta: list[dict[str, Any]] = []
    env_to_index: dict[str, int] = {}
    policy_to_index: dict[str, int] = {}
    run_group_to_index: dict[str, int] = {}
    skipped: list[dict[str, str]] = []
    episode_uid_offset = 0

    for checkpoint_index, selection in enumerate(selections):
        checkpoint = load_checkpoint(selection.checkpoint_path)
        replay = checkpoint.get("replay_buffer")
        calibration = checkpoint.get("calibration_buffer")
        if replay is None:
            skipped.append({"run_name": selection.run_name, "reason": "missing_replay_buffer"})
            continue
        size = int(replay["size"])
        if calibration is not None and int(calibration["size"]) != size:
            skipped.append({"run_name": selection.run_name, "reason": "size_mismatch"})
            continue

        obs = replay["obs"][:size]
        actions = replay["actions"][:size]
        next_obs = replay["next_obs"][:size]
        rewards = replay["rewards"][:size].reshape(-1)
        costs = replay["costs"][:size].reshape(-1)
        terminated = replay["terminated"][:size].reshape(-1)
        truncated = replay["truncated"][:size].reshape(-1)
        if calibration is not None and (
            not np.allclose(obs, calibration["obs"][:size], atol=1e-6)
            or not np.allclose(actions, calibration["actions"][:size], atol=1e-6)
        ):
            skipped.append({"run_name": selection.run_name, "reason": "alignment_mismatch"})
            continue

        gamma = float(checkpoint["config"].get("agent", {}).get("gamma", checkpoint["config"].get("gamma", 0.99)))
        trimmed_payload, episode_ids = _trim_incomplete_tail_episode(
            obs=obs,
            actions=actions,
            next_obs=next_obs,
            rewards=rewards,
            costs=costs,
            terminated=terminated,
            truncated=truncated,
            calibration=calibration,
            gamma=gamma,
        )
        obs = trimmed_payload["obs"]
        actions = trimmed_payload["actions"]
        next_obs = trimmed_payload["next_obs"]
        rewards = trimmed_payload["rewards"]
        costs = trimmed_payload["costs"]
        terminated = trimmed_payload["terminated"]
        truncated = trimmed_payload["truncated"]
        size = int(obs.shape[0])
        episode_count = int(episode_ids.max()) + 1 if episode_ids.size else 0
        episode_rows = _load_train_episode_rows(selection.run_dir)
        if episode_count > len(episode_rows):
            skipped.append({"run_name": selection.run_name, "reason": "episode_count_mismatch"})
            continue
        episode_rows = episode_rows[:episode_count]

        env_index = env_to_index.setdefault(selection.env_id, len(env_to_index))
        policy_index = policy_to_index.setdefault(selection.agent_name, len(policy_to_index))
        run_group_index = run_group_to_index.setdefault(selection.run_name, len(run_group_to_index))

        episode_lengths = np.bincount(episode_ids, minlength=episode_count).astype(np.int64)
        episode_positions = np.zeros_like(episode_ids)
        for episode_id in range(episode_count):
            mask = episode_ids == episode_id
            episode_positions[mask] = np.arange(int(mask.sum()), dtype=np.int64)
        steps_remaining = episode_lengths[episode_ids] - episode_positions - 1

        rarecat_multiplier = np.zeros(size, dtype=np.float32)
        catastrophic_flag = np.zeros(size, dtype=np.int64)
        for episode_id, row in enumerate(episode_rows):
            mask = episode_ids == episode_id
            rarecat_multiplier[mask] = float(row.get("rarecat_multiplier", 1.0))
            catastrophic_flag[mask] = int(row.get("catastrophic_flag", 0))

        critic_score = _score_transitions(checkpoint["config"], checkpoint, obs.astype(np.float32), actions.astype(np.float32)) if include_critic_score else np.zeros(size, dtype=np.float32)

        arrays["obs"].append(_pad_width(obs, max_obs_dim))
        arrays["actions"].append(_pad_width(actions, max_act_dim))
        arrays["next_obs"].append(_pad_width(next_obs, max_obs_dim))
        arrays["rewards"].append(rewards.astype(np.float32))
        arrays["costs"].append(costs.astype(np.float32))
        arrays["terminated"].append(terminated.astype(np.float32))
        arrays["truncated"].append(truncated.astype(np.float32))
        arrays["cost_to_go_disc"].append(trimmed_payload["cost_to_go_disc"].astype(np.float32))
        arrays["episode_cost_disc"].append(trimmed_payload["episode_cost_disc"].astype(np.float32))
        arrays["env_step"].append(trimmed_payload["env_step"].astype(np.float32))
        arrays["episode_uid"].append((episode_ids + episode_uid_offset).astype(np.int64))
        arrays["episode_position"].append(episode_positions.astype(np.int64))
        arrays["steps_remaining"].append(steps_remaining.astype(np.int64))
        arrays["env_index"].append(np.full(size, env_index, dtype=np.int64))
        arrays["policy_index"].append(np.full(size, policy_index, dtype=np.int64))
        arrays["run_group_index"].append(np.full(size, run_group_index, dtype=np.int64))
        arrays["checkpoint_index"].append(np.full(size, checkpoint_index, dtype=np.int64))
        arrays["critic_score"].append(critic_score.astype(np.float32))
        arrays["rarecat_multiplier"].append(rarecat_multiplier)
        arrays["catastrophic_flag"].append(catastrophic_flag.astype(np.int64))

        checkpoints_meta.append({
            "checkpoint_index": checkpoint_index,
            "run_name": selection.run_name,
            "env_id": selection.env_id,
            "agent_name": selection.agent_name,
            "checkpoint_path": str(selection.checkpoint_path),
            "checkpoint_step": selection.checkpoint_step,
            "checkpoint_role": selection.checkpoint_role,
            "latest_step": selection.latest_step,
            "obs_dim": int(obs.shape[1]),
            "act_dim": int(actions.shape[1]),
            "episode_count": episode_count,
        })
        episode_uid_offset += episode_count

    if not arrays["obs"]:
        raise ValueError("No checkpoints could be extracted into a precursor dataset.")

    return {
        "obs": np.concatenate(arrays["obs"], axis=0),
        "actions": np.concatenate(arrays["actions"], axis=0),
        "next_obs": np.concatenate(arrays["next_obs"], axis=0),
        "rewards": np.concatenate(arrays["rewards"], axis=0),
        "costs": np.concatenate(arrays["costs"], axis=0),
        "terminated": np.concatenate(arrays["terminated"], axis=0),
        "truncated": np.concatenate(arrays["truncated"], axis=0),
        "cost_to_go_disc": np.concatenate(arrays["cost_to_go_disc"], axis=0),
        "episode_cost_disc": np.concatenate(arrays["episode_cost_disc"], axis=0),
        "env_step": np.concatenate(arrays["env_step"], axis=0),
        "episode_uid": np.concatenate(arrays["episode_uid"], axis=0),
        "episode_position": np.concatenate(arrays["episode_position"], axis=0),
        "steps_remaining": np.concatenate(arrays["steps_remaining"], axis=0),
        "env_index": np.concatenate(arrays["env_index"], axis=0),
        "policy_index": np.concatenate(arrays["policy_index"], axis=0),
        "run_group_index": np.concatenate(arrays["run_group_index"], axis=0),
        "checkpoint_index": np.concatenate(arrays["checkpoint_index"], axis=0),
        "critic_score": np.concatenate(arrays["critic_score"], axis=0),
        "rarecat_multiplier": np.concatenate(arrays["rarecat_multiplier"], axis=0),
        "catastrophic_flag": np.concatenate(arrays["catastrophic_flag"], axis=0),
        "metadata": {
            "selection_mode": selection_mode,
            "checkpoints": checkpoints_meta,
            "env_names": {index: env_id for env_id, index in env_to_index.items()},
            "policy_names": {index: policy for policy, index in policy_to_index.items()},
            "run_group_names": {index: run_name for run_name, index in run_group_to_index.items()},
            "max_obs_dim": max_obs_dim,
            "max_act_dim": max_act_dim,
            "skipped": skipped,
        },
    }


def save_precursor_dataset(dataset: dict[str, Any], path: str | Path) -> Path:
    'Save extracted dataset with torch serialization.'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    # Some Unified38 precursor datasets exceed 2GB on Windows. The legacy
    # serializer plus atomic replace avoids leaving truncated zip checkpoints.
    torch.save(dataset, tmp_path, _use_new_zipfile_serialization=False)
    tmp_path.replace(path)
    return path


def load_precursor_dataset(path: str | Path) -> dict[str, Any]:
    'Load a previously extracted precursor dataset.'
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _split_group_names(group_names: list[int], seed: int, ratios: tuple[float, float, float]) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(group_names, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    total = shuffled.size
    train_count = max(1, int(math.floor(total * ratios[0]))) if total >= 3 else max(1, total - 2)
    val_count = max(1, int(math.floor(total * ratios[1]))) if total >= 3 else 1
    if train_count + val_count >= total:
        train_count = max(1, total - 2)
        val_count = 1
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
    return (
        shuffled[:train_count].tolist(),
        shuffled[train_count : train_count + val_count].tolist(),
        shuffled[train_count + val_count :].tolist(),
    )


def _split_sample_indices(sample_indices: np.ndarray, seed: int, ratios: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(sample_indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    total = int(shuffled.size)
    if total < 3:
        raise ValueError(f"Need at least 3 samples for a train/val/test split, got total={total}.")
    train_count = max(1, int(math.floor(total * ratios[0])))
    val_count = max(1, int(math.floor(total * ratios[1])))
    if train_count + val_count >= total:
        train_count = max(1, total - 2)
        val_count = 1
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
    return (
        np.sort(shuffled[:train_count]).astype(np.int64),
        np.sort(shuffled[train_count : train_count + val_count]).astype(np.int64),
        np.sort(shuffled[train_count + val_count :]).astype(np.int64),
    )


def _validate_split_family(split_family: str) -> SplitFamily:
    family = str(split_family)
    if family not in SPLIT_FAMILY_REGISTRY:
        raise ValueError(f"Unsupported split_family `{split_family}`. Supported={SPLIT_FAMILY_REGISTRY}.")
    return family  # type: ignore[return-value]


def build_group_splits(
    dataset: dict[str, Any],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, Any]:
    'Split samples by run-group within each environment to avoid leakage.'
    env_index = np.asarray(dataset["env_index"], dtype=np.int64)
    run_group_index = resolve_episode_group_index(dataset)
    train_mask = np.zeros(env_index.shape[0], dtype=bool)
    val_mask = np.zeros(env_index.shape[0], dtype=bool)
    test_mask = np.zeros(env_index.shape[0], dtype=bool)
    manifest: dict[str, Any] = {
        "split_family": "episode_group",
        "group_field": "episode_group_index" if "episode_group_index" in dataset else "run_group_index",
        "group_semantics": "reconstructed_episode_group_not_logging_run",
        "ratios": [float(ratio) for ratio in ratios],
        "train_groups": {},
        "val_groups": {},
        "test_groups": {},
        "train_counts": {},
        "val_counts": {},
        "test_counts": {},
    }

    for env_id in np.unique(env_index):
        mask = env_index == env_id
        groups = sorted(np.unique(run_group_index[mask]).tolist())
        if len(groups) < 3:
            raise ValueError(f"Environment index {env_id} needs at least 3 run groups for a train/val/test split.")
        train_groups, val_groups, test_groups = _split_group_names(groups, seed=int(seed) + int(env_id), ratios=ratios)
        manifest["train_groups"][int(env_id)] = train_groups
        manifest["val_groups"][int(env_id)] = val_groups
        manifest["test_groups"][int(env_id)] = test_groups
        train_local = mask & np.isin(run_group_index, train_groups)
        val_local = mask & np.isin(run_group_index, val_groups)
        test_local = mask & np.isin(run_group_index, test_groups)
        train_mask |= train_local
        val_mask |= val_local
        test_mask |= test_local
        manifest["train_counts"][int(env_id)] = int(np.count_nonzero(train_local))
        manifest["val_counts"][int(env_id)] = int(np.count_nonzero(val_local))
        manifest["test_counts"][int(env_id)] = int(np.count_nonzero(test_local))

    return {"train": np.flatnonzero(train_mask), "val": np.flatnonzero(val_mask), "test": np.flatnonzero(test_mask), "manifest": manifest}


def build_transition_random_splits(
    dataset: dict[str, Any],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, Any]:
    'Split samples by transition within each environment.'
    env_index = np.asarray(dataset["env_index"], dtype=np.int64)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    manifest: dict[str, Any] = {
        "split_family": "transition_random",
        "ratios": [float(ratio) for ratio in ratios],
        "train_counts": {},
        "val_counts": {},
        "test_counts": {},
    }

    for env_id in np.unique(env_index):
        env_indices = np.flatnonzero(env_index == env_id)
        train_env, val_env, test_env = _split_sample_indices(env_indices, seed=int(seed) + int(env_id), ratios=ratios)
        train_indices.append(train_env)
        val_indices.append(val_env)
        test_indices.append(test_env)
        manifest["train_counts"][int(env_id)] = int(train_env.shape[0])
        manifest["val_counts"][int(env_id)] = int(val_env.shape[0])
        manifest["test_counts"][int(env_id)] = int(test_env.shape[0])

    return {
        "train": np.concatenate(train_indices, axis=0) if train_indices else np.zeros((0,), dtype=np.int64),
        "val": np.concatenate(val_indices, axis=0) if val_indices else np.zeros((0,), dtype=np.int64),
        "test": np.concatenate(test_indices, axis=0) if test_indices else np.zeros((0,), dtype=np.int64),
        "manifest": manifest,
    }


def build_splits(
    dataset: dict[str, Any],
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    split_family: str = "episode_group",
) -> dict[str, Any]:
    family = _validate_split_family(split_family)
    if family == "episode_group":
        return build_group_splits(dataset, seed=int(seed), ratios=ratios)
    return build_transition_random_splits(dataset, seed=int(seed), ratios=ratios)


def summarize_split_diagnostics(dataset: dict[str, Any], splits: dict[str, Any]) -> dict[str, Any]:
    'Summarize overlap and fragmentation diagnostics for one split assignment.'
    run_group_index = resolve_episode_group_index(dataset)
    episode_uid = np.asarray(dataset["episode_uid"], dtype=np.int64)
    memberships = {
        split_name: np.asarray(splits[split_name], dtype=np.int64)
        for split_name in ("train", "val", "test")
    }
    group_sets = {
        split_name: set(np.unique(run_group_index[indices]).astype(np.int64).tolist())
        for split_name, indices in memberships.items()
    }
    episode_sets = {
        split_name: set(np.unique(episode_uid[indices]).astype(np.int64).tolist())
        for split_name, indices in memberships.items()
    }
    pairwise_group_overlap: dict[str, dict[str, float | int]] = {}
    pairwise_episode_overlap: dict[str, dict[str, float | int]] = {}
    pairings = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairings:
        group_overlap = group_sets[left] & group_sets[right]
        episode_overlap = episode_sets[left] & episode_sets[right]
        group_union = group_sets[left] | group_sets[right]
        episode_union = episode_sets[left] | episode_sets[right]
        pairwise_group_overlap[f"{left}_{right}"] = {
            "count": int(len(group_overlap)),
            "ratio": float(len(group_overlap) / max(len(group_union), 1)),
        }
        pairwise_episode_overlap[f"{left}_{right}"] = {
            "count": int(len(episode_overlap)),
            "ratio": float(len(episode_overlap) / max(len(episode_union), 1)),
        }

    episode_membership_counter: dict[int, set[str]] = {}
    for split_name, episode_ids in episode_sets.items():
        for episode_id in episode_ids:
            episode_membership_counter.setdefault(int(episode_id), set()).add(split_name)
    fragmented_episodes = sorted(int(episode_id) for episode_id, owners in episode_membership_counter.items() if len(owners) > 1)

    group_membership_counter: dict[int, set[str]] = {}
    for split_name, group_ids in group_sets.items():
        for group_id in group_ids:
            group_membership_counter.setdefault(int(group_id), set()).add(split_name)
    overlapping_groups = sorted(int(group_id) for group_id, owners in group_membership_counter.items() if len(owners) > 1)

    return {
        "split_family": str(splits.get("manifest", {}).get("split_family", "unknown")),
        "split_sizes": {split_name: int(indices.shape[0]) for split_name, indices in memberships.items()},
        "group_overlap": {
            "overlapping_group_count": int(len(overlapping_groups)),
            "overlapping_group_ratio": float(len(overlapping_groups) / max(len(group_membership_counter), 1)),
            "pairwise": pairwise_group_overlap,
            "group_leakage_detected": bool(overlapping_groups),
        },
        "episode_fragmentation": {
            "fragmented_episode_count": int(len(fragmented_episodes)),
            "fragmented_episode_ratio": float(len(fragmented_episodes) / max(len(episode_membership_counter), 1)),
            "pairwise": pairwise_episode_overlap,
            "fragmentation_detected": bool(fragmented_episodes),
        },
    }


def compute_label_thresholds(
    dataset: dict[str, Any],
    indices: np.ndarray,
    value_key: str,
    lower_quantile: float = 0.75,
    upper_quantile: float = 0.90,
) -> dict[int, dict[str, float]]:
    'Compute per-environment quantile thresholds using train-only samples.'
    if not 0.0 < float(lower_quantile) < float(upper_quantile) < 1.0:
        raise ValueError(
            "Quantiles must satisfy 0 < lower_quantile < upper_quantile < 1, "
            f"got lower_quantile={lower_quantile}, upper_quantile={upper_quantile}."
        )
    values = np.asarray(dataset[value_key], dtype=np.float64).reshape(-1)
    env_indices = np.asarray(dataset["env_index"], dtype=np.int64).reshape(-1)
    thresholds: dict[int, dict[str, float]] = {}
    for env_id in np.unique(env_indices[indices]):
        mask = indices[np.asarray(env_indices[indices] == env_id)]
        env_values = values[mask]
        thresholds[int(env_id)] = {
            "q75": float(np.quantile(env_values, float(lower_quantile))),
            "q90": float(np.quantile(env_values, float(upper_quantile))),
        }
    return thresholds


def assign_3way_labels(values: np.ndarray, env_indices: np.ndarray, thresholds: dict[int, dict[str, float]]) -> np.ndarray:
    'Assign safe/near-miss/catastrophic labels using per-env thresholds.'
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    env_indices = np.asarray(env_indices, dtype=np.int64).reshape(-1)
    labels = np.zeros(values.shape[0], dtype=np.int64)
    for env_id, limit in thresholds.items():
        mask = env_indices == int(env_id)
        labels[mask] = 0
        labels[mask & (values >= float(limit["q75"]))] = 1
        labels[mask & (values >= float(limit["q90"]))] = 2
    return labels


def compute_normalization_stats(dataset: dict[str, Any], indices: np.ndarray) -> dict[str, np.ndarray]:
    'Fit mean/std normalization on train-only flat features.'
    feature_matrix = build_feature_profile_matrix(dataset, profile_name="full")[np.asarray(indices, dtype=np.int64)]
    mean = feature_matrix.mean(axis=0).astype(np.float32)
    std = feature_matrix.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}

FEATURE_PROFILE_REGISTRY = (
    "full",
    "obs_action",
    "obs_action_cost",
    "obs_action_step",
    "cost_only",
    "step_only",
    "cost_step_only",
    "current_only",
    "full_shuffled",
    "history_without_current_cost",
    "lagged_cost_step",
    "current_without_cost",
    "history_obs_action_step_only",
)


def _validate_feature_profile(profile_name: str) -> str:
    profile = str(profile_name)
    if profile not in FEATURE_PROFILE_REGISTRY:
        raise ValueError(
            f"Unsupported feature profile `{profile_name}`. "
            f"Supported profiles={FEATURE_PROFILE_REGISTRY}."
        )
    return profile


def _profile_feature_blocks(profile_name: str) -> tuple[str, ...]:
    profile = _validate_feature_profile(profile_name)
    if profile in {"full", "current_only", "full_shuffled", "history_without_current_cost"}:
        return ("obs", "actions", "costs", "env_step")
    if profile == "obs_action":
        return ("obs", "actions")
    if profile == "obs_action_cost":
        return ("obs", "actions", "costs")
    if profile == "obs_action_step":
        return ("obs", "actions", "env_step")
    if profile in {"current_without_cost", "history_obs_action_step_only"}:
        return ("obs", "actions", "env_step")
    if profile == "cost_only":
        return ("costs",)
    if profile == "step_only":
        return ("env_step",)
    if profile in {"cost_step_only", "lagged_cost_step"}:
        return ("costs", "env_step")
    raise ValueError(f"Unhandled profile_name={profile_name}")


def feature_profile_component_names(dataset: dict[str, Any], profile_name: str) -> list[str]:
    'Return ordered component names used by a feature profile.'
    component_names: list[str] = []
    for block in _profile_feature_blocks(profile_name):
        if block == "obs":
            obs_dim = int(np.asarray(dataset["obs"]).shape[1])
            component_names.extend([f"obs[{idx}]" for idx in range(obs_dim)])
        elif block == "actions":
            act_dim = int(np.asarray(dataset["actions"]).shape[1])
            component_names.extend([f"action[{idx}]" for idx in range(act_dim)])
        elif block == "costs":
            component_names.append("cost")
        elif block == "env_step":
            component_names.append("env_step")
        else:
            raise ValueError(f"Unsupported block={block}")
    return component_names


def build_feature_profile_matrix(dataset: dict[str, Any], profile_name: str = "full") -> np.ndarray:
    'Build a flat per-step feature matrix for a named feature profile.'
    pieces: list[np.ndarray] = []
    for block in _profile_feature_blocks(profile_name):
        if block == "obs":
            pieces.append(np.asarray(dataset["obs"], dtype=np.float32))
        elif block == "actions":
            pieces.append(np.asarray(dataset["actions"], dtype=np.float32))
        elif block == "costs":
            pieces.append(np.asarray(dataset["costs"], dtype=np.float32).reshape(-1, 1))
        elif block == "env_step":
            pieces.append(np.asarray(dataset["env_step"], dtype=np.float32).reshape(-1, 1))
        else:
            raise ValueError(f"Unsupported block={block}")
    return np.concatenate(pieces, axis=1)


def feature_profile_current_cost_index(dataset: dict[str, Any], profile_name: str) -> int | None:
    'Return the flat-feature column index of current-step cost for a profile.'
    component_names = feature_profile_component_names(dataset, profile_name)
    for index, name in enumerate(component_names):
        if name == "cost":
            return int(index)
    return None


def compute_profile_normalization_stats(
    dataset: dict[str, Any],
    indices: np.ndarray,
    profile_name: str,
    feature_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray | str]:
    'Fit train-only normalization stats for a named feature profile.'
    matrix = (
        np.asarray(feature_matrix, dtype=np.float32)
        if feature_matrix is not None
        else build_feature_profile_matrix(dataset, profile_name=profile_name)
    )
    idx = np.asarray(indices, dtype=np.int64)
    mean = matrix[idx].mean(axis=0).astype(np.float32)
    std = matrix[idx].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std, "profile_name": _validate_feature_profile(profile_name)}


def sequence_view_for_profile(profile_name: str) -> str:
    'Map a feature profile to the required sequence-window view.'
    profile = _validate_feature_profile(profile_name)
    if profile in {"current_only", "current_without_cost"}:
        return "current_only"
    if profile == "full_shuffled":
        return "full_shuffled"
    if profile in {"history_without_current_cost", "lagged_cost_step"}:
        return "history_without_current_cost"
    return "full"


def _episode_first_index(episode_uid: np.ndarray) -> np.ndarray:
    episode_uid = np.asarray(episode_uid, dtype=np.int64)
    first_index = np.zeros_like(episode_uid)
    unique_episode_ids, starts = np.unique(episode_uid, return_index=True)
    for episode_id, start in zip(unique_episode_ids, starts):
        first_index[episode_uid == episode_id] = int(start)
    return first_index


class FeatureMatrixSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    'Fixed-window prefix dataset backed by an externally prepared feature matrix.'

    def __init__(
        self,
        features: np.ndarray,
        dataset: dict[str, Any],
        indices: np.ndarray,
        labels: np.ndarray,
        window_size: int = 32,
        sequence_view: str = "full",
        shuffle_seed: int = 0,
        current_cost_index: int | None = None,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.window_size = int(window_size)
        self.sequence_view = str(sequence_view)
        self.shuffle_seed = int(shuffle_seed)
        self.current_cost_index = current_cost_index
        self.episode_uid = np.asarray(dataset["episode_uid"], dtype=np.int64)
        self.episode_first_index = _episode_first_index(self.episode_uid)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _transform_window(self, window: np.ndarray, *, index: int) -> np.ndarray:
        transformed = np.asarray(window, dtype=np.float32).copy()
        if self.sequence_view == "full":
            return transformed
        if self.sequence_view == "current_only":
            current = transformed[-1].copy()
            transformed.fill(0.0)
            transformed[-1] = current
            return transformed
        if self.sequence_view == "full_shuffled":
            if transformed.shape[0] > 1:
                rng = np.random.default_rng(self.shuffle_seed + int(index) * 104_729)
                transformed = transformed[rng.permutation(transformed.shape[0])]
            return transformed
        if self.sequence_view == "history_without_current_cost":
            if self.current_cost_index is None:
                raise ValueError("current_cost_index is required for history_without_current_cost.")
            if 0 <= int(self.current_cost_index) < transformed.shape[1]:
                transformed[-1, int(self.current_cost_index)] = 0.0
            return transformed
        raise ValueError(f"Unsupported sequence_view={self.sequence_view}")

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        episode_start = int(self.episode_first_index[index])
        start = max(episode_start, index - self.window_size + 1)
        window = self._transform_window(self.features[start : index + 1], index=index)
        padded = np.zeros((self.window_size, self.features.shape[1]), dtype=np.float32)
        padded[-window.shape[0] :] = window
        return {
            "sequence": torch.as_tensor(padded, dtype=torch.float32),
            "label": torch.as_tensor(int(self.labels[index]), dtype=torch.long),
            "critic_score": torch.as_tensor(float(self.dataset["critic_score"][index]), dtype=torch.float32),
            "env_index": torch.as_tensor(int(self.dataset["env_index"][index]), dtype=torch.long),
            "episode_uid": torch.as_tensor(int(self.dataset["episode_uid"][index]), dtype=torch.long),
            "steps_remaining": torch.as_tensor(float(self.dataset["steps_remaining"][index]), dtype=torch.float32),
            "sample_index": torch.as_tensor(index, dtype=torch.long),
        }


class FeatureMatrixWindowFlatDataset(FeatureMatrixSequenceDataset):
    'Flattened fixed-window view for masked-history flat baselines.'

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(item)
        sequence = sample.pop("sequence")
        sample["flat_features"] = sequence.reshape(-1)
        return sample


class PrecursorSequenceDataset(FeatureMatrixSequenceDataset):
    'Fixed-window prefix dataset over extracted transitions.'

    def __init__(
        self,
        dataset: dict[str, Any],
        indices: np.ndarray,
        labels: np.ndarray,
        normalization_stats: dict[str, np.ndarray],
        window_size: int = 32,
    ) -> None:
        feature_matrix = build_feature_profile_matrix(dataset, profile_name="full")
        mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
        std = np.asarray(normalization_stats["std"], dtype=np.float32)
        normalized = (feature_matrix - mean) / std
        super().__init__(
            features=normalized,
            dataset=dataset,
            indices=indices,
            labels=labels,
            window_size=window_size,
            sequence_view="full",
            shuffle_seed=0,
            current_cost_index=feature_profile_current_cost_index(dataset, "full"),
        )
