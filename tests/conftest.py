from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pytest
import torch

from ctsc.precursor.public_offline import PublicOfflineProtocolConfig, build_public_precursor_dataset


def _build_mock_public_raw_dataset() -> dict[str, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    next_obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    rewards: list[float] = []
    costs: list[float] = []
    terminals: list[float] = []
    timeouts: list[float] = []

    episode_count = 6
    steps_per_episode = 6
    gamma = 0.99
    for episode_id in range(episode_count):
        episode_costs = np.asarray(
            [0.05 * (step + 1) + 0.08 * episode_id for step in range(steps_per_episode)],
            dtype=np.float32,
        )
        for step in range(steps_per_episode):
            obs = np.asarray(
                [
                    float(episode_id),
                    float(step),
                    float(episode_id + step) / 10.0,
                    float(episode_id - step) / 10.0,
                ],
                dtype=np.float32,
            )
            next_obs = obs + np.asarray([0.1, 0.1, 0.05, -0.05], dtype=np.float32)
            action = np.asarray(
                [
                    float((episode_id + 1) * 0.1),
                    float((step + 1) * 0.05),
                ],
                dtype=np.float32,
            )
            obs_rows.append(obs)
            next_obs_rows.append(next_obs)
            action_rows.append(action)
            rewards.append(float(1.0 - episode_costs[step]))
            costs.append(float(episode_costs[step]))
            terminals.append(1.0 if step == steps_per_episode - 1 else 0.0)
            timeouts.append(0.0)

    return {
        "observations": np.asarray(obs_rows, dtype=np.float32),
        "next_observations": np.asarray(next_obs_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32),
        "timeouts": np.asarray(timeouts, dtype=np.float32),
    }


@pytest.fixture()
def mock_public_dataset_files(tmp_path: Path) -> dict[str, Path]:
    raw_dataset = _build_mock_public_raw_dataset()
    raw_path = tmp_path / "mock_ant_run_raw.pt"
    torch.save(raw_dataset, raw_path)

    precursor_dataset = build_public_precursor_dataset(
        raw_dataset=raw_dataset,
        task_name="AntRun",
        protocol=PublicOfflineProtocolConfig(gamma=0.99, group_size_episodes=1),
    )
    precursor_path = tmp_path / "mock_ant_run_precursor.pt"
    torch.save(precursor_dataset, precursor_path)

    return {
        "raw_path": raw_path,
        "precursor_path": precursor_path,
    }
