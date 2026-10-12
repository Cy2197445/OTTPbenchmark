'Frozen, episode-level offline alarm utility for revision stage 9.'

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


STAGE9_BUDGETS = (0.01, 0.05, 0.10)
STAGE9_MODELS = ("gru", "flat_mlp", "transformer")


def full_prefix_windows(
    features: np.ndarray,
    indices: np.ndarray,
    episode_positions: np.ndarray,
    episode_uids: np.ndarray,
    *,
    window_size: int,
) -> np.ndarray:
    'Vectorize the frozen full-prefix zero-padded dataset view for one batch.'
    matrix = np.asarray(features, dtype=np.float32)
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    positions = np.asarray(episode_positions, dtype=np.int64).reshape(-1)
    uids = np.asarray(episode_uids, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or positions.size != matrix.shape[0] or uids.size != matrix.shape[0]:
        raise ValueError("Feature and episode arrays have incompatible shapes")
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= matrix.shape[0]) or window_size <= 0:
        raise ValueError("Nonempty valid indices and a positive window size are required")
    first = selected - positions[selected]
    if np.any(first < 0) or np.any(uids[first] != uids[selected]):
        raise ValueError("Episode positions do not identify valid episode starts")
    offsets = np.arange(window_size - 1, -1, -1, dtype=np.int64)
    window_indices = selected[:, None] - offsets[None, :]
    present = window_indices >= first[:, None]
    windows = matrix[np.maximum(window_indices, 0)].copy()
    windows[~present] = 0.0
    return windows


@dataclass(frozen=True)
class BudgetThreshold:
    threshold: float
    validation_step_alarm_rate: float
    budget: float
    candidate_count: int


def select_validation_threshold(scores: np.ndarray, budget: float) -> BudgetThreshold:
    'Match a validation step-alarm budget in O(n log n), including score ties.'
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Validation scores must be nonempty and finite")
    target = float(budget)
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("Alarm budget must be finite and strictly between 0 and 1")
    unique, counts = np.unique(values, return_counts=True)
    rates = np.cumsum(counts[::-1], dtype=np.int64)[::-1] / values.size
    sentinel = float(np.nextafter(float(unique[-1]), math.inf))
    thresholds = np.concatenate(([sentinel], unique[::-1]))
    candidate_rates = np.concatenate(([0.0], rates[::-1]))
    selected = min(
        range(thresholds.size),
        key=lambda i: (abs(float(candidate_rates[i]) - target), float(candidate_rates[i]), -float(thresholds[i])),
    )
    return BudgetThreshold(
        threshold=float(thresholds[selected]),
        validation_step_alarm_rate=float(candidate_rates[selected]),
        budget=target,
        candidate_count=int(thresholds.size),
    )


def episode_alarm_utility(
    *,
    scores: np.ndarray,
    threshold: float,
    episode_ids: np.ndarray,
    outcome_labels: np.ndarray,
    steps_remaining: np.ndarray,
) -> dict[str, int | float | None]:
    'Evaluate an already locked threshold on complete held-out episodes.'
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(outcome_labels, dtype=np.int64).reshape(-1)
    remaining = np.asarray(steps_remaining, dtype=np.float64).reshape(-1)
    if values.size == 0 or not (values.size == ids.size == labels.size == remaining.size):
        raise ValueError("Nonempty equal-length scores, episode IDs, labels and remaining steps are required")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(remaining)) or not np.isfinite(threshold):
        raise ValueError("Scores, threshold and remaining steps must be finite")
    if np.any(remaining < 0) or np.any(~np.isin(labels, (0, 1, 2))):
        raise ValueError("Invalid remaining steps or outcome labels")

    _, inverse = np.unique(ids, return_inverse=True)
    episode_count = int(inverse.max()) + 1
    label_min = np.full(episode_count, 3, dtype=np.int64)
    label_max = np.full(episode_count, -1, dtype=np.int64)
    np.minimum.at(label_min, inverse, labels)
    np.maximum.at(label_max, inverse, labels)
    if np.any(label_min != label_max):
        raise ValueError("Outcome label is not constant within an episode")

    alarm = values >= float(threshold)
    alarm_count_by_episode = np.bincount(inverse[alarm], minlength=episode_count)
    alarmed = alarm_count_by_episode > 0
    catastrophic = label_max == 2
    recalled = alarmed & catastrophic
    false_alarm = alarmed & ~catastrophic

    first_alarm_remaining = np.full(episode_count, -1.0, dtype=np.float64)
    np.maximum.at(first_alarm_remaining, inverse[alarm], remaining[alarm])
    recalled_lead = first_alarm_remaining[recalled]
    cat_count = int(catastrophic.sum())
    noncat_count = episode_count - cat_count
    alarmed_count = int(alarmed.sum())
    recalled_count = int(recalled.sum())
    false_count = int(false_alarm.sum())
    return {
        "test_step_count": int(values.size),
        "test_step_alarm_count": int(alarm.sum()),
        "test_step_alarm_rate": float(alarm.mean()),
        "test_episode_count": episode_count,
        "test_alarmed_episode_count": alarmed_count,
        "test_catastrophic_episode_count": cat_count,
        "test_noncatastrophic_episode_count": noncat_count,
        "test_recalled_catastrophic_episode_count": recalled_count,
        "test_false_alarm_episode_count": false_count,
        "test_episode_alarm_rate": alarmed_count / episode_count,
        "catastrophic_episode_recall": recalled_count / cat_count if cat_count else None,
        "episode_alarm_precision": recalled_count / alarmed_count if alarmed_count else None,
        "noncat_episode_false_alarm_rate": false_count / noncat_count if noncat_count else None,
        "recalled_first_alarm_steps_to_episode_end_median": float(np.median(recalled_lead)) if recalled_count else None,
        "recalled_first_alarm_steps_to_episode_end_mean": float(np.mean(recalled_lead)) if recalled_count else None,
    }


def evaluate_budget(
    validation_scores: np.ndarray,
    *,
    budget: float,
    test_scores: np.ndarray,
    test_episode_ids: np.ndarray,
    test_outcome_labels: np.ndarray,
    test_steps_remaining: np.ndarray,
) -> dict[str, int | float | None]:
    'Validation-only threshold selection followed by test-only evaluation.'
    locked = select_validation_threshold(validation_scores, budget)
    return {
        **asdict(locked),
        **episode_alarm_utility(
            scores=test_scores,
            threshold=locked.threshold,
            episode_ids=test_episode_ids,
            outcome_labels=test_outcome_labels,
            steps_remaining=test_steps_remaining,
        ),
    }
