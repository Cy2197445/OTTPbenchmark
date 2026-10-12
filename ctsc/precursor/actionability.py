'Offline actionability helpers for Unified38 current-facing studies.'

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


DEFAULT_ACTIONABILITY_BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
DEFAULT_ACTIONABILITY_MODELS = [
    "gru",
    "temporal-cnn",
    "flat-mlp",
    "linear",
    "prevalence",
    "critic_score",
]
ACTIONABILITY_MODEL_ORDER = [
    "gru",
    "flat_mlp",
    "prevalence",
    "critic_score",
    "temporal_cnn",
    "linear",
]
ACTIONABILITY_HEADLINE_MODELS = ["gru", "flat_mlp", "prevalence", "critic_score"]
ACTIONABILITY_HEADLINE_BUDGETS = [0.01, 0.05, 0.10]
ACTIONABILITY_ANCHOR_BUDGET = 0.05


def normalize_actionability_model_name(model_name: str) -> str:
    return str(model_name).replace("-", "_")


@dataclass(frozen=True)
class AlarmThresholdSelection:
    threshold: float
    alarm_rate: float
    abs_diff: float
    candidate_count: int


@dataclass(frozen=True)
class EpisodeAlarmSummary:
    step_alarm_rate: float
    episode_alarm_rate: float
    catastrophic_episode_recall: float
    noncat_episode_false_alarm_rate: float
    episode_alarm_precision: float
    recalled_catastrophic_lead_time_median: float
    recalled_catastrophic_lead_time_mean: float
    catastrophic_episode_count: int
    noncat_episode_count: int
    alarmed_episode_count: int
    recalled_catastrophic_episode_count: int


def choose_threshold_for_alarm_budget(scores: np.ndarray, budget: float) -> AlarmThresholdSelection:
    'Choose a threshold whose step alarm rate best matches the target budget.'

    local_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if local_scores.size == 0:
        raise ValueError("scores must contain at least one value.")
    if not np.all(np.isfinite(local_scores)):
        raise ValueError("scores must be finite.")

    target_budget = float(budget)
    sentinel_threshold = float(np.nextafter(float(local_scores.max()), math.inf))
    candidates = [sentinel_threshold, *sorted({float(value) for value in local_scores.tolist()}, reverse=True)]
    best: tuple[float, float, float] | None = None
    best_payload: AlarmThresholdSelection | None = None
    for threshold in candidates:
        alarm_rate = float(np.mean(local_scores >= float(threshold)))
        abs_diff = abs(alarm_rate - target_budget)
        key = (abs_diff, alarm_rate, -float(threshold))
        if best is None or key < best:
            best = key
            best_payload = AlarmThresholdSelection(
                threshold=float(threshold),
                alarm_rate=alarm_rate,
                abs_diff=abs_diff,
                candidate_count=len(candidates),
            )
    if best_payload is None:
        raise RuntimeError("Failed to choose an alarm threshold.")
    return best_payload


def summarize_episode_alarm_utility(
    *,
    scores: np.ndarray,
    threshold: float,
    episode_ids: np.ndarray,
    outcome_labels: np.ndarray,
    steps_remaining: np.ndarray,
) -> EpisodeAlarmSummary:
    'Summarize budget-conditioned utility from transition-level scores.'

    local_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    local_episode_ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    local_outcome_labels = np.asarray(outcome_labels, dtype=np.int64).reshape(-1)
    local_steps_remaining = np.asarray(steps_remaining, dtype=np.float64).reshape(-1)
    if not (
        local_scores.size
        == local_episode_ids.size
        == local_outcome_labels.size
        == local_steps_remaining.size
    ):
        raise ValueError("All actionability inputs must have the same length.")
    if local_scores.size == 0:
        return EpisodeAlarmSummary(
            step_alarm_rate=0.0,
            episode_alarm_rate=0.0,
            catastrophic_episode_recall=0.0,
            noncat_episode_false_alarm_rate=0.0,
            episode_alarm_precision=0.0,
            recalled_catastrophic_lead_time_median=0.0,
            recalled_catastrophic_lead_time_mean=0.0,
            catastrophic_episode_count=0,
            noncat_episode_count=0,
            alarmed_episode_count=0,
            recalled_catastrophic_episode_count=0,
        )

    step_alarm_mask = local_scores >= float(threshold)
    step_alarm_rate = float(np.mean(step_alarm_mask))

    episode_alarm_flags: list[int] = []
    episode_cat_flags: list[int] = []
    recalled_lead_times: list[float] = []

    for episode_id in np.unique(local_episode_ids):
        mask = local_episode_ids == int(episode_id)
        episode_catastrophic = int(np.max(local_outcome_labels[mask]) == 2)
        local_alarm = np.flatnonzero(step_alarm_mask[mask])
        alarmed = int(local_alarm.size > 0)
        episode_alarm_flags.append(alarmed)
        episode_cat_flags.append(episode_catastrophic)
        if alarmed and episode_catastrophic:
            masked_steps = local_steps_remaining[mask]
            recalled_lead_times.append(float(masked_steps[int(local_alarm[0])]))

    episode_alarm_arr = np.asarray(episode_alarm_flags, dtype=np.int64)
    episode_cat_arr = np.asarray(episode_cat_flags, dtype=np.int64)
    alarmed_episode_count = int(episode_alarm_arr.sum())
    catastrophic_episode_count = int(episode_cat_arr.sum())
    noncat_episode_count = int(episode_cat_arr.size - catastrophic_episode_count)
    recalled_catastrophic_episode_count = int(np.sum((episode_alarm_arr == 1) & (episode_cat_arr == 1)))
    false_alarm_count = int(np.sum((episode_alarm_arr == 1) & (episode_cat_arr == 0)))

    episode_alarm_rate = float(episode_alarm_arr.mean()) if episode_alarm_arr.size else 0.0
    catastrophic_episode_recall = (
        float(recalled_catastrophic_episode_count / catastrophic_episode_count) if catastrophic_episode_count > 0 else 0.0
    )
    noncat_episode_false_alarm_rate = (
        float(false_alarm_count / noncat_episode_count) if noncat_episode_count > 0 else 0.0
    )
    episode_alarm_precision = (
        float(recalled_catastrophic_episode_count / alarmed_episode_count) if alarmed_episode_count > 0 else 0.0
    )
    if recalled_lead_times:
        lead_arr = np.asarray(recalled_lead_times, dtype=np.float64)
        lead_median = float(np.median(lead_arr))
        lead_mean = float(np.mean(lead_arr))
    else:
        lead_median = 0.0
        lead_mean = 0.0

    return EpisodeAlarmSummary(
        step_alarm_rate=step_alarm_rate,
        episode_alarm_rate=episode_alarm_rate,
        catastrophic_episode_recall=catastrophic_episode_recall,
        noncat_episode_false_alarm_rate=noncat_episode_false_alarm_rate,
        episode_alarm_precision=episode_alarm_precision,
        recalled_catastrophic_lead_time_median=lead_median,
        recalled_catastrophic_lead_time_mean=lead_mean,
        catastrophic_episode_count=catastrophic_episode_count,
        noncat_episode_count=noncat_episode_count,
        alarmed_episode_count=alarmed_episode_count,
        recalled_catastrophic_episode_count=recalled_catastrophic_episode_count,
    )


def parse_budget_values(values: Iterable[float | str] | None) -> list[float]:
    if values is None:
        return list(DEFAULT_ACTIONABILITY_BUDGETS)
    result = sorted({float(value) for value in values})
    if not result:
        raise ValueError("At least one actionability budget is required.")
    for value in result:
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"Budgets must be in (0, 1), got {value}.")
    return result
