'Metrics for precursor prediction and diagnostic evaluation.'

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _validate_binary_inputs(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length.")
    if scores.size == 0:
        return scores, labels
    labels = (labels > 0).astype(np.int64)
    return scores, labels


def binary_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    'Compute ROC-AUC for binary labels without sklearn.'
    scores, labels = _validate_binary_inputs(scores, labels)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    pos_ranks = ranks[labels == 1].sum()
    auc = (pos_ranks - positives * (positives + 1) / 2.0) / float(positives * negatives)
    return float(auc)


def binary_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    'Compute average precision for binary labels without sklearn.'
    scores, labels = _validate_binary_inputs(scores, labels)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = np.sum(precision * (recall - recall_prev) * sorted_labels)
    return float(ap)


def binary_brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    'Compute binary Brier score.'
    scores, labels = _validate_binary_inputs(scores, labels)
    if scores.size == 0:
        return 0.0
    clipped = np.clip(scores, 0.0, 1.0)
    return float(np.mean((clipped - labels.astype(np.float64)) ** 2))


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    'Compute binary expected calibration error.'
    scores, labels = _validate_binary_inputs(scores, labels)
    if scores.size == 0:
        return 0.0
    clipped = np.clip(scores, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    ece = 0.0
    total = float(scores.size)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper >= 1.0:
            mask = (clipped >= lower) & (clipped <= upper)
        else:
            mask = (clipped >= lower) & (clipped < upper)
        count = int(mask.sum())
        if count <= 0:
            continue
        avg_conf = float(clipped[mask].mean())
        avg_acc = float(labels[mask].mean())
        ece += abs(avg_conf - avg_acc) * (count / total)
    return float(ece)


@dataclass
class LeadTimeSummary:
    'Lead-time summary for catastrophic episodes.'

    count: int
    median: float
    mean: float


def summarize_lead_time(
    scores: np.ndarray,
    threshold: float,
    catastrophic_episode_mask: np.ndarray,
    episode_ids: np.ndarray,
    steps_remaining: np.ndarray,
) -> LeadTimeSummary:
    'Return lead-time summary for episodes whose outcome is catastrophic.'
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    catastrophic_episode_mask = np.asarray(catastrophic_episode_mask, dtype=bool).reshape(-1)
    episode_ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    steps_remaining = np.asarray(steps_remaining, dtype=np.float64).reshape(-1)
    if not (scores.size == catastrophic_episode_mask.size == episode_ids.size == steps_remaining.size):
        raise ValueError("All lead-time inputs must have the same length.")

    lead_times: list[float] = []
    for episode_id in np.unique(episode_ids[catastrophic_episode_mask]):
        mask = episode_ids == episode_id
        above = np.flatnonzero(mask & (scores >= float(threshold)))
        if above.size <= 0:
            continue
        first_idx = int(above[0])
        lead_times.append(float(steps_remaining[first_idx]))
    if not lead_times:
        return LeadTimeSummary(count=0, median=0.0, mean=0.0)
    arr = np.asarray(lead_times, dtype=np.float64)
    return LeadTimeSummary(count=int(arr.size), median=float(np.median(arr)), mean=float(arr.mean()))


def prevalence(labels: Iterable[int]) -> float:
    'Compute positive prevalence for binary labels.'
    arr = np.asarray(list(labels), dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr > 0))
