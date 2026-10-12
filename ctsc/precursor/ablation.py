'Unified38 current-facing ablation mode definitions and helpers.'

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ABLATION_GROUP_REGISTRY = ("task_framing", "labels")
ABLATION_GROUP_MODE_REGISTRY: dict[str, tuple[str, ...]] = {
    "task_framing": (
        "step_level_current_3way",
        "episode_outcome_step_broadcast",
        "episode_outcome_terminal_only",
    ),
    "labels": (
        "step_3way_q75_q90",
        "binary_unsafe_vs_safe",
        "binary_cat_vs_rest",
    ),
}


@dataclass(frozen=True)
class Unified38AblationSpec:
    'Resolved behavior contract for one current-facing ablation mode.'

    group: str | None
    mode: str | None
    training_target_type: str
    selection_target_type: str
    primary_eval_target: str
    secondary_eval_target: str
    train_mask_mode: str
    positive_index: int
    num_classes: int


DEFAULT_ABLATION_SPEC = Unified38AblationSpec(
    group=None,
    mode=None,
    training_target_type="step_3way",
    selection_target_type="step_catastrophic",
    primary_eval_target="canonical_step_catastrophic",
    secondary_eval_target="canonical_outcome_catastrophic",
    train_mask_mode="all",
    positive_index=2,
    num_classes=3,
)


ABLATION_MODE_SPECS: dict[str, Unified38AblationSpec] = {
    "step_level_current_3way": Unified38AblationSpec(
        group="task_framing",
        mode="step_level_current_3way",
        training_target_type="step_3way",
        selection_target_type="step_catastrophic",
        primary_eval_target="canonical_step_catastrophic",
        secondary_eval_target="canonical_outcome_catastrophic",
        train_mask_mode="all",
        positive_index=2,
        num_classes=3,
    ),
    "episode_outcome_step_broadcast": Unified38AblationSpec(
        group="task_framing",
        mode="episode_outcome_step_broadcast",
        training_target_type="outcome_3way",
        selection_target_type="outcome_catastrophic",
        primary_eval_target="canonical_step_catastrophic",
        secondary_eval_target="canonical_outcome_catastrophic",
        train_mask_mode="all",
        positive_index=2,
        num_classes=3,
    ),
    "episode_outcome_terminal_only": Unified38AblationSpec(
        group="task_framing",
        mode="episode_outcome_terminal_only",
        training_target_type="outcome_3way",
        selection_target_type="outcome_catastrophic",
        primary_eval_target="canonical_step_catastrophic",
        secondary_eval_target="canonical_outcome_catastrophic",
        train_mask_mode="terminal_only",
        positive_index=2,
        num_classes=3,
    ),
    "step_3way_q75_q90": Unified38AblationSpec(
        group="labels",
        mode="step_3way_q75_q90",
        training_target_type="step_3way",
        selection_target_type="step_catastrophic",
        primary_eval_target="canonical_step_catastrophic",
        secondary_eval_target="canonical_outcome_catastrophic",
        train_mask_mode="all",
        positive_index=2,
        num_classes=3,
    ),
    "binary_unsafe_vs_safe": Unified38AblationSpec(
        group="labels",
        mode="binary_unsafe_vs_safe",
        training_target_type="step_binary_unsafe_vs_safe",
        selection_target_type="step_unsafe_vs_safe",
        primary_eval_target="canonical_step_catastrophic",
        secondary_eval_target="canonical_outcome_catastrophic",
        train_mask_mode="all",
        positive_index=1,
        num_classes=2,
    ),
    "binary_cat_vs_rest": Unified38AblationSpec(
        group="labels",
        mode="binary_cat_vs_rest",
        training_target_type="step_binary_catastrophic",
        selection_target_type="step_catastrophic",
        primary_eval_target="canonical_step_catastrophic",
        secondary_eval_target="canonical_outcome_catastrophic",
        train_mask_mode="all",
        positive_index=1,
        num_classes=2,
    ),
}


def resolve_unified38_ablation_spec(ablation_group: str | None, ablation_mode: str | None) -> Unified38AblationSpec:
    'Return the resolved ablation behavior or the default mainline contract.'
    group = None if ablation_group in {None, ""} else str(ablation_group)
    mode = None if ablation_mode in {None, ""} else str(ablation_mode)
    if group is None and mode is None:
        return DEFAULT_ABLATION_SPEC
    if group is None or mode is None:
        raise ValueError("ablation_group and ablation_mode must be provided together.")
    if group not in ABLATION_GROUP_MODE_REGISTRY:
        raise ValueError(f"Unsupported ablation_group `{group}`.")
    if mode not in ABLATION_GROUP_MODE_REGISTRY[group]:
        raise ValueError(f"Unsupported ablation_mode `{mode}` for group `{group}`.")
    return ABLATION_MODE_SPECS[mode]


def build_ablation_training_labels(
    *,
    step_labels: np.ndarray,
    outcome_labels: np.ndarray,
    spec: Unified38AblationSpec,
) -> np.ndarray:
    'Construct train/calibration labels aligned with the model head size.'
    if spec.training_target_type == "step_3way":
        return np.asarray(step_labels, dtype=np.int64)
    if spec.training_target_type == "outcome_3way":
        return np.asarray(outcome_labels, dtype=np.int64)
    if spec.training_target_type == "step_binary_unsafe_vs_safe":
        return (np.asarray(step_labels, dtype=np.int64) >= 1).astype(np.int64)
    if spec.training_target_type == "step_binary_catastrophic":
        return (np.asarray(step_labels, dtype=np.int64) == 2).astype(np.int64)
    raise ValueError(f"Unsupported training_target_type={spec.training_target_type}")


def build_ablation_selection_eval_labels(
    *,
    step_labels: np.ndarray,
    outcome_labels: np.ndarray,
    spec: Unified38AblationSpec,
) -> np.ndarray:
    'Construct binary labels used for validation selection and reporting.'
    if spec.selection_target_type == "step_catastrophic":
        return (np.asarray(step_labels, dtype=np.int64) == 2).astype(np.int64)
    if spec.selection_target_type == "outcome_catastrophic":
        return (np.asarray(outcome_labels, dtype=np.int64) == 2).astype(np.int64)
    if spec.selection_target_type == "step_unsafe_vs_safe":
        return (np.asarray(step_labels, dtype=np.int64) >= 1).astype(np.int64)
    raise ValueError(f"Unsupported selection_target_type={spec.selection_target_type}")


def apply_ablation_train_mask(
    dataset: dict[str, Any],
    splits: dict[str, Any],
    *,
    spec: Unified38AblationSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    'Filter train/val splits for ablation modes such as terminal-only.'
    effective_splits: dict[str, Any] = {"manifest": dict(splits["manifest"])}
    if spec.train_mask_mode == "all":
        for split_name in ("train", "val", "test"):
            effective_splits[split_name] = np.asarray(splits[split_name], dtype=np.int64)
        return effective_splits, {
            "train_mask_mode": "all",
            "mask_applies_to": [],
            "split_filtering": {
                split_name: {
                    "original_count": int(np.asarray(splits[split_name], dtype=np.int64).shape[0]),
                    "effective_count": int(np.asarray(splits[split_name], dtype=np.int64).shape[0]),
                    "dropped_count": 0,
                    "dropped_ratio": 0.0,
                }
                for split_name in ("train", "val", "test")
            },
        }

    if spec.train_mask_mode != "terminal_only":
        raise ValueError(f"Unsupported train_mask_mode={spec.train_mask_mode}")

    terminal_mask = np.isclose(np.asarray(dataset["steps_remaining"], dtype=np.float64), 0.0)
    split_filtering: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        original = np.asarray(splits[split_name], dtype=np.int64)
        if split_name == "test":
            kept = original
        else:
            kept = original[terminal_mask[original]]
        effective_splits[split_name] = np.asarray(kept, dtype=np.int64)
        dropped = int(original.shape[0] - kept.shape[0])
        split_filtering[split_name] = {
            "original_count": int(original.shape[0]),
            "effective_count": int(kept.shape[0]),
            "dropped_count": dropped,
            "dropped_ratio": float(dropped / original.shape[0]) if original.shape[0] else 0.0,
        }
    return effective_splits, {
        "train_mask_mode": "terminal_only",
        "mask_applies_to": ["train", "val"],
        "split_filtering": split_filtering,
    }
