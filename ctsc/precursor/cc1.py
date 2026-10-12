'explicit-RNG bridge and fair-matrix definitions.'

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ctsc.precursor.revision import canonical_json_sha256, file_sha256


AMENDMENT_ID = "OTTP-R26-CC1-v1"
PARENT_FREEZE_ID = "OTTP-R26-G1-v1"
TASK_COUNT = 38
SPLIT_SEEDS = (0, 1, 2)
BRIDGE_MODELS = ("gru", "flat_mlp")
FAIR_MODELS = ("gru", "flat_mlp", "transformer")
LOCKED_MODEL_LRS = {"gru": 3e-4, "flat_mlp": 3e-3, "transformer": 3e-3}


@dataclass(frozen=True)
class CC1Condition:
    condition_id: str
    feature_profile: str
    q_low: float
    q_high: float
    role: str


BRIDGE_CONDITION = CC1Condition("canonical_full", "full", 0.75, 0.90, "stage4_rng_bridge")
FAIR_CONDITIONS = (
    CC1Condition("canonical_full", "full", 0.75, 0.90, "confirmatory"),
    CC1Condition("sensitivity_full", "full", 0.80, 0.95, "descriptive_sensitivity"),
    CC1Condition("canonical_current_only", "current_only", 0.75, 0.90, "confirmatory"),
)
CONDITIONS_BY_ID = {condition.condition_id: condition for condition in FAIR_CONDITIONS}


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _validate_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(str(task) for task in tasks)
    if len(ordered) != TASK_COUNT or len(set(ordered)) != TASK_COUNT:
        raise ValueError(f"CC1 requires exactly {TASK_COUNT} unique tasks")
    return ordered


def bridge_cells(tasks: Sequence[str]) -> list[dict[str, Any]]:
    ordered = _validate_tasks(tasks)
    return [
        {
            "cell_id": f"bridge__{task}__ss{seed}",
            "matrix": "stage4_seed_bridge",
            "condition_id": BRIDGE_CONDITION.condition_id,
            "task_name": task,
            "split_seed": int(seed),
            "models": list(BRIDGE_MODELS),
            "model_learning_rates": {model: 1e-3 for model in BRIDGE_MODELS},
        }
        for task in ordered
        for seed in SPLIT_SEEDS
    ]


def fair_cells(tasks: Sequence[str]) -> list[dict[str, Any]]:
    ordered = _validate_tasks(tasks)
    return [
        {
            "cell_id": f"fair__{condition.condition_id}__{task}__ss{seed}",
            "matrix": "stage5_fair_matrix",
            "condition_id": condition.condition_id,
            "task_name": task,
            "split_seed": int(seed),
            "models": list(FAIR_MODELS),
            "model_learning_rates": dict(LOCKED_MODEL_LRS),
        }
        for condition in FAIR_CONDITIONS
        for task in ordered
        for seed in SPLIT_SEEDS
    ]



