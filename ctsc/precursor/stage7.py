'Frozen independent train-seed matrix and integrity checks.'

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ctsc.precursor.cc1 import LOCKED_MODEL_LRS
from ctsc.precursor.revision import canonical_json_sha256, file_sha256


TASKS = (
    "BallRun", "AntCircle", "mediumsparse", "easydense",
    "PointCircle2", "CarPush2", "SwimmerVelocity", "AntVelocity",
)
MODELS = ("gru", "flat_mlp", "transformer")
TRAIN_SEEDS = (0, 1, 2, 3, 4)
FIXED_SEED = 0


def frozen_jobs() -> list[dict[str, Any]]:
    return [
        {"task": task, "model": model, "split_seed": 0, "sampling_seed": 0,
         "loader_seed": 0, "train_seed": train_seed,
         "learning_rate": LOCKED_MODEL_LRS[model],
         "source": "G5A_REUSE" if train_seed == 0 else "STAGE7_NEW"}
        for task in TASKS for train_seed in TRAIN_SEEDS for model in MODELS
    ]


def validate_parent_run(metrics_path: Path, *, task: str, model: str, train_seed: int,
                        shared_record: Mapping[str, Any], dataset_sha256: str) -> dict[str, Any]:
    manifest_path = metrics_path.with_name("run_manifest.json")
    bundle_path = metrics_path.with_name("precursor_model.pt")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    training = manifest["training"]
    expected_seeds = {"split_seed": 0, "sampling_seed": 0, "train_seed": train_seed, "loader_seed": 0}
    if any(int(protocol[key]) != value for key, value in expected_seeds.items()):
        raise ValueError(f"Stage 7 seed mismatch: {metrics_path}")
    if protocol.get("legacy_compatible") is not False or training.get("loader_rng_mode") != "explicit_loader_seed":
        raise ValueError(f"Stage 7 requires explicit four-stream protocol: {metrics_path}")
    if manifest["dataset"]["task"] != task or manifest["dataset"]["sha256"] != dataset_sha256:
        raise ValueError(f"Dataset mismatch: {metrics_path}")
    if str(protocol["model_name"]).replace("-", "_") != model:
        raise ValueError(f"Model mismatch: {metrics_path}")
    if not math.isclose(float(training["learning_rate"]), LOCKED_MODEL_LRS[model], rel_tol=0, abs_tol=1e-15):
        raise ValueError(f"Locked learning rate mismatch: {metrics_path}")
    if protocol["feature_profile"] != "full" or protocol["split_family"] != "episode_group":
        raise ValueError(f"Canonical split/feature mismatch: {metrics_path}")
    if protocol["target_variant"] != "current_inclusive" or float(protocol["gamma"]) != 0.99:
        raise ValueError(f"Canonical target mismatch: {metrics_path}")
    if list(protocol["quantile_pair"]) != [0.75, 0.9] or list(protocol["outcome_quantile_pair"]) != [0.75, 0.9]:
        raise ValueError(f"Canonical quantiles mismatch: {metrics_path}")
    if training["epochs"] != 12 or training["window_size"] != 32 or training["hidden_dim"] != 128:
        raise ValueError(f"Training configuration mismatch: {metrics_path}")
    if manifest["indices"] != shared_record:
        raise ValueError(f"Shared indices/normalization record mismatch: {metrics_path}")
    for name, path in (("metrics.json", metrics_path), ("precursor_model.pt", bundle_path)):
        info = manifest["artifacts"][name]
        if file_sha256(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Run artifact hash mismatch: {path}")
    history = metrics["training"]["history"]
    if len(history) != 12 or [int(row["epoch"]) for row in history] != list(range(12)):
        raise ValueError(f"Incomplete training history: {metrics_path}")
    values = np.asarray([float(row["val_catastrophic_auprc"]) for row in history])
    if not np.isfinite(values).all() or int(metrics["training"]["best_epoch"]) != int(np.argmax(values)):
        raise ValueError(f"Non-finite or inconsistent best epoch: {metrics_path}")
    score = float(metrics["label_step"]["test"]["overall"]["catastrophic_auprc"])
    prevalence = float(metrics["label_step"]["test"]["overall"]["catastrophic_prevalence"])
    if not (math.isfinite(score) and math.isfinite(prevalence)):
        raise ValueError(f"Non-finite test metric: {metrics_path}")
    return {
        "task": task, "model": model, "train_seed": train_seed,
        "step_auprc": score, "step_prevalence": prevalence,
        "step_lift": score - prevalence,
        "best_epoch": int(metrics["training"]["best_epoch"]),
        "metrics_sha256": file_sha256(metrics_path),
        "bundle_sha256": file_sha256(bundle_path),
        "run_manifest_sha256": file_sha256(manifest_path),
        "batch_size": int(training["batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
    }


def validate_shared_files(record: Mapping[str, Any], shared_dir: Path) -> None:
    for name, info in record["files"].items():
        path = shared_dir / name
        if not path.is_file() or file_sha256(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Frozen G5A shared artifact mismatch: {path}")


