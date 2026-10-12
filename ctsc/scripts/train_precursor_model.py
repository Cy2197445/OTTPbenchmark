'Train one Unified38 precursor baseline.'

from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from ctsc.precursor.data import (
    FEATURE_PROFILE_REGISTRY,
    SPLIT_FAMILY_REGISTRY,
    FeatureMatrixSequenceDataset,
    FeatureMatrixWindowFlatDataset,
    assign_3way_labels,
    build_feature_profile_matrix,
    build_splits,
    compute_label_thresholds,
    compute_profile_normalization_stats,
    feature_profile_current_cost_index,
    load_precursor_dataset,
    sequence_view_for_profile,
    summarize_split_diagnostics,
)
from ctsc.precursor.ablation import (
    ABLATION_GROUP_REGISTRY,
    Unified38AblationSpec,
    apply_ablation_train_mask,
    build_ablation_selection_eval_labels,
    build_ablation_training_labels,
    resolve_unified38_ablation_spec,
)
from ctsc.precursor.metrics import (
    binary_average_precision,
    binary_brier_score,
    binary_roc_auc,
    expected_calibration_error,
    prevalence,
    summarize_lead_time,
)
from ctsc.precursor.model import (
    PrevalenceBaseline,
    TemperatureScaler,
    build_precursor_model,
    get_precursor_model_spec,
    select_transformer_d_model,
)
from ctsc.precursor.revision import (
    SeedBundle,
    assert_valid_run_manifest,
    export_reproducibility_artifacts,
    file_sha256,
    guard_revision_output_dir,
    protocol_identity,
    require_free_space,
    resolve_seed_bundle,
    seeded_output_path,
    source_snapshot_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_SOURCE_FILES = (
    "ctsc/precursor/data.py",
    "ctsc/precursor/model.py",
    "ctsc/precursor/metrics.py",
    "ctsc/precursor/revision.py",
    "ctsc/scripts/train_precursor_model.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["gru", "transformer", "temporal-cnn", "flat-mlp", "linear", "prevalence"], default="gru")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--loader-seed", type=int, default=None)
    parser.add_argument("--split-family", type=str, choices=list(SPLIT_FAMILY_REGISTRY), default="episode_group")
    parser.add_argument("--ablation-group", type=str, choices=list(ABLATION_GROUP_REGISTRY), default=None)
    parser.add_argument("--ablation-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-profile", type=str, choices=list(FEATURE_PROFILE_REGISTRY), default="full")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--transformer-d-model", type=int, default=None)
    parser.add_argument("--transformer-num-layers", type=int, default=2)
    parser.add_argument("--transformer-num-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-multiplier", type=int, default=4)
    parser.add_argument("--transformer-max-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--step-q-low", type=float, default=0.75)
    parser.add_argument("--step-q-high", type=float, default=0.90)
    parser.add_argument("--outcome-q-low", type=float, default=0.75)
    parser.add_argument("--outcome-q-high", type=float, default=0.90)
    parser.add_argument("--critic-baseline-mode", type=str, choices=["dataset", "disabled"], default="dataset")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--target-variant", choices=["current_inclusive", "future_only"], default="current_inclusive")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--label-sidecar", type=str, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--attempt-id", type=str, default="attempt-001")
    parser.add_argument("--work-package", type=str, default="revision")
    parser.add_argument("--source-snapshot-hash", type=str, default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--minimum-free-space-gib", type=float, default=10.0)
    return parser.parse_args()


@dataclass
class PreparedPrecursorInputs:
    dataset_path: Path
    dataset_sha256: str
    dataset: dict[str, Any]
    env_name_map: dict[int, str]
    base_splits: dict[str, Any]
    uncapped_splits: dict[str, Any]
    splits: dict[str, Any]
    step_thresholds: dict[int, dict[str, float]]
    step_labels: np.ndarray
    outcome_thresholds: dict[int, dict[str, float]]
    outcome_labels: np.ndarray
    training_labels: np.ndarray
    selection_eval_labels: np.ndarray
    normalized_features: np.ndarray
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    feature_profile: str
    seed: int
    seeds: SeedBundle
    target_variant: str
    gamma: float
    split_family: str
    ablation_group: str | None
    ablation_mode: str | None
    ablation_spec: Unified38AblationSpec
    split_diagnostics: dict[str, Any]
    ablation_filter_audit: dict[str, Any]
    label_protocol: dict[str, Any]


def _make_env_name_map(dataset: dict[str, Any]) -> dict[int, str]:
    raw = dataset["metadata"]["env_names"]
    return {int(key): str(value) for key, value in raw.items()}


def _class_weights(labels: np.ndarray, *, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=int(num_classes)).astype(np.float64)
    present = counts > 0
    if not np.any(present):
        raise ValueError(f"Training split must contain at least one observed class, got counts={counts.tolist()}")
    weights = np.zeros_like(counts, dtype=np.float64)
    weights[present] = counts[present].sum() / (counts[present] * int(present.sum()))
    weights[present] = weights[present] / weights[present].mean()
    return torch.as_tensor(weights.astype(np.float32))


class _ExplicitLegacyRandomSampler(Sampler[int]):
    "Explicit RNG sampler matching PyTorch's historical generator=None path."

    def __init__(self, data_source: Dataset[Any], generator: torch.Generator) -> None:
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        epoch_seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())
        epoch_generator = torch.Generator(device="cpu")
        epoch_generator.manual_seed(epoch_seed)
        yield from torch.randperm(len(self.data_source), generator=epoch_generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def _make_loader(
    model_dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    generator: torch.Generator | None = None,
) -> DataLoader:
    sampler = _ExplicitLegacyRandomSampler(model_dataset, generator) if shuffle and generator is not None else None
    return DataLoader(
        model_dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=int(num_workers),
        drop_last=False,
        generator=generator,
    )


def _cap_indices(indices: np.ndarray, limit: int | None, *, seed: int) -> np.ndarray:
    base = np.asarray(indices, dtype=np.int64)
    if limit is None or int(limit) <= 0 or base.shape[0] <= int(limit):
        return base
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(base, size=int(limit), replace=False))
    return chosen.astype(np.int64)


def _batch_inputs(batch: dict[str, torch.Tensor], *, input_view: str, device: torch.device) -> torch.Tensor:
    if input_view == "sequence":
        return batch["sequence"].to(device)
    if input_view == "flat":
        return batch["flat_features"].to(device)
    raise ValueError(f"Unsupported input_view={input_view}")


def _collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    input_view: str,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    logits_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    critic_list: list[np.ndarray] = []
    env_list: list[np.ndarray] = []
    episode_uid_list: list[np.ndarray] = []
    steps_remaining_list: list[np.ndarray] = []
    sample_index_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(_batch_inputs(batch, input_view=input_view, device=device))
            scaled_logits = logits / max(float(temperature), 1e-4)
            probs = torch.softmax(scaled_logits, dim=-1)
            logits_list.append(logits.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            labels_list.append(batch["label"].cpu().numpy())
            critic_list.append(batch["critic_score"].cpu().numpy())
            env_list.append(batch["env_index"].cpu().numpy())
            episode_uid_list.append(batch["episode_uid"].cpu().numpy())
            steps_remaining_list.append(batch["steps_remaining"].cpu().numpy())
            sample_index_list.append(batch["sample_index"].cpu().numpy())
    return {
        "logits": np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, 3), dtype=np.float32),
        "probabilities": np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, 3), dtype=np.float32),
        "labels": np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0,), dtype=np.int64),
        "critic_score": np.concatenate(critic_list, axis=0) if critic_list else np.zeros((0,), dtype=np.float32),
        "env_index": np.concatenate(env_list, axis=0) if env_list else np.zeros((0,), dtype=np.int64),
        "episode_uid": np.concatenate(episode_uid_list, axis=0) if episode_uid_list else np.zeros((0,), dtype=np.int64),
        "steps_remaining": np.concatenate(steps_remaining_list, axis=0) if steps_remaining_list else np.zeros((0,), dtype=np.float32),
        "sample_index": np.concatenate(sample_index_list, axis=0) if sample_index_list else np.zeros((0,), dtype=np.int64),
    }


def _collect_reference_outputs(
    dataset: dict[str, Any],
    indices: np.ndarray,
    *,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float32).reshape(1, -1).repeat(idx.shape[0], axis=0)
    logits = np.log(np.clip(probs, 1e-6, 1.0)).astype(np.float32)
    return {
        "logits": logits,
        "probabilities": probs,
        "labels": np.zeros((idx.shape[0],), dtype=np.int64),
        "critic_score": np.asarray(dataset["critic_score"], dtype=np.float32)[idx],
        "env_index": np.asarray(dataset["env_index"], dtype=np.int64)[idx],
        "episode_uid": np.asarray(dataset["episode_uid"], dtype=np.int64)[idx],
        "steps_remaining": np.asarray(dataset["steps_remaining"], dtype=np.float32)[idx],
        "sample_index": idx,
    }


def _binary_eval_labels(labels: np.ndarray, *, positive_label: int = 2) -> np.ndarray:
    return (np.asarray(labels, dtype=np.int64).reshape(-1) == int(positive_label)).astype(np.int64)


def _extract_positive_scores(probabilities: np.ndarray, positive_index: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be rank-2.")
    index = int(positive_index)
    if index < 0 or index >= probs.shape[1]:
        raise ValueError(f"positive_index out of bounds: {index} for shape={probs.shape}")
    return probs[:, index].reshape(-1)


def _summarize_binary_scores(
    *,
    scores: np.ndarray,
    binary_labels: np.ndarray,
    env_indices: np.ndarray,
    env_name_map: dict[int, str],
    episode_uid: np.ndarray,
    steps_remaining: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(binary_labels, dtype=np.int64).reshape(-1)

    def metrics_for(mask: np.ndarray) -> dict[str, float]:
        local_scores = np.asarray(scores[mask], dtype=np.float64)
        local_labels = np.asarray(labels[mask], dtype=np.int64)
        lead = summarize_lead_time(
            scores=local_scores,
            threshold=float(threshold),
            catastrophic_episode_mask=local_labels.astype(bool),
            episode_ids=np.asarray(episode_uid[mask], dtype=np.int64),
            steps_remaining=np.asarray(steps_remaining[mask], dtype=np.float64),
        )
        return {
            "catastrophic_prevalence": prevalence(local_labels),
            "catastrophic_auprc": binary_average_precision(local_scores, local_labels),
            "catastrophic_auroc": binary_roc_auc(local_scores, local_labels),
            "catastrophic_brier": binary_brier_score(local_scores, local_labels),
            "catastrophic_ece": expected_calibration_error(local_scores, local_labels),
            "lead_time_median_at_0.5": float(lead.median),
            "lead_time_mean_at_0.5": float(lead.mean),
        }

    overall = metrics_for(np.ones_like(labels, dtype=bool))
    by_env: dict[str, dict[str, float]] = {}
    for env_id in sorted(np.unique(env_indices).tolist()):
        env_mask = np.asarray(env_indices == int(env_id))
        by_env[env_name_map[int(env_id)]] = metrics_for(env_mask)
    return {"overall": overall, "by_env": by_env}


def _summarize_model_scores(
    scores: np.ndarray,
    labels_3way: np.ndarray,
    env_indices: np.ndarray,
    env_name_map: dict[int, str],
    episode_uid: np.ndarray,
    steps_remaining: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    return _summarize_binary_scores(
        scores=scores,
        binary_labels=_binary_eval_labels(labels_3way, positive_label=2),
        env_indices=env_indices,
        env_name_map=env_name_map,
        episode_uid=episode_uid,
        steps_remaining=steps_remaining,
        threshold=threshold,
    )


def _summarize_critic_scores(scores: np.ndarray, labels_3way: np.ndarray, env_indices: np.ndarray, env_name_map: dict[int, str]) -> dict[str, Any]:
    cat_labels = _binary_eval_labels(labels_3way, positive_label=2)

    def metrics_for(mask: np.ndarray) -> dict[str, float]:
        local_scores = np.asarray(scores[mask], dtype=np.float64)
        local_cat = np.asarray(cat_labels[mask], dtype=np.int64)
        return {
            "catastrophic_prevalence": prevalence(local_cat),
            "catastrophic_auprc": binary_average_precision(local_scores, local_cat),
            "catastrophic_auroc": binary_roc_auc(local_scores, local_cat),
        }

    overall = metrics_for(np.ones_like(cat_labels, dtype=bool))
    by_env: dict[str, dict[str, float]] = {}
    for env_id in sorted(np.unique(env_indices).tolist()):
        env_mask = np.asarray(env_indices == int(env_id))
        by_env[env_name_map[int(env_id)]] = metrics_for(env_mask)
    return {"overall": overall, "by_env": by_env}


def _evaluate_against_labels(
    *,
    outputs: dict[str, np.ndarray],
    eval_labels: np.ndarray,
    positive_index: int,
    env_name_map: dict[int, str],
) -> dict[str, Any]:
    return _summarize_binary_scores(
        scores=_extract_positive_scores(outputs["probabilities"], positive_index),
        binary_labels=_binary_eval_labels(eval_labels[outputs["sample_index"]], positive_label=2),
        env_indices=outputs["env_index"],
        env_name_map=env_name_map,
        episode_uid=outputs["episode_uid"],
        steps_remaining=outputs["steps_remaining"],
        threshold=0.5,
    )


def _evaluate_selection_target(
    *,
    outputs: dict[str, np.ndarray],
    selection_binary_labels: np.ndarray,
    positive_index: int,
    env_name_map: dict[int, str],
) -> dict[str, Any]:
    return _summarize_binary_scores(
        scores=_extract_positive_scores(outputs["probabilities"], positive_index),
        binary_labels=selection_binary_labels[outputs["sample_index"]],
        env_indices=outputs["env_index"],
        env_name_map=env_name_map,
        episode_uid=outputs["episode_uid"],
        steps_remaining=outputs["steps_remaining"],
        threshold=0.5,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    input_view: str,
    gradient_accumulation_steps: int = 1,
) -> float:
    accumulation = int(gradient_accumulation_steps)
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    batch_count = len(loader)
    for batch_index, batch in enumerate(loader):
        logits = model(_batch_inputs(batch, input_view=input_view, device=device))
        loss = criterion(logits, batch["label"].to(device))
        (loss / float(accumulation)).backward()
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count
        if should_step:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def _build_model_datasets(
    dataset: dict[str, Any],
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    normalized_features: np.ndarray,
    feature_profile: str,
    model_name: str,
    window_size: int,
    shuffle_seed: int,
) -> dict[str, Dataset[dict[str, torch.Tensor]]]:
    spec = get_precursor_model_spec(model_name)
    sequence_view = sequence_view_for_profile(feature_profile)
    current_cost_index = feature_profile_current_cost_index(dataset, feature_profile)
    dataset_cls: type[Dataset[dict[str, torch.Tensor]]]
    if spec.input_view == "sequence":
        dataset_cls = FeatureMatrixSequenceDataset
    elif spec.input_view == "flat":
        dataset_cls = FeatureMatrixWindowFlatDataset
    else:
        raise ValueError(f"Unsupported trainable input_view={spec.input_view}")
    return {
        split_name: dataset_cls(
            features=normalized_features,
            dataset=dataset,
            indices=indices,
            labels=labels,
            window_size=int(window_size),
            sequence_view=sequence_view,
            shuffle_seed=int(shuffle_seed),
            current_cost_index=current_cost_index,
        )
        for split_name, indices in [("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]
    }


def _major_step_metric(metrics: dict[str, Any], split_name: str) -> float:
    return float(metrics["label_step"][split_name]["overall"]["catastrophic_auprc"])


def prepare_precursor_inputs(
    dataset_path: str | Path,
    *,
    seed: int,
    split_seed: int | None = None,
    sampling_seed: int | None = None,
    train_seed: int | None = None,
    loader_seed: int | None = None,
    split_family: str,
    ablation_group: str | None,
    ablation_mode: str | None,
    feature_profile: str,
    step_q_low: float,
    step_q_high: float,
    outcome_q_low: float,
    outcome_q_high: float,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    target_variant: str = "current_inclusive",
    gamma: float | None = None,
    label_sidecar_path: str | Path | None = None,
) -> PreparedPrecursorInputs:
    dataset_path = Path(dataset_path)
    seeds = resolve_seed_bundle(
        seed=int(seed),
        split_seed=split_seed,
        sampling_seed=sampling_seed,
        train_seed=train_seed,
        loader_seed=loader_seed,
    )
    dataset = load_precursor_dataset(dataset_path)
    env_name_map = _make_env_name_map(dataset)
    uncapped_splits = build_splits(dataset, seed=int(seeds.split_seed), split_family=str(split_family))
    base_splits = {
        "train": np.asarray(uncapped_splits["train"], dtype=np.int64).copy(),
        "val": np.asarray(uncapped_splits["val"], dtype=np.int64).copy(),
        "test": np.asarray(uncapped_splits["test"], dtype=np.int64).copy(),
        "manifest": deepcopy(uncapped_splits["manifest"]),
    }
    base_splits["train"] = _cap_indices(base_splits["train"], max_train_samples, seed=int(seeds.sampling_seed) + 11)
    base_splits["val"] = _cap_indices(base_splits["val"], max_val_samples, seed=int(seeds.sampling_seed) + 17)
    base_splits["test"] = _cap_indices(base_splits["test"], max_test_samples, seed=int(seeds.sampling_seed) + 23)
    split_diagnostics = summarize_split_diagnostics(dataset, base_splits)

    dataset_gamma = float(dataset.get("metadata", {}).get("protocol", {}).get("gamma", 0.99))
    resolved_gamma = dataset_gamma if gamma is None else float(gamma)
    step_values: np.ndarray
    if label_sidecar_path is None:
        if str(target_variant) != "current_inclusive" or not np.isclose(resolved_gamma, dataset_gamma, rtol=0.0, atol=1e-12):
            raise ValueError(
                "A label sidecar is required when target_variant or gamma differs from the base precursor dataset."
            )
        step_values = np.asarray(dataset["cost_to_go_disc"], dtype=np.float32)
    else:
        sidecar_path = Path(label_sidecar_path)
        manifest_path = sidecar_path.with_suffix(sidecar_path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing label sidecar manifest: {manifest_path}")
        sidecar_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(sidecar_manifest.get("base_dataset_sha256")) != file_sha256(dataset_path):
            raise ValueError("Label sidecar base_dataset_sha256 does not match the supplied precursor dataset.")
        if str(sidecar_manifest.get("target_variant")) != str(target_variant):
            raise ValueError("Label sidecar target_variant mismatch.")
        if not np.isclose(float(sidecar_manifest.get("gamma")), resolved_gamma, rtol=0.0, atol=1e-12):
            raise ValueError("Label sidecar gamma mismatch.")
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            step_values = np.asarray(sidecar["target_values"], dtype=np.float32)
        if step_values.shape[0] != np.asarray(dataset["costs"]).shape[0]:
            raise ValueError("Label sidecar length does not match base dataset.")

    step_thresholds = compute_label_thresholds(
        {**dataset, "revision_step_target": step_values},
        base_splits["train"],
        value_key="revision_step_target",
        lower_quantile=float(step_q_low),
        upper_quantile=float(step_q_high),
    )
    step_labels = assign_3way_labels(step_values, dataset["env_index"], step_thresholds)
    outcome_thresholds = compute_label_thresholds(
        dataset,
        base_splits["train"],
        value_key="episode_cost_disc",
        lower_quantile=float(outcome_q_low),
        upper_quantile=float(outcome_q_high),
    )
    outcome_labels = assign_3way_labels(dataset["episode_cost_disc"], dataset["env_index"], outcome_thresholds)
    ablation_spec = resolve_unified38_ablation_spec(ablation_group, ablation_mode)
    training_labels = build_ablation_training_labels(
        step_labels=step_labels,
        outcome_labels=outcome_labels,
        spec=ablation_spec,
    )
    selection_eval_labels = build_ablation_selection_eval_labels(
        step_labels=step_labels,
        outcome_labels=outcome_labels,
        spec=ablation_spec,
    )
    splits, ablation_filter_audit = apply_ablation_train_mask(
        dataset,
        base_splits,
        spec=ablation_spec,
    )

    feature_matrix = build_feature_profile_matrix(dataset, profile_name=feature_profile)
    normalization_stats = compute_profile_normalization_stats(
        dataset,
        base_splits["train"],
        profile_name=feature_profile,
        feature_matrix=feature_matrix,
    )
    norm_mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
    norm_std = np.asarray(normalization_stats["std"], dtype=np.float32)
    normalized_features = ((feature_matrix - norm_mean) / norm_std).astype(np.float32)
    return PreparedPrecursorInputs(
        dataset_path=dataset_path,
        dataset_sha256=file_sha256(dataset_path),
        dataset=dataset,
        env_name_map=env_name_map,
        base_splits=base_splits,
        uncapped_splits=uncapped_splits,
        splits=splits,
        step_thresholds=step_thresholds,
        step_labels=step_labels,
        outcome_thresholds=outcome_thresholds,
        outcome_labels=outcome_labels,
        training_labels=training_labels,
        selection_eval_labels=selection_eval_labels,
        normalized_features=normalized_features,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        feature_profile=str(feature_profile),
        seed=int(seeds.split_seed),
        seeds=seeds,
        target_variant=str(target_variant),
        gamma=float(resolved_gamma),
        split_family=str(split_family),
        ablation_group=ablation_spec.group,
        ablation_mode=ablation_spec.mode,
        ablation_spec=ablation_spec,
        split_diagnostics=split_diagnostics,
        ablation_filter_audit=ablation_filter_audit,
        label_protocol={
            "split_family": str(split_family),
            "target_variant": str(target_variant),
            "gamma": float(resolved_gamma),
            "split_seed": int(seeds.split_seed),
            "sampling_seed": int(seeds.sampling_seed),
            "train_seed": int(seeds.train_seed),
            "loader_seed": int(seeds.loader_seed),
            "legacy_seed_compatible": bool(seeds.legacy_compatible),
            "ablation_group": ablation_spec.group,
            "ablation_mode": ablation_spec.mode,
            "training_target_type": ablation_spec.training_target_type,
            "selection_target_type": ablation_spec.selection_target_type,
            "primary_eval_target": ablation_spec.primary_eval_target,
            "secondary_eval_target": ablation_spec.secondary_eval_target,
            "train_mask_mode": ablation_spec.train_mask_mode,
            "positive_index": int(ablation_spec.positive_index),
            "num_classes": int(ablation_spec.num_classes),
            "step_q_low": float(step_q_low),
            "step_q_high": float(step_q_high),
            "outcome_q_low": float(outcome_q_low),
            "outcome_q_high": float(outcome_q_high),
            "max_train_samples": None if max_train_samples is None else int(max_train_samples),
            "max_val_samples": None if max_val_samples is None else int(max_val_samples),
            "max_test_samples": None if max_test_samples is None else int(max_test_samples),
        },
    )


def run_precursor_model(
    *,
    prepared: PreparedPrecursorInputs,
    model_name: str,
    output_dir: str | Path,
    device: str,
    window_size: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    cnn_kernel_size: int,
    mlp_depth: int,
    transformer_d_model: int | None = None,
    transformer_num_layers: int = 2,
    transformer_num_heads: int = 4,
    transformer_ffn_multiplier: int = 4,
    transformer_max_length: int = 32,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    critic_baseline_mode: str,
    experiment_id: str | None = None,
    attempt_id: str = "attempt-001",
    work_package: str = "revision",
    source_snapshot_hash: str | None = None,
    allow_existing_output: bool = False,
    minimum_free_space_gib: float = 10.0,
    reproducibility_record: dict[str, Any] | None = None,
    reproducibility_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = guard_revision_output_dir(
        output_dir,
        repo_root=REPO_ROOT,
        allow_existing=bool(allow_existing_output),
    )
    free_bytes_before = require_free_space(output_dir, minimum_gib=float(minimum_free_space_gib))
    torch.manual_seed(int(prepared.seeds.train_seed))
    np.random.seed(int(prepared.seeds.train_seed))
    device_obj = torch.device(device)
    spec = get_precursor_model_spec(model_name)
    dataset = prepared.dataset
    env_name_map = prepared.env_name_map
    splits = prepared.splits
    step_labels = prepared.step_labels
    outcome_labels = prepared.outcome_labels
    training_labels = prepared.training_labels
    selection_eval_labels = prepared.selection_eval_labels
    normalized_features = prepared.normalized_features
    norm_mean = prepared.normalization_mean
    norm_std = prepared.normalization_std

    if reproducibility_record is None:
        resolved_reproducibility_dir = Path(reproducibility_dir) if reproducibility_dir is not None else output_dir / "reproducibility"
        reproducibility_record = export_reproducibility_artifacts(
            output_dir=resolved_reproducibility_dir,
            dataset=dataset,
            uncapped_splits=prepared.uncapped_splits,
            capped_splits=prepared.base_splits,
            step_thresholds=prepared.step_thresholds,
            outcome_thresholds=prepared.outcome_thresholds,
            normalization_mean=norm_mean,
            normalization_std=norm_std,
        )


    train_outputs: dict[str, np.ndarray] | None = None
    model_state_dict: dict[str, Any] | None = None
    training_history: list[dict[str, float | int]] = []
    best_epoch = -1
    best_val_auprc = float("-inf")
    fitted_temperature = 1.0
    trainable_parameter_count = 0
    fit_wall_seconds = 0.0
    peak_cuda_allocated_bytes = 0
    peak_cuda_reserved_bytes = 0
    transformer_match: dict[str, int | float] | None = None
    if int(gradient_accumulation_steps) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if model_name != "transformer" and int(gradient_accumulation_steps) != 1:
        raise ValueError("Gradient accumulation fallback is restricted to Transformer runs")

    if spec.trainable:
        model_datasets = _build_model_datasets(
            dataset,
            splits,
            training_labels,
            normalized_features=normalized_features,
            feature_profile=prepared.feature_profile,
            model_name=model_name,
            window_size=int(window_size),
            shuffle_seed=int(prepared.seeds.split_seed),
        )

        feature_dim = int(normalized_features.shape[1])
        effective_input_dim = feature_dim if spec.input_view == "sequence" else int(window_size) * feature_dim
        resolved_transformer_d_model = transformer_d_model
        if model_name == "transformer":
            transformer_match = select_transformer_d_model(
                feature_dim,
                gru_hidden_dim=int(hidden_dim),
                num_classes=int(prepared.ablation_spec.num_classes),
                window_size=int(window_size),
            )
            resolved_transformer_d_model = (
                int(transformer_d_model)
                if transformer_d_model is not None
                else int(transformer_match["d_model"])
            )
            if int(resolved_transformer_d_model) != int(transformer_match["d_model"]):
                raise ValueError(
                    "Transformer d_model must equal the performance-blind parameter match: "
                    f"requested={resolved_transformer_d_model}, expected={transformer_match['d_model']}"
                )
            if abs(float(transformer_match["relative_deviation_percent"])) > 10.0:
                raise ValueError(f"Transformer parameter deviation exceeds 10%: {transformer_match}")
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
            torch.cuda.reset_peak_memory_stats(device_obj)
        model = build_precursor_model(
            model_name,
            input_dim=effective_input_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            num_classes=int(prepared.ablation_spec.num_classes),
            cnn_kernel_size=int(cnn_kernel_size),
            mlp_depth=int(mlp_depth),
            transformer_d_model=resolved_transformer_d_model,
            transformer_num_layers=int(transformer_num_layers),
            transformer_num_heads=int(transformer_num_heads),
            transformer_ffn_multiplier=int(transformer_ffn_multiplier),
            transformer_max_length=int(transformer_max_length),
        )
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected a torch.nn.Module for trainable model `{model_name}`.")
        model = model.to(device_obj)
        trainable_parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        loader_generator = torch.Generator(device="cpu")
        if prepared.seeds.legacy_compatible:
            # Clone the post-initialization RNG state so the historical one-seed
            # default retains its batch order while DataLoader owns a recorded,
            # explicit generator.
            loader_generator.set_state(torch.random.get_rng_state())
            loader_rng_mode = "legacy_post_model_init_state"
        else:
            loader_generator.manual_seed(int(prepared.seeds.loader_seed))
            loader_rng_mode = "explicit_loader_seed"
        train_loader = _make_loader(
            model_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=loader_generator,
        )
        val_loader = _make_loader(
            model_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=loader_generator,
        )
        test_loader = _make_loader(
            model_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=loader_generator,
        )
        class_weights = _class_weights(
            training_labels[splits["train"]],
            num_classes=int(prepared.ablation_spec.num_classes),
        ).to(device_obj)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_state_dict: dict[str, Any] | None = None
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        fit_started = time.perf_counter()
        for epoch in range(int(epochs)):
            train_loss = _run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device_obj,
                input_view=spec.input_view,
                gradient_accumulation_steps=int(gradient_accumulation_steps),
            )
            val_outputs = _collect_outputs(model=model, loader=val_loader, device=device_obj, input_view=spec.input_view, temperature=1.0)
            scaler = TemperatureScaler()
            scaler.fit(
                logits=torch.as_tensor(val_outputs["logits"], dtype=torch.float32, device=device_obj),
                labels=torch.as_tensor(training_labels[val_outputs["sample_index"]], dtype=torch.long, device=device_obj),
            )
            val_summary = _evaluate_selection_target(
                outputs={
                    **val_outputs,
                    "probabilities": torch.softmax(
                        scaler.apply(torch.as_tensor(val_outputs["logits"], dtype=torch.float32, device=device_obj)),
                        dim=-1,
                    ).cpu().numpy(),
                },
                selection_binary_labels=selection_eval_labels,
                positive_index=int(prepared.ablation_spec.positive_index),
                env_name_map=env_name_map,
            )
            val_auprc = float(val_summary["overall"]["catastrophic_auprc"])
            training_history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_selection_auprc": val_auprc,
                    "val_catastrophic_auprc": val_auprc,
                    "temperature": float(scaler.temperature),
                }
            )
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_epoch = int(epoch)
                best_state_dict = deepcopy(model.state_dict())
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        fit_wall_seconds = float(time.perf_counter() - fit_started)
        if device_obj.type == "cuda":
            peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated(device_obj))
            peak_cuda_reserved_bytes = int(torch.cuda.max_memory_reserved(device_obj))

        if best_state_dict is None:
            raise RuntimeError("Training failed to produce a valid model state.")

        model.load_state_dict(best_state_dict)
        model_state_dict = deepcopy(model.state_dict())
        val_outputs = _collect_outputs(model=model, loader=val_loader, device=device_obj, input_view=spec.input_view, temperature=1.0)
        scaler = TemperatureScaler()
        scaler.fit(
            logits=torch.as_tensor(val_outputs["logits"], dtype=torch.float32, device=device_obj),
            labels=torch.as_tensor(training_labels[val_outputs["sample_index"]], dtype=torch.long, device=device_obj),
        )
        fitted_temperature = float(scaler.temperature)

        val_outputs = _collect_outputs(model=model, loader=val_loader, device=device_obj, input_view=spec.input_view, temperature=fitted_temperature)
        test_outputs = _collect_outputs(model=model, loader=test_loader, device=device_obj, input_view=spec.input_view, temperature=fitted_temperature)
        selection_target_val = _evaluate_selection_target(
            outputs=val_outputs,
            selection_binary_labels=selection_eval_labels,
            positive_index=int(prepared.ablation_spec.positive_index),
            env_name_map=env_name_map,
        )
        selection_target_test = _evaluate_selection_target(
            outputs=test_outputs,
            selection_binary_labels=selection_eval_labels,
            positive_index=int(prepared.ablation_spec.positive_index),
            env_name_map=env_name_map,
        )
        label_step_val = _evaluate_against_labels(
            outputs=val_outputs,
            eval_labels=step_labels,
            positive_index=int(prepared.ablation_spec.positive_index),
            env_name_map=env_name_map,
        )
        label_step_test = _evaluate_against_labels(
            outputs=test_outputs,
            eval_labels=step_labels,
            positive_index=int(prepared.ablation_spec.positive_index),
            env_name_map=env_name_map,
        )
        feature_dim = int(normalized_features.shape[1])
        effective_input_dim = feature_dim if spec.input_view == "sequence" else int(window_size) * feature_dim
    else:
        loader_rng_mode = "not_applicable_reference_baseline"
        baseline = build_precursor_model(
            model_name,
            input_dim=int(normalized_features.shape[1]),
            num_classes=int(prepared.ablation_spec.num_classes),
        )
        if not isinstance(baseline, PrevalenceBaseline):
            raise TypeError(f"Expected PrevalenceBaseline for model `{model_name}`.")
        train_probabilities = baseline.fit(torch.as_tensor(training_labels[splits["train"]], dtype=torch.long)).cpu().numpy()
        train_outputs = _collect_reference_outputs(dataset, splits["train"], probabilities=train_probabilities)
        val_outputs = _collect_reference_outputs(dataset, splits["val"], probabilities=train_probabilities)
        test_outputs = _collect_reference_outputs(dataset, splits["test"], probabilities=train_probabilities)
        selection_target_val = _evaluate_selection_target(
            outputs=val_outputs,
            selection_binary_labels=selection_eval_labels,
            positive_index=int(prepared.ablation_spec.positive_index),
            env_name_map=env_name_map,
        )
        selection_target_test = _evaluate_selection_target(
            outputs=test_outputs,
            selection_binary_labels=selection_eval_labels,
            positive_index=int(prepared.ablation_spec.positive_index),
            env_name_map=env_name_map,
        )
        label_step_val = _evaluate_against_labels(
            outputs=val_outputs,
            eval_labels=step_labels,
            positive_index=int(prepared.ablation_spec.positive_index),
            env_name_map=env_name_map,
        )
        label_step_test = _evaluate_against_labels(
            outputs=test_outputs,
            eval_labels=step_labels,
            positive_index=int(prepared.ablation_spec.positive_index),
            env_name_map=env_name_map,
        )
        best_val_auprc = float(selection_target_val["overall"]["catastrophic_auprc"])
        fitted_temperature = 1.0
        model_state_dict = None
        effective_input_dim = int(normalized_features.shape[1])

    label_outcome_val = _evaluate_against_labels(
        outputs=val_outputs,
        eval_labels=outcome_labels,
        positive_index=int(prepared.ablation_spec.positive_index),
        env_name_map=env_name_map,
    )
    label_outcome_test = _evaluate_against_labels(
        outputs=test_outputs,
        eval_labels=outcome_labels,
        positive_index=int(prepared.ablation_spec.positive_index),
        env_name_map=env_name_map,
    )
    critic_step_val = _summarize_critic_scores(
        scores=val_outputs["critic_score"],
        labels_3way=step_labels[val_outputs["sample_index"]],
        env_indices=val_outputs["env_index"],
        env_name_map=env_name_map,
    )
    critic_step_test = _summarize_critic_scores(
        scores=test_outputs["critic_score"],
        labels_3way=step_labels[test_outputs["sample_index"]],
        env_indices=test_outputs["env_index"],
        env_name_map=env_name_map,
    )
    critic_outcome_val = _summarize_critic_scores(
        scores=val_outputs["critic_score"],
        labels_3way=outcome_labels[val_outputs["sample_index"]],
        env_indices=val_outputs["env_index"],
        env_name_map=env_name_map,
    )
    critic_outcome_test = _summarize_critic_scores(
        scores=test_outputs["critic_score"],
        labels_3way=outcome_labels[test_outputs["sample_index"]],
        env_indices=test_outputs["env_index"],
        env_name_map=env_name_map,
    )

    if critic_baseline_mode == "disabled":
        critic_baseline_metrics: dict[str, Any] = {
            "disabled": True,
            "reason": "critic baseline disabled for public/offline feasibility runs",
        }
    else:
        critic_baseline_metrics = {
            "label_step": {"val": critic_step_val, "test": critic_step_test},
            "label_outcome": {"val": critic_outcome_val, "test": critic_outcome_test},
        }

    metrics = {
        "freeze_id": "OTTP-R26-G1-v1",
        "target_variant": str(prepared.target_variant),
        "gamma": float(prepared.gamma),
        "split_seed": int(prepared.seeds.split_seed),
        "sampling_seed": int(prepared.seeds.sampling_seed),
        "train_seed": int(prepared.seeds.train_seed),
        "loader_seed": int(prepared.seeds.loader_seed),
        "loader_rng_mode": str(loader_rng_mode),
        "model_name": spec.model_name,
        "model_family": spec.model_family,
        "input_view": spec.input_view,
        "feature_profile": str(prepared.feature_profile),
        "split_family": str(prepared.split_family),
        "ablation_group": prepared.ablation_group,
        "ablation_mode": prepared.ablation_mode,
        "selection_mode": dataset["metadata"].get("selection_mode", "unknown"),
        "split_sizes": {
            name: int(len(indices))
            for name, indices in [("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]
        },
        "base_split_sizes": {
            name: int(len(indices))
            for name, indices in [("train", prepared.base_splits["train"]), ("val", prepared.base_splits["val"]), ("test", prepared.base_splits["test"])]
        },
        "effective_split_sizes": {
            name: int(len(indices))
            for name, indices in [("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]
        },
        "split_manifest": dict(splits["manifest"]),
        "split_diagnostics": dict(prepared.split_diagnostics),
        "split_filter_audit": dict(prepared.ablation_filter_audit),
        "label_protocol": dict(prepared.label_protocol),
        "selection_target_type": str(prepared.ablation_spec.selection_target_type),
        "primary_eval_target": str(prepared.ablation_spec.primary_eval_target),
        "secondary_eval_target": str(prepared.ablation_spec.secondary_eval_target),
        "train_mask_mode": str(prepared.ablation_spec.train_mask_mode),
        "positive_index": int(prepared.ablation_spec.positive_index),
        "num_classes": int(prepared.ablation_spec.num_classes),
        "selection_target": {"val": selection_target_val, "test": selection_target_test},
        "label_step": {"val": label_step_val, "test": label_step_test},
        "label_outcome": {"val": label_outcome_val, "test": label_outcome_test},
        "critic_baseline": critic_baseline_metrics,
        "training": {
            "trainable": bool(spec.trainable),
            "trainable_parameter_count": int(trainable_parameter_count),
            "best_epoch": int(best_epoch),
            "best_val_selection_auprc": float(best_val_auprc),
            "best_val_catastrophic_auprc": float(best_val_auprc),
            "temperature": float(fitted_temperature),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
            "fit_wall_seconds": float(fit_wall_seconds),
            "peak_cuda_allocated_bytes": int(peak_cuda_allocated_bytes),
            "peak_cuda_reserved_bytes": int(peak_cuda_reserved_bytes),
            "device": str(device_obj),
            "device_name": (
                str(torch.cuda.get_device_name(device_obj)) if device_obj.type == "cuda" else platform.processor()
            ),
            "transformer_parameter_match": transformer_match,
            "history": training_history,
        },
    }

    bundle = {
        "model_state_dict": model_state_dict,
        "model_config": {
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "input_view": spec.input_view,
            "feature_profile": str(prepared.feature_profile),
            "reference_baseline": not bool(spec.trainable),
            "input_dim": int(effective_input_dim),
            "window_size": int(window_size),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "cnn_kernel_size": int(cnn_kernel_size),
            "mlp_depth": int(mlp_depth),
            "transformer_d_model": (
                None if model_name != "transformer" else int(transformer_match["d_model"])
            ),
            "transformer_num_layers": int(transformer_num_layers),
            "transformer_num_heads": int(transformer_num_heads),
            "transformer_ffn_multiplier": int(transformer_ffn_multiplier),
            "transformer_max_length": int(transformer_max_length),
            "num_classes": int(prepared.ablation_spec.num_classes),
        },
        "temperature": float(fitted_temperature),
        "window_size": int(window_size),
        "normalization_mean": norm_mean,
        "normalization_std": norm_std,
        "freeze_id": "OTTP-R26-G1-v1",
        "target_variant": str(prepared.target_variant),
        "gamma": float(prepared.gamma),
        "seeds": prepared.seeds.to_dict(),
        "split_family": str(prepared.split_family),
        "ablation_group": prepared.ablation_group,
        "ablation_mode": prepared.ablation_mode,
        "step_thresholds": prepared.step_thresholds,
        "outcome_thresholds": prepared.outcome_thresholds,
        "split_manifest": splits["manifest"],
        "split_diagnostics": prepared.split_diagnostics,
        "split_filter_audit": prepared.ablation_filter_audit,
        "selection_mode": dataset["metadata"].get("selection_mode", "unknown"),
        "env_names": env_name_map,
        "policy_names": {int(key): str(value) for key, value in dataset["metadata"]["policy_names"].items()},
        "max_obs_dim": int(dataset["metadata"]["max_obs_dim"]),
        "max_act_dim": int(dataset["metadata"]["max_act_dim"]),
        "metrics": metrics,
    }

    bundle_path = output_dir / "precursor_model.pt"
    metrics_path = output_dir / "metrics.json"
    torch.save(bundle, bundle_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    protocol = protocol_identity(
        target_variant=prepared.target_variant,
        gamma=prepared.gamma,
        quantile_pair=(
            float(prepared.label_protocol["step_q_low"]),
            float(prepared.label_protocol["step_q_high"]),
        ),
        outcome_quantile_pair=(
            float(prepared.label_protocol["outcome_q_low"]),
            float(prepared.label_protocol["outcome_q_high"]),
        ),
        split_family=prepared.split_family,
        seeds=prepared.seeds,
        feature_profile=prepared.feature_profile,
        model_name=spec.model_name,
        extra={
            "window_size": int(window_size),
            "max_train_samples": prepared.label_protocol["max_train_samples"],
            "max_val_samples": prepared.label_protocol["max_val_samples"],
            "max_test_samples": prepared.label_protocol["max_test_samples"],
            "ablation_group": prepared.ablation_group,
            "ablation_mode": prepared.ablation_mode,
            "transformer_d_model": None if transformer_match is None else int(transformer_match["d_model"]),
            "transformer_num_layers": int(transformer_num_layers),
            "transformer_num_heads": int(transformer_num_heads),
            "transformer_ffn_multiplier": int(transformer_ffn_multiplier),
            "transformer_max_length": int(transformer_max_length),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
        },
    )
    task_name = str(dataset.get("metadata", {}).get("task_name", prepared.dataset_path.stem))
    default_experiment_id = (
        f"r26_{str(work_package).replace('_', '-')}_unified38_"
        f"{task_name.lower().replace('_', '-')}_{spec.model_name.replace('_', '-')}_"
        f"{prepared.target_variant.replace('_', '-')}_ss{prepared.seeds.split_seed}_"
        f"ts{prepared.seeds.train_seed}_{protocol['protocol_hash'][:12]}"
    )
    resolved_source_snapshot_hash = source_snapshot_hash or source_snapshot_sha256(REPO_ROOT, REVISION_SOURCE_FILES)
    run_manifest = {
        "schema_version": "ottp_revision_run_manifest_v1",
        "freeze_id": "OTTP-R26-G1-v1",
        "experiment_id": str(experiment_id or default_experiment_id),
        "attempt_id": str(attempt_id),
        "work_package": str(work_package),
        "source_snapshot_hash": str(resolved_source_snapshot_hash),
        "dataset": {
            "task": task_name,
            "path": str(prepared.dataset_path),
            "sha256": str(prepared.dataset_sha256),
        },
        "protocol": protocol,
        "indices": reproducibility_record,
        "training": {
            "device": str(device),
            "window_size": int(window_size),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
            "loader_rng_mode": str(loader_rng_mode),
            "trainable_parameter_count": int(trainable_parameter_count),
            "fit_wall_seconds": float(fit_wall_seconds),
            "peak_cuda_allocated_bytes": int(peak_cuda_allocated_bytes),
            "peak_cuda_reserved_bytes": int(peak_cuda_reserved_bytes),
            "device_name": (
                str(torch.cuda.get_device_name(device_obj)) if device_obj.type == "cuda" else platform.processor()
            ),
            "transformer_parameter_match": transformer_match,
        },
        "resources": {
            "minimum_free_space_gib": float(minimum_free_space_gib),
            "free_bytes_before": int(free_bytes_before),
        },
        "artifacts": {
            "metrics.json": {"sha256": file_sha256(metrics_path), "bytes": int(metrics_path.stat().st_size)},
            "precursor_model.pt": {"sha256": file_sha256(bundle_path), "bytes": int(bundle_path.stat().st_size)},
        },
    }
    assert_valid_run_manifest(run_manifest)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "metrics": str(metrics_path),
                "run_manifest": str(manifest_path),
                "best_val_catastrophic_auprc": _major_step_metric(metrics, "val"),
                "model_name": spec.model_name,
            },
            indent=2,
        )
    )
    return metrics


def main() -> None:
    args = parse_args()
    seeds = resolve_seed_bundle(
        seed=int(args.seed),
        split_seed=args.split_seed,
        sampling_seed=args.sampling_seed,
        train_seed=args.train_seed,
        loader_seed=args.loader_seed,
    )
    prepared = prepare_precursor_inputs(
        args.dataset,
        seed=int(args.seed),
        split_seed=args.split_seed,
        sampling_seed=args.sampling_seed,
        train_seed=args.train_seed,
        loader_seed=args.loader_seed,
        split_family=str(args.split_family),
        ablation_group=args.ablation_group,
        ablation_mode=args.ablation_mode,
        feature_profile=str(args.feature_profile),
        step_q_low=float(args.step_q_low),
        step_q_high=float(args.step_q_high),
        outcome_q_low=float(args.outcome_q_low),
        outcome_q_high=float(args.outcome_q_high),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        target_variant=str(args.target_variant),
        gamma=args.gamma,
        label_sidecar_path=args.label_sidecar,
    )
    output_dir = Path(args.output_dir) if seeds.legacy_compatible else seeded_output_path(args.output_dir, seeds)
    run_precursor_model(
        prepared=prepared,
        model_name=str(args.model),
        output_dir=output_dir,
        device=str(args.device),
        window_size=int(args.window_size),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        cnn_kernel_size=int(args.cnn_kernel_size),
        mlp_depth=int(args.mlp_depth),
        transformer_d_model=args.transformer_d_model,
        transformer_num_layers=int(args.transformer_num_layers),
        transformer_num_heads=int(args.transformer_num_heads),
        transformer_ffn_multiplier=int(args.transformer_ffn_multiplier),
        transformer_max_length=int(args.transformer_max_length),
        batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        critic_baseline_mode=str(args.critic_baseline_mode),
        experiment_id=args.experiment_id,
        attempt_id=str(args.attempt_id),
        work_package=str(args.work_package),
        source_snapshot_hash=args.source_snapshot_hash,
        allow_existing_output=bool(args.allow_existing_output),
        minimum_free_space_gib=float(args.minimum_free_space_gib),
    )


if __name__ == "__main__":
    main()
