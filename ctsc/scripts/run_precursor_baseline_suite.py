'Run the Unified38 baseline zoo on one dataset / slice / seed.'

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = ["gru", "temporal-cnn", "flat-mlp", "linear", "prevalence"]
SUPPORTED_MODELS = [*DEFAULT_MODELS, "transformer"]
SEQUENCE_BASELINES = ["gru", "temporal-cnn", "transformer"]
NONSEQUENTIAL_BASELINES = ["flat-mlp", "linear"]

from ctsc.precursor.ablation import ABLATION_GROUP_REGISTRY
from ctsc.precursor.data import FEATURE_PROFILE_REGISTRY, SPLIT_FAMILY_REGISTRY
from ctsc.precursor.revision import (
    export_reproducibility_artifacts,
    file_sha256,
    guard_revision_output_dir,
    require_free_space,
    resolve_seed_bundle,
    seeded_output_path,
)


def _jsonable_thresholds(payload: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        str(int(env_id)): {str(name): float(value) for name, value in values.items()}
        for env_id, values in payload.items()
    }


def _split_class_audit(labels: Any, splits: dict[str, Any], *, num_classes: int) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(labels, dtype=np.int64)
    result: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        indices = np.asarray(splits[split_name], dtype=np.int64)
        counts = np.bincount(values[indices], minlength=int(num_classes)).astype(np.int64)
        total = int(counts.sum())
        result[split_name] = {
            "sample_count": total,
            "class_counts": counts.tolist(),
            "class_ratios": (counts.astype(np.float64) / max(total, 1)).tolist(),
        }
    return result
from ctsc.scripts.train_precursor_model import prepare_precursor_inputs, run_precursor_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--loader-seed", type=int, default=None)
    parser.add_argument("--split-family", type=str, choices=list(SPLIT_FAMILY_REGISTRY), default="episode_group")
    parser.add_argument("--ablation-group", type=str, choices=list(ABLATION_GROUP_REGISTRY), default=None)
    parser.add_argument("--ablation-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
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
    parser.add_argument("--step-q-low", type=float, required=True)
    parser.add_argument("--step-q-high", type=float, required=True)
    parser.add_argument("--outcome-q-low", type=float, required=True)
    parser.add_argument("--outcome-q-high", type=float, required=True)
    parser.add_argument("--critic-baseline-mode", type=str, choices=["dataset", "disabled"], default="dataset")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--target-variant", choices=["current_inclusive", "future_only"], default="current_inclusive")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--label-sidecar", type=str, default=None)
    parser.add_argument("--attempt-id", type=str, default="attempt-001")
    parser.add_argument("--work-package", type=str, default="revision")
    parser.add_argument("--source-snapshot-hash", type=str, default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--minimum-free-space-gib", type=float, default=10.0)
    return parser.parse_args()

def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": str(metrics["model_name"]),
        "model_family": str(metrics["model_family"]),
        "input_view": str(metrics["input_view"]),
        "feature_profile": str(metrics["feature_profile"]),
        "split_family": str(metrics["split_family"]),
        "target_variant": str(metrics["target_variant"]),
        "gamma": float(metrics["gamma"]),
        "split_seed": int(metrics["split_seed"]),
        "sampling_seed": int(metrics["sampling_seed"]),
        "train_seed": int(metrics["train_seed"]),
        "loader_seed": int(metrics["loader_seed"]),
        "ablation_group": metrics.get("ablation_group"),
        "ablation_mode": metrics.get("ablation_mode"),
        "selection_target_type": str(metrics["selection_target_type"]),
        "primary_eval_target": str(metrics["primary_eval_target"]),
        "secondary_eval_target": str(metrics["secondary_eval_target"]),
        "train_mask_mode": str(metrics["train_mask_mode"]),
        "val_selection_target_auprc": float(metrics["selection_target"]["val"]["overall"]["catastrophic_auprc"]),
        "test_selection_target_auprc": float(metrics["selection_target"]["test"]["overall"]["catastrophic_auprc"]),
        "val_step_catastrophic_auprc": float(metrics["label_step"]["val"]["overall"]["catastrophic_auprc"]),
        "test_step_catastrophic_auprc": float(metrics["label_step"]["test"]["overall"]["catastrophic_auprc"]),
        "val_outcome_catastrophic_auprc": float(metrics["label_outcome"]["val"]["overall"]["catastrophic_auprc"]),
        "test_outcome_catastrophic_auprc": float(metrics["label_outcome"]["test"]["overall"]["catastrophic_auprc"]),
        "test_catastrophic_ece": float(metrics["label_step"]["test"]["overall"]["catastrophic_ece"]),
        "test_lead_time_median_at_0.5": float(metrics["label_step"]["test"]["overall"]["lead_time_median_at_0.5"]),
        "training": dict(metrics["training"]),
    }


def _step_test_auprc(summary: dict[str, Any], model_name: str) -> float:
    return float(summary["models"][model_name]["test_step_catastrophic_auprc"])


def main() -> None:
    args = parse_args()
    seeds = resolve_seed_bundle(
        seed=int(args.seed),
        split_seed=args.split_seed,
        sampling_seed=args.sampling_seed,
        train_seed=args.train_seed,
        loader_seed=args.loader_seed,
    )
    requested_output = REPO_ROOT / args.output_dir
    output_dir = requested_output if seeds.legacy_compatible else seeded_output_path(requested_output, seeds)
    output_dir = guard_revision_output_dir(
        output_dir,
        repo_root=REPO_ROOT,
        allow_existing=bool(args.allow_existing_output),
    )
    require_free_space(output_dir, minimum_gib=float(args.minimum_free_space_gib))

    models = list(dict.fromkeys(str(model) for model in args.models))
    unsupported = [model for model in models if model not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(f"Unsupported baseline(s): {unsupported}")

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

    reproducibility_record = export_reproducibility_artifacts(
        output_dir=output_dir / "reproducibility",
        dataset=prepared.dataset,
        uncapped_splits=prepared.uncapped_splits,
        capped_splits=prepared.base_splits,
        step_thresholds=prepared.step_thresholds,
        outcome_thresholds=prepared.outcome_thresholds,
        normalization_mean=prepared.normalization_mean,
        normalization_std=prepared.normalization_std,
    )

    model_summaries: dict[str, Any] = {}
    for model_name in models:
        model_dir = output_dir / model_name.replace("-", "_")
        metrics = run_precursor_model(
            prepared=prepared,
            model_name=model_name,
            output_dir=model_dir,
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
            attempt_id=str(args.attempt_id),
            work_package=str(args.work_package),
            source_snapshot_hash=args.source_snapshot_hash,
            allow_existing_output=bool(args.allow_existing_output),
            minimum_free_space_gib=float(args.minimum_free_space_gib),
            reproducibility_record=reproducibility_record,
            reproducibility_dir=output_dir / "reproducibility",
        )
        metrics_path = model_dir / "metrics.json"
        bundle_path = model_dir / "precursor_model.pt"
        model_summaries[model_name] = {
            "bundle_path": str(bundle_path),
            "metrics_path": str(metrics_path),
            **_metrics_summary(metrics),
        }

    available_sequence = [name for name in SEQUENCE_BASELINES if name in model_summaries]
    available_nonsequential = [name for name in NONSEQUENTIAL_BASELINES if name in model_summaries]
    if not available_sequence:
        raise ValueError("At least one sequential baseline is required to build the suite summary.")

    best_sequence_baseline = max(
        available_sequence,
        key=lambda name: float(model_summaries[name]["val_step_catastrophic_auprc"]),
    )
    best_sequence_test_step_auprc = float(model_summaries[best_sequence_baseline]["test_step_catastrophic_auprc"])
    gap_vs_flat = (
        best_sequence_test_step_auprc - _step_test_auprc({"models": model_summaries}, "flat-mlp")
        if "flat-mlp" in model_summaries
        else None
    )
    gap_vs_linear = (
        best_sequence_test_step_auprc - _step_test_auprc({"models": model_summaries}, "linear")
        if "linear" in model_summaries
        else None
    )
    strict_task_level_sequence_support = bool(available_nonsequential) and all(
        _step_test_auprc({"models": model_summaries}, seq_name) > _step_test_auprc({"models": model_summaries}, nonseq_name)
        for seq_name in available_sequence
        for nonseq_name in available_nonsequential
    )

    suite_summary = {
        "slice_name": str(args.slice_name),
        "seed": int(args.seed),
        "split_seed": int(seeds.split_seed),
        "sampling_seed": int(seeds.sampling_seed),
        "train_seed": int(seeds.train_seed),
        "loader_seed": int(seeds.loader_seed),
        "target_variant": str(prepared.target_variant),
        "gamma": float(prepared.gamma),
        "split_family": str(args.split_family),
        "ablation_group": prepared.ablation_group,
        "ablation_mode": prepared.ablation_mode,
        "selection_target_type": str(prepared.ablation_spec.selection_target_type),
        "primary_eval_target": str(prepared.ablation_spec.primary_eval_target),
        "secondary_eval_target": str(prepared.ablation_spec.secondary_eval_target),
        "train_mask_mode": str(prepared.ablation_spec.train_mask_mode),
        "feature_profile": str(args.feature_profile),
        "label_audit": {
            "threshold_fit_scope": "capped_train_only",
            "normalization_fit_scope": "capped_train_only",
            "step_thresholds": _jsonable_thresholds(prepared.step_thresholds),
            "outcome_thresholds": _jsonable_thresholds(prepared.outcome_thresholds),
            "step_class_distribution": _split_class_audit(
                prepared.step_labels,
                prepared.splits,
                num_classes=int(prepared.ablation_spec.num_classes),
            ),
            "outcome_class_distribution": _split_class_audit(
                prepared.outcome_labels,
                prepared.splits,
                num_classes=int(prepared.ablation_spec.num_classes),
            ),
        },
        "label_sidecar": None,
        "split_manifest": dict(prepared.splits["manifest"]),
        "split_diagnostics": dict(prepared.split_diagnostics),
        "split_filter_audit": dict(prepared.ablation_filter_audit),
        "effective_split_sizes": {
            name: int(len(prepared.splits[name]))
            for name in ("train", "val", "test")
        },
        "models": model_summaries,
        "best_sequence_baseline": best_sequence_baseline,
        "best_sequence_test_step_auprc": best_sequence_test_step_auprc,
        "gap_vs_flat": None if gap_vs_flat is None else float(gap_vs_flat),
        "gap_vs_linear": None if gap_vs_linear is None else float(gap_vs_linear),
        "strict_task_level_sequence_support": bool(strict_task_level_sequence_support),
    }
    if args.label_sidecar is not None:
        sidecar_path = Path(args.label_sidecar).resolve()
        sidecar_manifest_path = sidecar_path.with_suffix(sidecar_path.suffix + ".manifest.json")
        suite_summary["label_sidecar"] = {
            "path": str(sidecar_path),
            "sha256": file_sha256(sidecar_path),
            "manifest_path": str(sidecar_manifest_path),
            "manifest_sha256": file_sha256(sidecar_manifest_path),
        }
    summary_path = output_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(suite_summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "best_sequence_baseline": best_sequence_baseline,
                "gap_vs_flat": None if gap_vs_flat is None else float(gap_vs_flat),
                "gap_vs_linear": None if gap_vs_linear is None else float(gap_vs_linear),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
