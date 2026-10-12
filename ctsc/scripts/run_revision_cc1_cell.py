'Run one cell with shared preprocessing artifacts.'

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from ctsc.precursor.cc1 import AMENDMENT_ID, write_json
from ctsc.precursor.revision import (
    array_sha256,
    canonical_json_sha256,
    export_reproducibility_artifacts,
    file_sha256,
    require_free_space,
    source_snapshot_sha256,
)
from ctsc.scripts.train_precursor_model import prepare_precursor_inputs, run_precursor_model


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ALIASES = {"gru": "gru", "flat_mlp": "flat-mlp", "transformer": "transformer"}
SOURCE_FILES = (
    "ctsc/precursor/cc1.py",
    "ctsc/precursor/data.py",
    "ctsc/precursor/model.py",
    "ctsc/precursor/metrics.py",
    "ctsc/precursor/revision.py",
    "ctsc/scripts/run_revision_cc1_cell.py",
    "ctsc/scripts/train_precursor_model.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cell-dir", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--feature-profile", choices=["full", "current_only"], required=True)
    parser.add_argument("--q-low", type=float, required=True)
    parser.add_argument("--q-high", type=float, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--model-lr", action="append", required=True, help="MODEL=LR")
    parser.add_argument("--matrix", choices=["stage4_seed_bridge", "stage5_fair_matrix"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-samples", type=int, default=100000)
    parser.add_argument("--max-val-samples", type=int, default=50000)
    parser.add_argument("--minimum-free-space-gib", type=float, default=10.0)
    return parser.parse_args()


def _canonical_model(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _parse_model_lrs(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected MODEL=LR, got {item!r}")
        model_text, lr_text = item.split("=", 1)
        model = _canonical_model(model_text)
        if model not in MODEL_ALIASES or model in parsed:
            raise ValueError(f"Unsupported or duplicate model: {model!r}")
        learning_rate = float(lr_text)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"Invalid learning rate for {model}: {learning_rate}")
        parsed[model] = learning_rate
    return parsed


def _relative_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(record))
    payload["root"] = "reproducibility"
    return payload


def _expected_thresholds(prepared: Any) -> dict[str, dict[str, dict[str, float]]]:
    def convert(values: Mapping[int, Mapping[str, float]]) -> dict[str, dict[str, float]]:
        return {
            str(int(env_id)): {str(key): float(value) for key, value in row.items()}
            for env_id, row in values.items()
        }

    return {"step": convert(prepared.step_thresholds), "outcome": convert(prepared.outcome_thresholds)}


def _validate_shared(cell_dir: Path, record: Mapping[str, Any], prepared: Any) -> None:
    repro = cell_dir / "reproducibility"
    for section, names, source in (
        ("uncapped", ("train", "val", "test"), prepared.uncapped_splits),
        ("capped", ("train", "val"), prepared.base_splits),
    ):
        for name in names:
            expected = array_sha256(np.asarray(source[name], dtype=np.int64))
            if str(record[section][name]["sha256"]) != expected:
                raise ValueError(f"Shared {section}/{name} hash differs from freshly prepared inputs")
    for name, metadata in record["files"].items():
        path = repro / name
        if not path.is_file() or file_sha256(path) != str(metadata["sha256"]):
            raise ValueError(f"Shared reproducibility artifact is missing or changed: {path}")
    with np.load(repro / "normalization_stats.npz", allow_pickle=False) as archive:
        if not np.array_equal(np.asarray(archive["mean"]), prepared.normalization_mean):
            raise ValueError("Shared normalization mean differs from freshly prepared inputs")
        if not np.array_equal(np.asarray(archive["std"]), prepared.normalization_std):
            raise ValueError("Shared normalization std differs from freshly prepared inputs")
    thresholds = json.loads((repro / "thresholds.json").read_text(encoding="utf-8"))
    if thresholds != _expected_thresholds(prepared):
        raise ValueError("Shared thresholds differ from freshly prepared inputs")


def _shared_record(cell_dir: Path, prepared: Any) -> dict[str, Any]:
    record_path = cell_dir / "reproducibility_record.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        _validate_shared(cell_dir, record, prepared)
        return record
    repro = cell_dir / "reproducibility"
    if repro.exists() and any(repro.iterdir()):
        raise FileExistsError(f"Unrecorded shared reproducibility artifacts exist: {repro}")
    record = export_reproducibility_artifacts(
        output_dir=repro,
        dataset=prepared.dataset,
        uncapped_splits=prepared.uncapped_splits,
        capped_splits=prepared.base_splits,
        step_thresholds=prepared.step_thresholds,
        outcome_thresholds=prepared.outcome_thresholds,
        normalization_mean=prepared.normalization_mean,
        normalization_std=prepared.normalization_std,
    )
    relative = _relative_record(record)
    write_json(record_path, relative)
    _validate_shared(cell_dir, relative, prepared)
    return relative


def _candidate_metrics(cell_dir: Path, model: str) -> list[Path]:
    return sorted((cell_dir / "models" / model).glob("attempt-*/metrics.json"))


def _valid_success(path: Path, *, model: str, lr: float, args: argparse.Namespace, shared: Mapping[str, Any]) -> bool:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(path.with_name("run_manifest.json").read_text(encoding="utf-8"))
        protocol = manifest["protocol"]
        training = manifest["training"]
        seeds = ("split_seed", "sampling_seed", "train_seed", "loader_seed")
        return bool(
            _canonical_model(metrics["model_name"]) == model
            and all(int(protocol[name]) == int(args.split_seed) for name in seeds)
            and protocol["legacy_compatible"] is False
            and str(training["loader_rng_mode"]) == "explicit_loader_seed"
            and math.isclose(float(training["learning_rate"]), lr, rel_tol=0.0, abs_tol=1e-15)
            and str(metrics["feature_profile"]) == str(args.feature_profile)
            and math.isclose(float(metrics["label_protocol"]["step_q_low"]), args.q_low, abs_tol=1e-15)
            and math.isclose(float(metrics["label_protocol"]["step_q_high"]), args.q_high, abs_tol=1e-15)
            and len(metrics["training"]["history"]) == int(args.epochs)
            and manifest["indices"] == shared
            and file_sha256(path) == str(manifest["artifacts"]["metrics.json"]["sha256"])
            and file_sha256(path.with_name("precursor_model.pt"))
            == str(manifest["artifacts"]["precursor_model.pt"]["sha256"])
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _existing_success(cell_dir: Path, *, model: str, lr: float, args: argparse.Namespace, shared: Mapping[str, Any]) -> Path | None:
    valid = [
        path
        for path in _candidate_metrics(cell_dir, model)
        if _valid_success(path, model=model, lr=lr, args=args, shared=shared)
    ]
    return valid[-1] if valid else None


def _next_attempt(cell_dir: Path, model: str) -> tuple[int, Path]:
    base = cell_dir / "models" / model
    for number in range(1, 1000):
        candidate = base / f"attempt-{number:03d}"
        if not candidate.exists():
            return number, candidate
    raise RuntimeError(f"Exhausted attempts for {model}")


def _is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "cuda" in text and "out of memory" in text


def _run_model(
    *,
    model: str,
    lr: float,
    args: argparse.Namespace,
    cell_dir: Path,
    prepared: Any,
    shared: Mapping[str, Any],
    source_hash: str,
    fallback: bool,
) -> Path:
    attempt_number, attempt_dir = _next_attempt(cell_dir, model)
    batch_size = 128 if fallback else int(args.batch_size)
    accumulation = 2 if fallback else 1
    run_precursor_model(
        prepared=prepared,
        model_name=MODEL_ALIASES[model],
        output_dir=attempt_dir,
        device=str(args.device),
        window_size=int(args.window_size),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        cnn_kernel_size=int(args.cnn_kernel_size),
        mlp_depth=int(args.mlp_depth),
        transformer_num_layers=2,
        transformer_num_heads=4,
        transformer_ffn_multiplier=4,
        transformer_max_length=32,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        epochs=int(args.epochs),
        lr=float(lr),
        weight_decay=float(args.weight_decay),
        num_workers=0,
        critic_baseline_mode="dataset",
        experiment_id=f"cc1__{args.matrix}__{args.condition_id}__{args.task_name}__ss{args.split_seed}__{model}",
        attempt_id=f"attempt-{attempt_number:03d}",
        work_package=args.matrix,
        source_snapshot_hash=source_hash,
        minimum_free_space_gib=float(args.minimum_free_space_gib),
        reproducibility_record=dict(shared),
    )
    metrics = attempt_dir / "metrics.json"
    if not _valid_success(metrics, model=model, lr=lr, args=args, shared=shared):
        raise RuntimeError(f"Completed run failed CC1 validation: {metrics}")
    return metrics


def main() -> None:
    args = parse_args()
    model_lrs = _parse_model_lrs(args.model_lr)
    cell_dir = Path(args.cell_dir).resolve()
    cell_dir.mkdir(parents=True, exist_ok=True)
    require_free_space(cell_dir, minimum_gib=float(args.minimum_free_space_gib))
    seed = int(args.split_seed)
    prepared = prepare_precursor_inputs(
        Path(args.dataset).resolve(),
        seed=seed,
        split_seed=seed,
        sampling_seed=seed,
        train_seed=seed,
        loader_seed=seed,
        split_family="episode_group",
        ablation_group=None,
        ablation_mode=None,
        feature_profile=str(args.feature_profile),
        step_q_low=float(args.q_low),
        step_q_high=float(args.q_high),
        outcome_q_low=float(args.q_low),
        outcome_q_high=float(args.q_high),
        max_train_samples=int(args.max_train_samples),
        max_val_samples=int(args.max_val_samples),
        max_test_samples=None,
        target_variant="current_inclusive",
        gamma=0.99,
    )
    shared = _shared_record(cell_dir, prepared)
    descriptor = {
        "schema_version": "ottp_cc1_cell_v1",
        "amendment_id": AMENDMENT_ID,
        "matrix": str(args.matrix),
        "task_name": str(args.task_name),
        "condition_id": str(args.condition_id),
        "feature_profile": str(args.feature_profile),
        "quantile_pair": [float(args.q_low), float(args.q_high)],
        "split_seed": seed,
        "sampling_seed": seed,
        "train_seed": seed,
        "loader_seed": seed,
        "seed_protocol": "explicit_four_stream",
        "model_learning_rates": model_lrs,
        "shared_reproducibility_record_sha256": canonical_json_sha256(shared),
    }
    descriptor["cell_sha256"] = canonical_json_sha256(descriptor)
    descriptor_path = cell_dir / "cell_manifest.json"
    if descriptor_path.is_file() and json.loads(descriptor_path.read_text(encoding="utf-8")) != descriptor:
        raise ValueError(f"Cell descriptor changed on resume: {descriptor_path}")
    write_json(descriptor_path, descriptor)
    source_hash = source_snapshot_sha256(REPO_ROOT, SOURCE_FILES)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for model, lr in model_lrs.items():
        existing = _existing_success(cell_dir, model=model, lr=lr, args=args, shared=shared)
        if existing is not None:
            results.append({"model_name": model, "status": "SKIPPED_VALID", "metrics_path": str(existing)})
            continue
        try:
            metrics = _run_model(
                model=model,
                lr=lr,
                args=args,
                cell_dir=cell_dir,
                prepared=prepared,
                shared=shared,
                source_hash=source_hash,
                fallback=False,
            )
            results.append({"model_name": model, "status": "PASS", "metrics_path": str(metrics)})
        except BaseException as error:
            if model == "transformer" and _is_cuda_oom(error):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    metrics = _run_model(
                        model=model,
                        lr=lr,
                        args=args,
                        cell_dir=cell_dir,
                        prepared=prepared,
                        shared=shared,
                        source_hash=source_hash,
                        fallback=True,
                    )
                    results.append({"model_name": model, "status": "PASS_OOM_FALLBACK", "metrics_path": str(metrics)})
                    continue
                except BaseException as fallback_error:
                    error = fallback_error
            failures.append({"model_name": model, "error_type": type(error).__name__, "error": str(error)})
    summary = {
        "schema_version": "ottp_cc1_cell_summary_v1",
        "cell_sha256": descriptor["cell_sha256"],
        "status": "PASS" if not failures and len(results) == len(model_lrs) else "FAIL",
        "results": results,
        "failures": failures,
    }
    write_json(cell_dir / "cell_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
