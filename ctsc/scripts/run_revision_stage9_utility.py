"""Replay canonical models with validation-selected offline alarm budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from ctsc.precursor.metrics import binary_average_precision
from ctsc.precursor.model import build_precursor_model, get_precursor_model_spec
from ctsc.precursor.revision import array_sha256, canonical_json_sha256, file_sha256, require_free_space
from ctsc.precursor.stage9 import STAGE9_BUDGETS, STAGE9_MODELS, evaluate_budget, full_prefix_windows
from ctsc.scripts.train_precursor_model import (
    _build_model_datasets,
    _collect_outputs,
    _extract_positive_scores,
    _make_loader,
    prepare_precursor_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "runs/main"
STAGE9_ROOT = ROOT / "runs/offline_utility"
# Compact JSON plus its same-volume mirror remain well below this reserve.
CELL_OUTPUT_RESERVE_GIB = 0.01
REGISTRY = ROOT / "manifests/experiment_registry.json"
PRECURSORS = ROOT / "work/precursors"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.resolve().as_posix()


def _latest_attempt(cell: Path, model_name: str) -> Path:
    attempts = sorted((cell / "models" / model_name).glob("attempt-*/run_manifest.json"))
    if not attempts:
        raise FileNotFoundError(f"Missing canonical {model_name} run: {cell}")
    return attempts[-1].parent


def _verify_saved_preprocessing(prepared: Any, cell: Path) -> dict[str, str]:
    shared_path = cell / "reproducibility_record.json"
    saved = _read_json(shared_path)
    repro = cell / "reproducibility"
    for file_name, item in saved["files"].items():
        file_path = repro / file_name
        if file_sha256(file_path) != item["sha256"] or file_path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Shared preprocessing file changed: {file_path}")
    for split_name in ("train", "val", "test"):
        for family, values in (("capped", prepared.splits), ("uncapped", prepared.uncapped_splits)):
            indices = np.asarray(values[split_name], dtype=np.int64)
            expected = saved[family][split_name]
            if indices.size != int(expected["count"]) or array_sha256(indices) != expected["sha256"]:
                raise ValueError(f"{family}/{split_name} does not reproduce the saved preprocessing: {cell}")
    with np.load(repro / "normalization_stats.npz", allow_pickle=False) as values:
        if not np.array_equal(values["mean"], prepared.normalization_mean) or not np.array_equal(
            values["std"], prepared.normalization_std
        ):
            raise ValueError(f"Normalization does not reproduce the saved preprocessing: {cell}")
    expected_thresholds = _read_json(repro / "thresholds.json")
    actual_thresholds = {
        "step": {str(key): {str(k): float(v) for k, v in value.items()} for key, value in prepared.step_thresholds.items()},
        "outcome": {str(key): {str(k): float(v) for k, v in value.items()} for key, value in prepared.outcome_thresholds.items()},
    }
    if actual_thresholds != expected_thresholds:
        raise ValueError(f"Label thresholds do not reproduce the saved preprocessing: {cell}")
    return {"record": file_sha256(shared_path), **{name: item["sha256"] for name, item in saved["files"].items()}}


def _model_from_bundle(bundle: dict[str, Any], requested_model: str, device: torch.device) -> tuple[torch.nn.Module, str, int]:
    config = dict(bundle["model_config"])
    saved_name = str(config["model_name"])
    if saved_name.replace("-", "_") != requested_model:
        raise ValueError(f"Saved bundle model mismatch: {saved_name} != {requested_model}")
    spec = get_precursor_model_spec(saved_name)
    if not spec.trainable:
        raise ValueError("Offline replay uses the three canonical trained models")
    model = build_precursor_model(
        saved_name,
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        num_classes=int(config["num_classes"]),
        cnn_kernel_size=int(config["cnn_kernel_size"]),
        mlp_depth=int(config["mlp_depth"]),
        transformer_d_model=config.get("transformer_d_model"),
        transformer_num_layers=int(config.get("transformer_num_layers", 2)),
        transformer_num_heads=int(config.get("transformer_num_heads", 4)),
        transformer_ffn_multiplier=int(config.get("transformer_ffn_multiplier", 4)),
        transformer_max_length=int(config.get("transformer_max_length", bundle["window_size"])),
    )
    model.load_state_dict(bundle["model_state_dict"])
    positive_index = int(bundle.get("metrics", {}).get("positive_index", int(config["num_classes"]) - 1))
    return model.to(device), spec.input_view, positive_index


def _predict_split(
    prepared: Any,
    bundle: dict[str, Any],
    model: torch.nn.Module,
    input_view: str,
    positive_index: int,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    datasets = _build_model_datasets(
        prepared.dataset,
        prepared.splits,
        prepared.training_labels,
        normalized_features=prepared.normalized_features,
        feature_profile=prepared.feature_profile,
        model_name=str(bundle["model_config"]["model_name"]),
        window_size=int(bundle["window_size"]),
        shuffle_seed=int(prepared.seed),
    )
    loader = _make_loader(datasets[split_name], batch_size=batch_size, shuffle=False, num_workers=0)
    output = _collect_outputs(
        model, loader, device, input_view=input_view, temperature=float(bundle["temperature"])
    )
    indices = np.asarray(prepared.splits[split_name], dtype=np.int64)
    if not np.array_equal(output["sample_index"], indices):
        raise ValueError(f"Inference order differs from frozen {split_name} indices")
    scores = _extract_positive_scores(output["probabilities"], positive_index=positive_index)
    if scores.size != indices.size or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid replay scores")
    return scores, output


def _predict_split_fast(
    prepared: Any,
    bundle: dict[str, Any],
    model: torch.nn.Module,
    input_view: str,
    positive_index: int,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if prepared.feature_profile != "full" or input_view not in {"sequence", "flat"}:
        raise ValueError("Vectorized replay supports only the frozen canonical full profile")
    indices = np.asarray(prepared.splits[split_name], dtype=np.int64)
    episode_positions = np.asarray(prepared.dataset["episode_position"], dtype=np.int64)
    episode_uids = np.asarray(prepared.dataset["episode_uid"], dtype=np.int64)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            batch_indices = indices[start : start + batch_size]
            windows = full_prefix_windows(
                prepared.normalized_features,
                batch_indices,
                episode_positions,
                episode_uids,
                window_size=int(bundle["window_size"]),
            )
            tensor = torch.from_numpy(windows)
            if input_view == "flat":
                tensor = tensor.reshape(tensor.shape[0], -1)
            logits = model(tensor.to(device))
            probabilities = torch.softmax(logits / max(float(bundle["temperature"]), 1e-4), dim=-1)
            scores.append(probabilities[:, positive_index].cpu().numpy())
    values = np.concatenate(scores).astype(np.float64) if scores else np.zeros(0, dtype=np.float64)
    if values.size != indices.size or not np.all(np.isfinite(values)):
        raise ValueError("Invalid vectorized replay scores")
    output = {
        "sample_index": indices,
        "episode_uid": episode_uids[indices],
        "steps_remaining": np.asarray(prepared.dataset["steps_remaining"], dtype=np.float32)[indices],
    }
    return values, output


def _run_model(
    *,
    task: str,
    platform: str,
    seed: int,
    model_name: str,
    prepared: Any,
    cell: Path,
    shared_hashes: dict[str, str],
    gate_hash: str,
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> str:
    attempt = _latest_attempt(cell, model_name)
    manifest_path = attempt / "run_manifest.json"
    manifest = _read_json(manifest_path)
    bundle_path = attempt / "precursor_model.pt"
    metrics_path = attempt / "metrics.json"
    for name, path in (("precursor_model.pt", bundle_path), ("metrics.json", metrics_path)):
        if file_sha256(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"Saved model parent changed: {path}")
    protocol = manifest["protocol"]
    if any(int(protocol[f"{stream}_seed"]) != seed for stream in ("split", "sampling", "train", "loader")):
        raise ValueError("Model does not use the explicit four-seed protocol")
    if protocol["feature_profile"] != "full" or protocol["split_family"] != "episode_group":
        raise ValueError("Model is not canonical full")
    if manifest["training"]["loader_rng_mode"] != "explicit_loader_seed":
        raise ValueError("Model has legacy loader RNG")
    if manifest["dataset"]["sha256"] != prepared.dataset_sha256:
        raise ValueError("Precursor dataset differs from Saved model parent")
    if manifest["indices"]["capped"] != _read_json(cell / "reproducibility_record.json")["capped"]:
        raise ValueError("Model does not reference shared split indices")
    parents = {
        "g5a_gate_sha256": gate_hash,
        "dataset_sha256": prepared.dataset_sha256,
        "run_manifest_sha256": file_sha256(manifest_path),
        "model_bundle_sha256": manifest["artifacts"]["precursor_model.pt"]["sha256"],
        "metrics_sha256": manifest["artifacts"]["metrics.json"]["sha256"],
        "shared_preprocessing_sha256": shared_hashes,
    }
    target = output_root / task / f"split_seed_{seed}" / f"{model_name}.json"
    if target.is_file():
        previous = _read_json(target)
        if previous.get("parents") != parents or previous.get("result_sha256") != canonical_json_sha256(previous.get("results", {})):
            raise ValueError(f"Existing Stage 9 result failed resume validation: {target}")
        return "reused"

    require_free_space(output_root, minimum_gib=10.0 + CELL_OUTPUT_RESERVE_GIB)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    model, input_view, positive_index = _model_from_bundle(bundle, model_name, device)
    expected = _read_json(metrics_path)
    split_data: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for split_name in ("val", "test"):
        scores, output = _predict_split_fast(
            prepared, bundle, model, input_view, positive_index, split_name, device, batch_size
        )
        step_cat = (prepared.step_labels[output["sample_index"]] == 2).astype(np.int64)
        replay_ap = binary_average_precision(scores, step_cat)
        saved_ap = float(expected["label_step"][split_name]["overall"]["catastrophic_auprc"])
        if not np.isclose(replay_ap, saved_ap, rtol=0.0, atol=1e-7):
            raise ValueError(f"Saved {split_name} Step AUPRC mismatch for {task}/{seed}/{model_name}: {replay_ap} vs {saved_ap}")
        split_data[split_name] = scores, output
    val_scores, _ = split_data["val"]
    test_scores, test_output = split_data["test"]
    test_indices = test_output["sample_index"]
    results = {
        str(budget): evaluate_budget(
            val_scores,
            budget=budget,
            test_scores=test_scores,
            test_episode_ids=test_output["episode_uid"],
            test_outcome_labels=prepared.outcome_labels[test_indices],
            test_steps_remaining=test_output["steps_remaining"],
        )
        for budget in STAGE9_BUDGETS
    }
    payload = {
        "schema_version": "ottp_revision_stage9_utility_result_v1",
        "task_name": task,
        "source_platform": platform,
        "split_seed": seed,
        "model_name": model_name,
        "condition": "G5A_canonical_full",
        "threshold_source": "validation_step_scores_only",
        "outcome_use": "test_episode_evaluation_only",
        "lead_definition": "first_alarm_steps_to_episode_end_not_catastrophe_onset",
        "parents": parents,
        "results": results,
        "result_sha256": canonical_json_sha256(results),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    del model, bundle, split_data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return "computed"


def main() -> None:
    global EXPERIMENT, PRECURSORS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-names", nargs="*", default=None)
    parser.add_argument("--split-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--models", nargs="*", choices=STAGE9_MODELS, default=list(STAGE9_MODELS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    output_root = Path(args.output_dir).resolve()
    from ctsc.scripts.run_benchmark import output_directory
    output_root = output_directory(output_root)
    EXPERIMENT = args.experiment_root.resolve()
    PRECURSORS = args.dataset_dir.resolve()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    gate_hash = file_sha256(EXPERIMENT / "matrix.json")
    registry = _read_json(REGISTRY)
    task_rows = {str(row["task_name"]): row for row in registry["tasks"]}
    tasks = list(task_rows) if args.task_names is None else list(dict.fromkeys(args.task_names))
    if not tasks or any(task not in task_rows for task in tasks):
        raise ValueError("Unknown or empty task selection")
    seeds = (0, 1, 2) if args.split_seeds is None else tuple(dict.fromkeys(args.split_seeds))
    if not seeds or any(seed not in (0, 1, 2) for seed in seeds):
        raise ValueError("Only split seeds 0/1/2 are permitted")
    canonical = next(row for row in registry["mainline_fresh_rerun"]["slices"] if row["name"] == "canonical")
    defaults = registry["mainline_fresh_rerun"]["shared_defaults"]
    device = torch.device(args.device)
    for task in tasks:
        dataset_path = PRECURSORS / f"{task}.pt"
        for seed in seeds:
            require_free_space(output_root, minimum_gib=10.0 + CELL_OUTPUT_RESERVE_GIB)
            cell = EXPERIMENT / "cell_runs" / "canonical_full" / task / f"split_seed_{seed}"
            prepared = prepare_precursor_inputs(
                dataset_path,
                seed=seed,
                split_seed=seed,
                sampling_seed=seed,
                train_seed=seed,
                loader_seed=seed,
                split_family="episode_group",
                ablation_group=None,
                ablation_mode=None,
                feature_profile="full",
                step_q_low=float(canonical["step_q_low"]),
                step_q_high=float(canonical["step_q_high"]),
                outcome_q_low=float(canonical["outcome_q_low"]),
                outcome_q_high=float(canonical["outcome_q_high"]),
                max_train_samples=defaults["max_train_samples"],
                max_val_samples=defaults["max_val_samples"],
                max_test_samples=defaults["max_test_samples"],
                target_variant="current_inclusive",
                gamma=0.99,
            )
            shared = _verify_saved_preprocessing(prepared, cell)
            for model_name in args.models:
                state = _run_model(
                    task=task,
                    platform=str(task_rows[task]["source_platform"]),
                    seed=seed,
                    model_name=model_name,
                    prepared=prepared,
                    cell=cell,
                    shared_hashes=shared,
                    gate_hash=gate_hash,
                    output_root=output_root,
                    device=device,
                    batch_size=args.batch_size,
                )
                print(json.dumps({"task": task, "split_seed": seed, "model": model_name, "state": state}), flush=True)


if __name__ == "__main__":
    main()
