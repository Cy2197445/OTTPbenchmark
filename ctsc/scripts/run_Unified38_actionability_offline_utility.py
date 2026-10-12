'Run the Unified38 offline actionability / downstream-utility study.'

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from ctsc.precursor.actionability import (
    DEFAULT_ACTIONABILITY_BUDGETS,
    DEFAULT_ACTIONABILITY_MODELS,
    normalize_actionability_model_name,
    parse_budget_values,
)
from ctsc.scripts.train_precursor_model import (
    _build_model_datasets,
    _collect_outputs,
    _collect_reference_outputs,
    _extract_positive_scores,
    _make_loader,
    prepare_precursor_inputs,
)
from ctsc.precursor.model import PrevalenceBaseline, build_precursor_model, get_precursor_model_spec
from ctsc.scripts.run_benchmark import output_directory


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "manifests" / "experiment_registry.json"
DEFAULT_TASK_RUN_ROOT = REPO_ROOT / "work" / "reference_diagnostics" / "task_runs"
DEFAULT_PRECURSOR_SOURCE_DIR = REPO_ROOT / "work" / "precursors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=str, default=str(REGISTRY_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--task-names", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="work/reference_utility")
    parser.add_argument("--task-run-root", type=str, default=str(DEFAULT_TASK_RUN_ROOT.relative_to(REPO_ROOT)))
    parser.add_argument(
        "--precursor-source-dir",
        type=str,
        default=str(DEFAULT_PRECURSOR_SOURCE_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_ACTIONABILITY_MODELS))
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_ACTIONABILITY_BUDGETS))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _canonical_slice(registry: dict[str, Any]) -> dict[str, Any]:
    for slice_spec in registry["mainline_fresh_rerun"]["slices"]:
        if str(slice_spec["name"]) == "canonical":
            return dict(slice_spec)
    raise ValueError("Registry does not contain a canonical slice.")


def _resolve_positive_index(bundle: dict[str, Any]) -> int:
    metrics = bundle.get("metrics")
    if isinstance(metrics, dict) and "positive_index" in metrics:
        return int(metrics["positive_index"])
    model_config = bundle.get("model_config")
    if isinstance(model_config, dict):
        num_classes = int(model_config.get("num_classes", 0))
        if num_classes >= 2:
            return num_classes - 1
    raise KeyError("Unable to resolve positive_index from saved bundle.")


def _prediction_rows_from_outputs(
    *,
    task_name: str,
    slice_name: str,
    split_seed: int,
    model_name: str,
    split_name: str,
    outputs: dict[str, np.ndarray],
    prepared: Any,
    score_values: np.ndarray,
) -> list[dict[str, Any]]:
    sample_index = np.asarray(outputs["sample_index"], dtype=np.int64)
    step_labels = prepared.step_labels[sample_index]
    outcome_labels = prepared.outcome_labels[sample_index]
    critic_scores = np.asarray(outputs["critic_score"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for idx in range(sample_index.shape[0]):
        rows.append(
            {
                "task_name": task_name,
                "slice_name": slice_name,
                "split_seed": int(split_seed),
                "model_name": normalize_actionability_model_name(model_name),
                "split_name": split_name,
                "sample_index": int(sample_index[idx]),
                "episode_uid": int(outputs["episode_uid"][idx]),
                "env_index": int(outputs["env_index"][idx]),
                "steps_remaining": float(outputs["steps_remaining"][idx]),
                "score_catastrophic": float(score_values[idx]),
                "step_label": int(step_labels[idx]),
                "outcome_label": int(outcome_labels[idx]),
                "critic_score": float(critic_scores[idx]),
            }
        )
    return rows


def _replay_model_predictions(
    *,
    task_name: str,
    slice_name: str,
    split_seed: int,
    model_name: str,
    prepared: Any,
    bundle_path: Path | None,
    device: str,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    if model_name == "critic_score":
        rows: list[dict[str, Any]] = []
        for split_name in ["val", "test"]:
            indices = np.asarray(prepared.splits[split_name], dtype=np.int64)
            outputs = {
                "sample_index": indices,
                "episode_uid": np.asarray(prepared.dataset["episode_uid"], dtype=np.int64)[indices],
                "env_index": np.asarray(prepared.dataset["env_index"], dtype=np.int64)[indices],
                "steps_remaining": np.asarray(prepared.dataset["steps_remaining"], dtype=np.float32)[indices],
                "critic_score": np.asarray(prepared.dataset["critic_score"], dtype=np.float32)[indices],
            }
            rows.extend(
                _prediction_rows_from_outputs(
                    task_name=task_name,
                    slice_name=slice_name,
                    split_seed=split_seed,
                    model_name=model_name,
                    split_name=split_name,
                    outputs=outputs,
                    prepared=prepared,
                    score_values=np.asarray(outputs["critic_score"], dtype=np.float64),
                )
            )
        return rows

    if bundle_path is None or not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found for `{model_name}`: {bundle_path}")

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    config = dict(bundle["model_config"])
    saved_model_name = str(config["model_name"])
    spec = get_precursor_model_spec(saved_model_name)
    normalized_model_name = normalize_actionability_model_name(saved_model_name)
    if normalized_model_name != normalize_actionability_model_name(model_name):
        raise ValueError(f"Bundle/model mismatch: requested={model_name} bundle={saved_model_name}")

    if spec.trainable:
        model_datasets = _build_model_datasets(
            prepared.dataset,
            prepared.splits,
            prepared.training_labels,
            normalized_features=prepared.normalized_features,
            feature_profile=prepared.feature_profile,
            model_name=saved_model_name,
            window_size=int(bundle["window_size"]),
            shuffle_seed=int(prepared.seed),
        )
        model = build_precursor_model(
            saved_model_name,
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
        model = model.to(torch.device(device))
        positive_index = _resolve_positive_index(bundle)
        rows = []
        for split_name in ["val", "test"]:
            loader = _make_loader(
                model_datasets[split_name],
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=int(num_workers),
            )
            outputs = _collect_outputs(
                model=model,
                loader=loader,
                device=torch.device(device),
                input_view=spec.input_view,
                temperature=float(bundle["temperature"]),
            )
            score_values = _extract_positive_scores(
                outputs["probabilities"],
                positive_index=positive_index,
            )
            rows.extend(
                _prediction_rows_from_outputs(
                    task_name=task_name,
                    slice_name=slice_name,
                    split_seed=split_seed,
                    model_name=saved_model_name,
                    split_name=split_name,
                    outputs=outputs,
                    prepared=prepared,
                    score_values=score_values,
                )
            )
        return rows

    baseline = build_precursor_model(
        saved_model_name,
        input_dim=int(prepared.normalized_features.shape[1]),
        num_classes=int(config["num_classes"]),
    )
    if not isinstance(baseline, PrevalenceBaseline):
        raise TypeError(f"Expected PrevalenceBaseline for `{saved_model_name}`.")
    train_probabilities = baseline.fit(
        torch.as_tensor(prepared.training_labels[prepared.splits["train"]], dtype=torch.long)
    ).cpu().numpy()
    positive_index = _resolve_positive_index(bundle)
    rows = []
    for split_name in ["val", "test"]:
        outputs = _collect_reference_outputs(
            prepared.dataset,
            prepared.splits[split_name],
            probabilities=train_probabilities,
        )
        score_values = _extract_positive_scores(
            outputs["probabilities"],
            positive_index=positive_index,
        )
        rows.extend(
            _prediction_rows_from_outputs(
                task_name=task_name,
                slice_name=slice_name,
                split_seed=split_seed,
                model_name=saved_model_name,
                split_name=split_name,
                outputs=outputs,
                prepared=prepared,
                score_values=score_values,
            )
        )
    return rows


def _prediction_summary(rows: list[dict[str, Any]], *, task_name: str, split_seed: int, model_name: str) -> dict[str, Any]:
    split_groups: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    for row in rows:
        split_groups[str(row["split_name"])].append(row)
    payload = {
        "task_name": task_name,
        "slice_name": "canonical",
        "split_seed": int(split_seed),
        "model_name": normalize_actionability_model_name(model_name),
        "row_count": int(len(rows)),
        "split_counts": {name: int(len(split_rows)) for name, split_rows in split_groups.items()},
    }
    for split_name, split_rows in split_groups.items():
        if split_rows:
            scores = np.asarray([float(row["score_catastrophic"]) for row in split_rows], dtype=np.float64)
            payload[f"{split_name}_score_min"] = float(scores.min())
            payload[f"{split_name}_score_max"] = float(scores.max())
            payload[f"{split_name}_step_cat_prevalence"] = float(
                np.mean(np.asarray([int(row["step_label"]) == 2 for row in split_rows], dtype=np.float64))
            )
            payload[f"{split_name}_outcome_cat_prevalence"] = float(
                np.mean(np.asarray([int(row["outcome_label"]) == 2 for row in split_rows], dtype=np.float64))
            )
        else:
            payload[f"{split_name}_score_min"] = 0.0
            payload[f"{split_name}_score_max"] = 0.0
            payload[f"{split_name}_step_cat_prevalence"] = 0.0
            payload[f"{split_name}_outcome_cat_prevalence"] = 0.0
    return payload


def _run_task_seed_job(job: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_precursor_inputs(
        job["dataset_path"],
        seed=int(job["split_seed"]),
        split_family="episode_group",
        ablation_group=None,
        ablation_mode=None,
        feature_profile=str(job["feature_profile"]),
        step_q_low=float(job["step_q_low"]),
        step_q_high=float(job["step_q_high"]),
        outcome_q_low=float(job["outcome_q_low"]),
        outcome_q_high=float(job["outcome_q_high"]),
        max_train_samples=job["max_train_samples"],
        max_val_samples=job["max_val_samples"],
        max_test_samples=job["max_test_samples"],
    )
    completed_models: list[str] = []
    for model_name in job["models"]:
        normalized_model = normalize_actionability_model_name(model_name)
        model_dir = Path(job["prediction_root"]) / job["task_name"] / "canonical" / f"seed_{int(job['split_seed'])}" / normalized_model
        rows_path = model_dir / "prediction_rows.csv"
        summary_path = model_dir / "prediction_summary.json"
        if bool(job["skip_existing"]) and rows_path.exists() and summary_path.exists():
            completed_models.append(normalized_model)
            continue
        bundle_path = None
        if model_name != "critic_score":
            bundle_path = Path(job["task_run_root"]) / job["task_name"] / "canonical" / f"seed_{int(job['split_seed'])}" / normalized_model / "precursor_model.pt"
        rows = _replay_model_predictions(
            task_name=str(job["task_name"]),
            slice_name="canonical",
            split_seed=int(job["split_seed"]),
            model_name=str(model_name),
            prepared=prepared,
            bundle_path=bundle_path,
            device=str(job["device"]),
            batch_size=int(job["batch_size"]),
            num_workers=int(job["num_workers"]),
        )
        _write_csv(rows_path, rows)
        _write_json(
            summary_path,
            _prediction_summary(rows, task_name=str(job["task_name"]), split_seed=int(job["split_seed"]), model_name=str(model_name)),
        )
        completed_models.append(normalized_model)
    return {
        "task_name": str(job["task_name"]),
        "split_seed": int(job["split_seed"]),
        "completed_models": completed_models,
        "success": True,
    }


def main() -> None:
    args = parse_args()
    registry = _load_json(_resolve(args.registry))
    output_dir = output_directory(_resolve(args.output_dir))
    prediction_root = output_dir / "predictions"
    aggregate_dir = output_dir / "aggregates"
    analysis_dir = output_dir / "analysis"
    prediction_root.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    task_run_root = _resolve(args.task_run_root)
    precursor_source_dir = _resolve(args.precursor_source_dir)
    tasks = (
        list(dict.fromkeys(str(task_name) for task_name in args.task_names))
        if args.task_names
        else [str(task["task_name"]) for task in registry["tasks"]]
    )
    split_seeds = list(args.split_seeds) if args.split_seeds is not None else list(registry["mainline_fresh_rerun"]["split_seeds"])
    canonical_slice = _canonical_slice(registry)
    defaults = dict(registry["mainline_fresh_rerun"]["shared_defaults"])
    models = list(dict.fromkeys(str(model_name) for model_name in args.models))
    unsupported_models = [model_name for model_name in models if model_name not in DEFAULT_ACTIONABILITY_MODELS]
    if unsupported_models:
        raise ValueError(f"Unsupported actionability comparator(s): {unsupported_models}")
    budgets = parse_budget_values(args.budgets)

    jobs: list[dict[str, Any]] = []
    for task_name in tasks:
        dataset_path = precursor_source_dir / f"{task_name}.pt"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing precursor dataset for `{task_name}`: {dataset_path}")
        for split_seed in split_seeds:
            jobs.append(
                {
                    "task_name": task_name,
                    "dataset_path": dataset_path,
                    "split_seed": int(split_seed),
                    "task_run_root": task_run_root,
                    "prediction_root": prediction_root,
                    "models": models,
                    "device": str(args.device),
                    "batch_size": int(args.batch_size),
                    "num_workers": int(args.num_workers),
                    "feature_profile": str(defaults["feature_profile"]),
                    "step_q_low": float(canonical_slice["step_q_low"]),
                    "step_q_high": float(canonical_slice["step_q_high"]),
                    "outcome_q_low": float(canonical_slice["outcome_q_low"]),
                    "outcome_q_high": float(canonical_slice["outcome_q_high"]),
                    "max_train_samples": defaults.get("max_train_samples"),
                    "max_val_samples": defaults.get("max_val_samples"),
                    "max_test_samples": defaults.get("max_test_samples"),
                    "skip_existing": bool(args.skip_existing),
                }
            )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(int(args.concurrency), 1)) as executor:
        futures = [executor.submit(_run_task_seed_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

    aggregate_cmd = [
        sys.executable,
        "ctsc/scripts/aggregate_Unified38_actionability_offline_utility.py",
        "--input-root",
        _repo_relative_or_absolute(prediction_root),
        "--output-dir",
        _repo_relative_or_absolute(aggregate_dir),
        "--task-names",
        *tasks,
        "--split-seeds",
        *[str(int(seed)) for seed in split_seeds],
        "--models",
        *models,
        "--budgets",
        *[str(float(value)) for value in budgets],
    ]
    analyze_cmd = [
        sys.executable,
        "ctsc/scripts/analyze_Unified38_actionability_offline_utility.py",
        "--aggregate-dir",
        _repo_relative_or_absolute(aggregate_dir),
        "--output-dir",
        _repo_relative_or_absolute(analysis_dir),
    ]
    subprocess.run(aggregate_cmd, cwd=str(REPO_ROOT), check=True)
    subprocess.run(analyze_cmd, cwd=str(REPO_ROOT), check=True)

    launch_summary = {
        "title": "Unified38 Offline Actionability / Downstream Utility",
        "task_count": len(tasks),
        "split_seed_count": len(split_seeds),
        "model_count": len(models),
        "budget_count": len(budgets),
        "prediction_root": str(prediction_root),
        "aggregate_dir": str(aggregate_dir),
        "analysis_dir": str(analysis_dir),
        "results": results,
    }
    _write_json(output_dir / "Unified38_actionability_offline_utility_launch_summary.json", launch_summary)
    print(
        json.dumps(
            {
                "prediction_root": str(prediction_root),
                "aggregate_dir": str(aggregate_dir),
                "analysis_dir": str(analysis_dir),
                "task_count": len(tasks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
