"""Build seed metrics, main tables, history gains, and paired statistics from model runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from ctsc.precursor.cc1 import CONDITIONS_BY_ID, FAIR_MODELS, LOCKED_MODEL_LRS
from ctsc.precursor.revision import assert_valid_run_manifest, file_sha256
from ctsc.precursor.stage5 import build_frozen_statistics
from ctsc.precursor.stage10 import make_tables, validate_fair_rows, write_csv, write_json
from ctsc.scripts.analyze_revision_stage5_fair_matrix import _task_rows
from ctsc.scripts.run_benchmark import ROOT, output_directory


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_results(experiment: Path, *, partial: bool = False) -> list[dict]:
    experiment = experiment.resolve()
    registry = read_json(ROOT / "manifests/experiment_registry.json")
    platforms = {r["task_name"]: r["source_platform"] for r in registry["tasks"]}
    defaults = registry["mainline_fresh_rerun"]["shared_defaults"]
    archived = experiment == (ROOT / "runs/main").resolve()
    reuse = {}
    if archived:
        reuse = {r["condition_cell"]: r for r in read_json(ROOT / "manifests/index_reuse.json")["rows"]}
    plan_path = experiment / "matrix.json"
    planned = None
    if plan_path.is_file():
        plan = read_json(plan_path)
        planned = {(c["condition_id"], c["task_name"], int(c["split_seed"]), model)
                   for c in plan["cells"] for model in c["models"]}
        if not partial and plan.get("smoke"):
            raise ValueError("Smoke runs require --partial; full-matrix statistics use the recorded training settings")
    expected = {(c, task, seed, model) for c in CONDITIONS_BY_ID for task in platforms
                for seed in (0, 1, 2) for model in FAIR_MODELS}
    cells = sorted((experiment / "cell_runs").glob("*/*/split_seed_*"))
    rows = []
    keys = set()
    shared_by_task_seed = {}
    for cell in cells:
        condition_id, task = cell.parent.parent.name, cell.parent.name
        seed = int(cell.name.removeprefix("split_seed_"))
        if condition_id not in CONDITIONS_BY_ID or task not in platforms or seed not in (0, 1, 2):
            raise ValueError(f"Unknown experiment cell: {cell}")
        condition = CONDITIONS_BY_ID[condition_id]
        shared_path = cell / "reproducibility_record.json"
        if archived and not shared_path.is_file():
            first = sorted((cell / "models/gru").glob("attempt-*/run_manifest.json"))[-1]
            shared = read_json(first)["indices"]
        else:
            shared = read_json(shared_path)
        mapping = reuse.get(cell.relative_to(ROOT).as_posix()) if archived else None
        for name, metadata in shared["files"].items():
            if Path(name).name != name:
                raise ValueError(f"Invalid preprocessing file name: {name}")
            parent = ROOT / mapping["shared_arrays_at"] if mapping and name.endswith(".npz") else cell / "reproducibility"
            path = parent / name
            if not path.is_file() or file_sha256(path) != metadata["sha256"]:
                raise ValueError(f"Preprocessing hash mismatch: {path}")
        shared_hashes = tuple(shared["files"][name]["sha256"] for name in
                              ("uncapped_split_indices.npz", "capped_train_validation_indices.npz", "normalization_stats.npz"))
        old = shared_by_task_seed.setdefault((task, seed), shared_hashes)
        if old != shared_hashes:
            raise ValueError(f"Condition preprocessing differs for {task}/seed {seed}")
        prevalences = []
        for model in FAIR_MODELS:
            attempts = sorted((cell / "models" / model).glob("attempt-*/run_manifest.json"))
            if not attempts:
                raise FileNotFoundError(f"No model result for {cell}/{model}")
            manifest_path = attempts[-1]
            record = read_json(manifest_path)
            assert_valid_run_manifest(record)
            protocol, training = record["protocol"], record["training"]
            if record["dataset"]["task"] != task or str(protocol["model_name"]).replace("-", "_") != model:
                raise ValueError(f"Model/task identity mismatch: {manifest_path}")
            if record["indices"] != shared:
                raise ValueError(f"Model does not share preprocessing: {manifest_path}")
            if any(int(protocol[k]) != seed for k in ("split_seed", "sampling_seed", "train_seed", "loader_seed")):
                raise ValueError(f"Four-seed mismatch: {manifest_path}")
            if (protocol.get("legacy_compatible") is not False
                    or training.get("loader_rng_mode") != "explicit_loader_seed"
                    or protocol["feature_profile"] != condition.feature_profile
                    or protocol["target_variant"] != "current_inclusive"
                    or protocol["split_family"] != "episode_group"
                    or float(protocol["gamma"]) != 0.99
                    or protocol["quantile_pair"] != [condition.q_low, condition.q_high]
                    or protocol["outcome_quantile_pair"] != [condition.q_low, condition.q_high]
                    or not math.isclose(float(training["learning_rate"]), LOCKED_MODEL_LRS[model], rel_tol=0, abs_tol=1e-15)):
                raise ValueError(f"Main-matrix protocol mismatch: {manifest_path}")
            if not partial:
                for field in ("epochs", "batch_size", "hidden_dim", "num_layers", "dropout", "weight_decay", "window_size"):
                    if training[field] != defaults[field]:
                        raise ValueError(f"Full-matrix {field} mismatch: {manifest_path}")
                for field in ("max_train_samples", "max_val_samples", "max_test_samples"):
                    if protocol["extra"][field] != defaults[field]:
                        raise ValueError(f"Full-matrix {field} mismatch: {manifest_path}")
            metrics_path = manifest_path.with_name("metrics.json")
            if file_sha256(metrics_path) != record["artifacts"]["metrics.json"]["sha256"]:
                raise ValueError(f"Metrics hash mismatch: {metrics_path}")
            bundle = manifest_path.with_name("precursor_model.pt")
            if bundle.is_file():
                if file_sha256(bundle) != record["artifacts"]["precursor_model.pt"]["sha256"]:
                    raise ValueError(f"Model hash mismatch: {bundle}")
            elif not archived:
                raise FileNotFoundError(bundle)
            metrics = read_json(metrics_path)
            history = metrics["training"]["history"]
            if (len(history) != int(training["epochs"]) or not history
                    or any(not math.isfinite(float(e[k])) for e in history
                           for k in ("train_loss", "val_selection_auprc", "temperature"))
                    or int(metrics["training"]["best_epoch"]) != int(np.argmax([e["val_selection_auprc"] for e in history]))):
                raise ValueError(f"Training history mismatch: {metrics_path}")
            if metrics["feature_profile"] != condition.feature_profile:
                raise ValueError(f"Metric feature profile mismatch: {metrics_path}")
            step, outcome = metrics["label_step"]["test"]["overall"], metrics["label_outcome"]["test"]["overall"]
            numeric = [float(step["catastrophic_auprc"]), float(step["catastrophic_prevalence"]),
                       float(outcome["catastrophic_auprc"]), float(outcome["catastrophic_prevalence"]),
                       float(step["catastrophic_ece"]), float(step["lead_time_median_at_0.5"])]
            if not all(math.isfinite(v) for v in numeric):
                raise ValueError(f"Nonfinite test metric: {metrics_path}")
            key = (condition_id, task, seed, model)
            if key in keys:
                raise ValueError(f"Duplicate model result: {key}")
            keys.add(key)
            prevalences.append((numeric[1], numeric[3]))
            rows.append({"condition_id": condition_id, "condition_role": condition.role,
                         "task_name": task, "source_platform": platforms[task], "split_seed": seed,
                         "model_name": model, "learning_rate": LOCKED_MODEL_LRS[model],
                         "feature_profile": condition.feature_profile, "q_low": condition.q_low, "q_high": condition.q_high,
                         "step_auprc": numeric[0], "matched_step_prevalence": numeric[1],
                         "step_auprc_minus_prevalence": numeric[0] - numeric[1], "outcome_auprc": numeric[2],
                         "matched_outcome_prevalence": numeric[3], "step_ece": numeric[4], "step_lead_median": numeric[5],
                         "metrics_path": str(metrics_path), "bundle_path": str(bundle), "metrics_sha256": file_sha256(metrics_path)})
        if len(set(prevalences)) != 1:
            raise ValueError(f"Model prevalences differ: {cell}")
    if not keys:
        raise ValueError(f"No main-matrix runs found: {experiment}")
    if planned is not None and keys != planned:
        raise ValueError("Model results do not cover the recorded matrix.json plan")
    if not partial and keys != expected:
        raise ValueError("Incomplete 38-task matrix; use --partial to export subset seed metrics")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="Export subset or smoke seed metrics without full-matrix statistics")
    args = parser.parse_args()
    output = output_directory(args.output_dir)
    experiment = args.experiment_root.resolve()
    if output == experiment or experiment.is_relative_to(output) or output.is_relative_to(experiment):
        raise ValueError("Analysis output must be separate from the experiment root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Analysis output is nonempty: {output}")
    rows = collect_results(experiment, partial=args.partial)
    tables = {"seed_level_all_results.csv": rows}
    if not args.partial:
        tasks = _task_rows(rows)
        main_table, outcome, history = make_tables(validate_fair_rows(tasks))
        paired, statistics = build_frozen_statistics(tasks)
        tables.update({"task_level_all_results.csv": tasks, "fixed_three_model_main_step.csv": main_table,
                       "fixed_three_model_outcome_auxiliary.csv": outcome, "history_gain_task_level.csv": history,
                       "f1_f2_f3_paired_task_differences.csv": paired, "f1_f2_f3_statistics.csv": statistics})
    for name, table in tables.items():
        write_csv(output / name, table)
    summary = {"experiment_root": str(experiment), "partial": args.partial,
               "tables": {name: len(table) for name, table in tables.items()}}
    write_json(output / "analysis_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
