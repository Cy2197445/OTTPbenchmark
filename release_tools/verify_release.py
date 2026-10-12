"""Verify the distributed archive, run records, indices, and result tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def archive_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Archive path escapes its root: {relative}")
    return path


def verify_files() -> int:
    count = 0
    for line in (ROOT / "provenance/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = archive_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed archive file: {relative}")
        count += 1
    return count


def verify_runs() -> int:
    from ctsc.precursor.revision import assert_valid_run_manifest
    manifests = sorted((ROOT / "runs").rglob("run_manifest.json"))
    if not manifests:
        raise ValueError("No recorded runs")
    for path in manifests:
        record = read_json(path)
        assert_valid_run_manifest(record)
        metrics_path = path.with_name("metrics.json")
        if sha256(metrics_path) != record["artifacts"]["metrics.json"]["sha256"]:
            raise ValueError(f"Metrics hash mismatch: {path}")
        metrics = read_json(metrics_path)
        if len(metrics["training"]["history"]) != int(record["training"]["epochs"]):
            raise ValueError(f"Incomplete training history: {path}")
        for epoch in metrics["training"]["history"]:
            if any(not math.isfinite(float(epoch[k])) for k in ("train_loss", "val_catastrophic_auprc", "temperature")):
                raise ValueError(f"Non-finite training history: {path}")
    rows = read_csv(ROOT / "results/model_comparison/seed_level_all_results.csv")
    if len(rows) != 1026:
        raise ValueError("Incomplete main model matrix")
    keys = {(r["condition_id"], r["task_name"], int(r["split_seed"]), r["model_name"]) for r in rows}
    tasks = {r["task_name"] for r in rows}
    expected = {(condition, task, seed, model) for condition in ("canonical_full", "canonical_current_only", "sensitivity_full") for task in tasks for seed in (0, 1, 2) for model in ("gru", "flat_mlp", "transformer")}
    if len(tasks) != 38 or keys != expected:
        raise ValueError("Main task/seed/model coverage mismatch")
    for row in rows:
        path = archive_path(row["metrics_path"])
        if sha256(path) != row["metrics_sha256"]:
            raise ValueError(f"Seed table does not match its run: {path}")
        check_seed_metric(row, path)
    return len(manifests)


def check_seed_metric(row, path):
    metrics = read_json(path)
    step = metrics["label_step"]["test"]["overall"]
    outcome = metrics["label_outcome"]["test"]["overall"]
    expected = {
        "step_auprc": step["catastrophic_auprc"],
        "matched_step_prevalence": step["catastrophic_prevalence"],
        "step_auprc_minus_prevalence": step["catastrophic_auprc"] - step["catastrophic_prevalence"],
        "outcome_auprc": outcome["catastrophic_auprc"],
        "matched_outcome_prevalence": outcome["catastrophic_prevalence"],
        "step_ece": step["catastrophic_ece"],
        "step_lead_median": step["lead_time_median_at_0.5"],
    }
    for field, value in expected.items():
        if not math.isclose(float(row[field]), float(value), rel_tol=0, abs_tol=1e-11):
            raise ValueError(f"Seed-table metric mismatch: {path}/{field}")


def verify_indices() -> int:
    import numpy as np
    from ctsc.precursor.revision import array_sha256

    reuse = read_json(ROOT / "manifests/index_reuse.json")["rows"]
    if len(reuse) != 342 or len({r["condition_cell"] for r in reuse}) != 342:
        raise ValueError("Incomplete shared-index mapping")
    checked = {}
    for row in reuse:
        cell = archive_path(row["condition_cell"])
        records = [read_json(sorted((cell / "models" / model).glob("attempt-*/run_manifest.json"))[-1])["indices"] for model in ("gru", "flat_mlp", "transformer")]
        if records[0] != records[1] or records[0] != records[2]:
            raise ValueError(f"Models do not share preprocessing: {cell}")
        record = records[0]
        for name, expected in row["shared_array_sha256"].items():
            path = archive_path(row["shared_arrays_at"]) / name
            if path not in checked:
                checked[path] = sha256(path)
                if checked[path] != expected:
                    raise ValueError(f"Shared-array hash mismatch: {path}")
                with np.load(path, allow_pickle=False) as data:
                    if any(not np.isfinite(data[k]).all() for k in data.files):
                        raise ValueError(f"Non-finite array: {path}")
                    if name == "uncapped_split_indices.npz":
                        splits = [data[k] for k in ("train", "validation", "test")]
                        combined = np.concatenate(splits)
                        if np.unique(combined).size != combined.size:
                            raise ValueError(f"Overlapping split indices: {path}")
            elif checked[path] != expected:
                raise ValueError(f"Inconsistent shared-array hash mapping: {path}")
            if record["files"][name]["sha256"] != expected:
                raise ValueError(f"Run record and shared-index mapping disagree: {cell}/{name}")
        for kind, filename in (("uncapped", "uncapped_split_indices.npz"), ("capped", "capped_train_validation_indices.npz")):
            with np.load(archive_path(row["shared_arrays_at"]) / filename, allow_pickle=False) as arrays:
                for split in ("train", "validation", "test") if kind == "uncapped" else ("train", "validation"):
                    key = "val" if split == "validation" else split
                    values = arrays[split]
                    if values.size != record[kind][key]["count"] or array_sha256(values) != record[kind][key]["sha256"]:
                        raise ValueError(f"Index-array content does not match run: {cell}/{kind}/{key}")
        for filename, relative in (("thresholds.json", row["own_thresholds_at"]), ("episode_group_assignment.csv.gz", row["own_episode_assignment_at"])):
            if sha256(archive_path(relative)) != record["files"][filename]["sha256"]:
                raise ValueError(f"Condition-specific preprocessing changed: {cell}/{filename}")
        thresholds = read_json(archive_path(row["own_thresholds_at"]))
        if any(not math.isfinite(float(v)) for endpoint in thresholds.values() for env in endpoint.values() for v in env.values()):
            raise ValueError(f"Non-finite label threshold: {cell}")
        with gzip.open(archive_path(row["own_episode_assignment_at"]), "rt", encoding="utf-8", newline="") as handle:
            groups = {}
            for item in csv.DictReader(handle):
                key = item["episode_group_index"]
                if key in groups and groups[key] != item["split"]:
                    raise ValueError("Episode group spans dataset splits")
                groups[key] = item["split"]
    return len(reuse)


def compare(actual, expected, keys):
    left = {tuple(str(row[k]) for k in keys): row for row in actual}
    right = {tuple(str(row[k]) for k in keys): row for row in expected}
    if len(left) != len(actual) or len(right) != len(expected):
        raise ValueError(f"Duplicate table keys: {keys}")
    if left.keys() != right.keys():
        raise ValueError(f"Table key mismatch: {keys}")
    for key in left:
        for field, value in left[key].items():
            if isinstance(value, (int, float)) and field not in right[key]:
                raise ValueError(f"Missing numeric field: {key}/{field}")
            if isinstance(value, bool):
                if str(value).lower() != str(right[key][field]).lower():
                    raise ValueError(f"Boolean mismatch: {key}/{field}")
            elif isinstance(value, (int, float)):
                a, b = float(value), float(right[key][field])
                if not abs(a - b) <= 1e-11:
                    raise ValueError(f"Numeric mismatch: {key}/{field}: {a} != {b}")
            elif value is None and right[key].get(field) not in (None, ""):
                raise ValueError(f"Undefined value mismatch: {key}/{field}")


def verify_statistics():
    from ctsc.precursor.stage4 import build_frozen_statistics as ablation_statistics, task_level_rows as ablation_tasks
    from ctsc.precursor.stage5 import build_frozen_statistics as main_statistics
    from ctsc.precursor.stage10 import make_tables, validate_fair_rows
    from ctsc.scripts.analyze_revision_stage5_fair_matrix import _task_rows

    seed_rows = read_csv(ROOT / "results/model_comparison/seed_level_all_results.csv")
    task_rows = _task_rows(seed_rows)
    compare(task_rows, read_csv(ROOT / "results/model_comparison/task_level_all_results.csv"), ("condition_id", "task_name", "model_name"))
    main, outcome, history = make_tables(validate_fair_rows(task_rows))
    for rows, name, keys in ((main, "fixed_three_model_main_step.csv", ("task_name",)), (outcome, "fixed_three_model_outcome_auxiliary.csv", ("task_name", "model_name")), (history, "history_gain_task_level.csv", ("task_name", "model_name"))):
        compare(rows, read_csv(ROOT / "results/main" / name), keys)
    paired, stats = main_statistics(task_rows)
    compare(stats, read_csv(ROOT / "results/model_comparison/f1_f2_f3_statistics.csv"), ("family", "comparison"))
    compare(paired, read_csv(ROOT / "results/model_comparison/f1_f2_f3_paired_task_differences.csv"), ("family", "comparison", "task_name"))
    ablation_rows = read_csv(ROOT / "results/target_ablation/task_level_all_results.csv")
    ablation_seeds = read_csv(ROOT / "results/target_ablation/seed_level_all_results.csv")
    for row in ablation_seeds:
        check_seed_metric(row, archive_path(row["metrics_path"]))
        row["reused_g0"] = row["reused_g0"].lower() == "true"
    compare(ablation_tasks(ablation_seeds), ablation_rows, ("matrix", "condition_id", "task_name", "model_name"))
    paired4, stats4 = ablation_statistics(ablation_rows)
    compare(stats4, read_csv(ROOT / "results/target_ablation/f4_f5_statistics.csv"), ("family", "comparison"))
    compare(paired4, read_csv(ROOT / "results/target_ablation/f4_f5_paired_task_differences.csv"), ("family", "comparison", "task_name"))
    from ctsc.scripts.analyze_revision_stage9_utility import collect_results, _task_rows as utility_tasks, _summaries, _paired
    utility_seed_rows = collect_results(ROOT / "runs/offline_utility/cell_runs")
    utility_rows = utility_tasks(utility_seed_rows)
    utility_keys = ("task_name", "model_name", "alarm_budget")
    compare(utility_rows, read_csv(ROOT / "results/offline_utility/task_level_342_budget_results.csv"), utility_keys)
    overall, environment, groups = _summaries(utility_rows, rounds=20000, seed=260922)
    paired_utility, raw_utility = _paired(utility_rows, rounds=20000, seed=260922)
    for rows, filename, keys in (
        (overall, "overall_9_model_budget_summary.csv", ("model_name", "alarm_budget")),
        (environment, "simulation_environment_27_descriptive_summary.csv", ("model_name", "alarm_budget", "simulation_environment")),
        (groups, "registry_task_group_36_descriptive_summary.csv", ("model_name", "alarm_budget", "registry_task_group")),
        (paired_utility, "paired_model_difference_summary_descriptive.csv", ("alarm_budget", "metric", "difference")),
        (raw_utility, "paired_model_task_differences_raw.csv", ("alarm_budget", "metric", "task_name", "left_model", "right_model")),
    ):
        compare(rows, read_csv(ROOT / "results/offline_utility" / filename), keys)
    from ctsc.scripts.analyze_revision_stage7 import _numeric_summary
    stability = read_csv(ROOT / "results/seed_stability/within_fixed_split_stability_24_task_models.csv")
    recomputed = []
    for row in stability:
        scores = json.loads(row["step_auprc_values_by_train_seed"])
        epochs = json.loads(row["best_epoch_values_by_train_seed"])
        recomputed.append({"task": row["task"], "model": row["model"], **_numeric_summary(scores),
                           **{"best_epoch_" + k: v for k, v in _numeric_summary(epochs).items()}})
    compare(recomputed, stability, ("task", "model"))
    return {"main_tasks": len(main), "history_rows": len(history), "main_comparisons": len(stats), "ablation_comparisons": len(stats4), "offline_utility_seed_rows": len(utility_seed_rows), "seed_stability_rows": len(stability)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", action="store_true", help="Recompute task tables and F1-F5 comparisons")
    args = parser.parse_args()
    result = {"files_checked": verify_files(), "runs_checked": verify_runs(), "index_cells_checked": verify_indices()}
    if args.statistics:
        result["statistics"] = verify_statistics()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
