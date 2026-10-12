"""Run the recorded three-model, three-condition OTTP matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctsc.precursor.cc1 import CONDITIONS_BY_ID, fair_cells, write_json

ROOT = Path(__file__).resolve().parents[2]


def output_directory(value: str | Path) -> Path:
    path = Path(value).resolve()
    if path == ROOT or ROOT.is_relative_to(path) or any(path.is_relative_to(ROOT / name) for name in ("runs", "results", "manifests", "provenance", "ctsc", "tests", "docs", "release_tools")):
        raise ValueError("Choose a working output directory outside archived assets")
    return path


def selected_plan(tasks=None, seeds=None, conditions=None):
    registry = json.loads((ROOT / "manifests/experiment_registry.json").read_text(encoding="utf-8"))
    all_tasks = [row["task_name"] for row in registry["tasks"]]
    chosen_tasks = all_tasks if tasks is None else list(dict.fromkeys(tasks))
    chosen_seeds = [0, 1, 2] if seeds is None else list(dict.fromkeys(seeds))
    chosen_conditions = list(CONDITIONS_BY_ID) if conditions is None else list(dict.fromkeys(conditions))
    if not chosen_tasks or set(chosen_tasks) - set(all_tasks):
        raise ValueError("Unknown or empty task selection")
    if not chosen_seeds or set(chosen_seeds) - {0, 1, 2}:
        raise ValueError("Split seeds must be selected from 0, 1, 2")
    if not chosen_conditions or set(chosen_conditions) - set(CONDITIONS_BY_ID):
        raise ValueError("Unknown or empty condition selection")
    cells = [cell for cell in fair_cells(all_tasks) if cell["task_name"] in chosen_tasks and cell["split_seed"] in chosen_seeds and cell["condition_id"] in chosen_conditions]
    return registry["mainline_fresh_rerun"]["shared_defaults"], cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--split-seeds", nargs="+", type=int)
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS_BY_ID))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--minimum-free-space-gib", type=float, default=10.0)
    args = parser.parse_args()
    defaults, cells = selected_plan(args.tasks, args.split_seeds, args.conditions)
    if args.smoke:
        defaults.update(epochs=1, max_train_samples=1000, max_val_samples=500)
    output = output_directory(args.output_dir)
    plan = {"cells": cells, "defaults": defaults, "smoke": args.smoke, "dataset_dir": str(args.dataset_dir.resolve()), "device": args.device}
    plan_path = output / "matrix.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume or not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise FileExistsError("Output is nonempty or its recorded matrix differs")
    write_json(plan_path, plan)
    if not args.prepare_only:
        for cell in cells:
            condition = CONDITIONS_BY_ID[cell["condition_id"]]
            dataset = args.dataset_dir.resolve() / f"{cell['task_name']}.pt"
            if not dataset.is_file():
                raise FileNotFoundError(dataset)
            dest = output / "cell_runs" / cell["condition_id"] / cell["task_name"] / f"split_seed_{cell['split_seed']}"
            command = [sys.executable, "-m", "ctsc.scripts.run_revision_cc1_cell", "--dataset", str(dataset), "--cell-dir", str(dest), "--task-name", cell["task_name"], "--condition-id", cell["condition_id"], "--feature-profile", condition.feature_profile, "--q-low", str(condition.q_low), "--q-high", str(condition.q_high), "--split-seed", str(cell["split_seed"]), "--matrix", cell["matrix"], "--device", args.device, "--minimum-free-space-gib", str(args.minimum_free_space_gib)]
            for name in ("window_size", "hidden_dim", "num_layers", "dropout", "cnn_kernel_size", "mlp_depth", "batch_size", "epochs", "weight_decay", "max_train_samples", "max_val_samples"):
                command.extend(["--" + name.replace("_", "-"), str(defaults[name])])
            for model in cell["models"]:
                command.extend(["--model-lr", f"{model}={cell['model_learning_rates'][model]:.17g}"])
            subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({"status": "prepared" if args.prepare_only else "completed", "cells": len(cells), "model_runs": sum(len(c["models"]) for c in cells)}))


if __name__ == "__main__":
    main()
