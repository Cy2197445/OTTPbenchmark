"""Run the recorded fixed-split seed study or FISOR Point Robot matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctsc.precursor.cc1 import LOCKED_MODEL_LRS, write_json
from ctsc.precursor.data import load_precursor_dataset
from ctsc.precursor.stage7 import TASKS
from ctsc.scripts.run_benchmark import output_directory
from ctsc.scripts.train_precursor_model import prepare_precursor_inputs, run_precursor_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("seed_stability", "external"), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = output_directory(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if args.experiment == "external":
        cells = [("FISOR_PointRobot", seed, seed, seed, profile) for seed in (0, 1, 2) for profile in ("full", "current_only")]
    else:
        cells = [(task, 0, seed, 0, "full") for task in TASKS for seed in range(5)]
    write_json(output / "matrix.json", {"experiment": args.experiment, "cells": cells, "learning_rates": LOCKED_MODEL_LRS})
    if not args.prepare_only:
        for task, split, train, loader, profile in cells:
            dataset = args.dataset_dir.resolve() / f"{task}.pt"
            if args.experiment == "external":
                metadata = load_precursor_dataset(dataset)["metadata"]["external_source"]
                if metadata["raw_sha256"] != "20773a5530cbda57644adac7226812cf492593a617735b4685a969131d37c470":
                    raise ValueError("FISOR source hash differs from the recorded file")
            prepared = prepare_precursor_inputs(dataset, seed=split, split_seed=split, sampling_seed=split, train_seed=train, loader_seed=loader, split_family="episode_group", ablation_group=None, ablation_mode=None, feature_profile=profile, step_q_low=0.75, step_q_high=0.90, outcome_q_low=0.75, outcome_q_high=0.90, max_train_samples=100000, max_val_samples=50000, max_test_samples=None, target_variant="current_inclusive", gamma=0.99)
            for model, lr in LOCKED_MODEL_LRS.items():
                run_precursor_model(prepared=prepared, model_name=model.replace("_", "-"), output_dir=output / task / profile / f"train_seed_{train}" / model, device=args.device, window_size=32, hidden_dim=128, num_layers=1, dropout=0.0, cnn_kernel_size=3, mlp_depth=2, transformer_num_layers=2, transformer_num_heads=4, transformer_ffn_multiplier=4, transformer_max_length=32, batch_size=256, gradient_accumulation_steps=1, epochs=12, lr=lr, weight_decay=1e-5, num_workers=0, critic_baseline_mode="dataset", experiment_id=f"{args.experiment}__{task}__{split}__{train}__{profile}__{model}", attempt_id="attempt-001", work_package=args.experiment, minimum_free_space_gib=10.0)
    print(json.dumps({"cells": len(cells), "model_runs": len(cells) * 3}))


if __name__ == "__main__":
    main()
