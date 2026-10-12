"""Train the fixed-1e-3 GRU and FlatMLP reference used by the target/cost ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctsc.precursor.cc1 import write_json
from ctsc.precursor.revision import assert_valid_run_manifest, file_sha256
from ctsc.scripts.run_benchmark import ROOT, output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--minimum-free-space-gib", type=float, default=10.0)
    args = parser.parse_args()
    registry = json.loads((ROOT / "manifests/experiment_registry.json").read_text(encoding="utf-8"))
    defaults = registry["mainline_fresh_rerun"]["shared_defaults"]
    known = [row["task_name"] for row in registry["tasks"]]
    tasks = known if args.tasks is None else list(dict.fromkeys(args.tasks))
    seeds = list(dict.fromkeys(args.split_seeds))
    if not tasks or set(tasks) - set(known) or not seeds or set(seeds) - {0, 1, 2}:
        raise ValueError("Unknown task or split seed")
    output = output_directory(args.output_dir)
    plan = {"tasks": tasks, "split_seeds": seeds, "models": ["gru", "flat-mlp"], "learning_rate": 1e-3,
            "defaults": defaults, "dataset_dir": str(args.dataset_dir.resolve()), "device": args.device,
            "rng_mode": "legacy_seed", "target_variant": "current_inclusive", "gamma": 0.99}
    plan_path = output / "reference_matrix.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume or not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise FileExistsError("Reference output is nonempty or its recorded settings differ")
    write_json(plan_path, plan)
    if not args.prepare_only:
        for task in tasks:
            dataset = args.dataset_dir.resolve() / f"{task}.pt"
            if not dataset.is_file():
                raise FileNotFoundError(dataset)
            for seed in seeds:
                destination = output / "task_runs" / task / "canonical" / f"seed_{seed}"
                if args.resume and (destination / "suite_summary.json").is_file():
                    summary = json.loads((destination / "suite_summary.json").read_text(encoding="utf-8"))
                    for model in plan["models"]:
                        model_dir = destination / model.replace("-", "_")
                        record = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
                        assert_valid_run_manifest(record)
                        protocol = record["protocol"]
                        if (protocol.get("legacy_compatible") is not True
                                or protocol["model_name"] != model or protocol["feature_profile"] != "full"
                                or protocol["target_variant"] != "current_inclusive" or protocol["gamma"] != 0.99
                                or protocol["split_family"] != "episode_group"
                                or protocol["quantile_pair"] != [0.75, 0.90] or protocol["outcome_quantile_pair"] != [0.75, 0.90]
                                or any(protocol[field] != seed for field in ("split_seed", "sampling_seed", "train_seed", "loader_seed"))
                                or record["training"]["learning_rate"] != 1e-3):
                            raise ValueError(f"Reference protocol differs: {model_dir}")
                        for field in ("epochs", "batch_size", "hidden_dim", "num_layers", "dropout", "weight_decay", "window_size"):
                            if record["training"][field] != defaults[field]:
                                raise ValueError(f"Reference {field} differs: {model_dir}")
                        for field in ("max_train_samples", "max_val_samples", "max_test_samples"):
                            if protocol["extra"][field] != defaults[field]:
                                raise ValueError(f"Reference {field} differs: {model_dir}")
                        if record["dataset"]["sha256"] != file_sha256(dataset) or record["protocol"]["split_seed"] != seed:
                            raise ValueError(f"Reference data/seed differs: {model_dir}")
                        for name in ("metrics.json", "precursor_model.pt"):
                            if file_sha256(model_dir / name) != record["artifacts"][name]["sha256"]:
                                raise ValueError(f"Reference artifact hash differs: {model_dir / name}")
                        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
                        if len(metrics["training"]["history"]) != defaults["epochs"] or model not in summary["models"]:
                            raise ValueError(f"Incomplete reference suite: {destination}")
                    continue
                command = [sys.executable, "-m", "ctsc.scripts.run_precursor_baseline_suite", "--dataset", str(dataset),
                           "--output-dir", str(destination), "--slice-name", "canonical", "--seed", str(seed),
                           "--device", args.device, "--models", *plan["models"], "--feature-profile", "full", "--lr", "0.001",
                           "--target-variant", "current_inclusive", "--gamma", "0.99", "--step-q-low", "0.75", "--step-q-high", "0.90",
                           "--outcome-q-low", "0.75", "--outcome-q-high", "0.90", "--minimum-free-space-gib", str(args.minimum_free_space_gib)]
                for name in ("window_size", "hidden_dim", "num_layers", "dropout", "cnn_kernel_size", "mlp_depth", "batch_size",
                             "epochs", "weight_decay", "max_train_samples", "max_val_samples", "max_test_samples"):
                    if defaults.get(name) is not None:
                        command.extend(["--" + name.replace("_", "-"), str(defaults[name])])
                subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({"status": "prepared" if args.prepare_only else "completed", "suite_count": len(tasks) * len(seeds),
                      "model_runs": 2 * len(tasks) * len(seeds)}))


if __name__ == "__main__":
    main()
