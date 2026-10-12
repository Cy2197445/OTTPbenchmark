'Run the Unified38 anti-shortcut boundary suite over tasks and split seeds.'

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import torch

from ctsc.precursor.public_source_catalog import dsrl_catalog_records
from ctsc.precursor.public_paths import raw_dataset_path
from ctsc.scripts.run_benchmark import output_directory


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--download-manifest", type=str, default="manifests/datasets.json")
    parser.add_argument("--raw-dir", type=Path, default=Path("work/raw"))
    parser.add_argument("--output-dir", type=str, default="work/anti_shortcut")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--step-q-low", type=float, default=0.75)
    parser.add_argument("--step-q-high", type=float, default=0.90)
    parser.add_argument("--outcome-q-low", type=float, default=0.75)
    parser.add_argument("--outcome-q-high", type=float, default=0.90)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_dataset_path(task_name: str, manifest: dict[str, Any], raw_dir: Path | None = None) -> str:
    return str(raw_dataset_path(task_name, manifest, raw_dir or REPO_ROOT / "work/raw", root=REPO_ROOT))


def main() -> None:
    args = parse_args()
    output_dir = output_directory(REPO_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = (
        list(dict.fromkeys(str(task) for task in args.tasks))
        if args.tasks
        else [str(row["task_name"]) for row in dsrl_catalog_records()]
    )
    split_seeds = [int(seed) for seed in args.split_seeds]
    manifest = _load_manifest(REPO_ROOT / args.download_manifest)

    for task_name in tasks:
        task_dir = output_dir / "tasks" / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        precursor_path = task_dir / "precursor.pt"
        precursor_summary = task_dir / "precursor.summary.json"
        if not precursor_path.exists():
            _run(
                [
                    sys.executable,
                    "ctsc/scripts/build_public_precursor_dataset.py",
                    "--dataset-path",
                    _task_dataset_path(task_name, manifest, REPO_ROOT / args.raw_dir),
                    "--task-name",
                    task_name,
                    "--output",
                    str(precursor_path),
                    "--summary-json",
                    str(precursor_summary),
                ],
                cwd=REPO_ROOT,
            )

        for split_seed in split_seeds:
            seed_dir = task_dir / f"seed_{int(split_seed)}"
            summary_path = seed_dir / "boundary_summary.json"
            if args.skip_existing and summary_path.exists():
                continue
            command = [
                sys.executable,
                "ctsc/scripts/run_Unified38_anti_shortcut_task.py",
                "--dataset",
                str(precursor_path),
                "--task-name",
                task_name,
                "--output-dir",
                str(seed_dir),
                "--split-seed",
                str(int(split_seed)),
                "--device",
                str(args.device),
                "--window-size",
                str(int(args.window_size)),
                "--hidden-dim",
                str(int(args.hidden_dim)),
                "--num-layers",
                str(int(args.num_layers)),
                "--dropout",
                str(float(args.dropout)),
                "--batch-size",
                str(int(args.batch_size)),
                "--epochs",
                str(int(args.epochs)),
                "--lr",
                str(float(args.lr)),
                "--weight-decay",
                str(float(args.weight_decay)),
                "--num-workers",
                str(int(args.num_workers)),
                "--step-q-low",
                str(float(args.step_q_low)),
                "--step-q-high",
                str(float(args.step_q_high)),
                "--outcome-q-low",
                str(float(args.outcome_q_low)),
                "--outcome-q-high",
                str(float(args.outcome_q_high)),
            ]
            if args.max_train_samples is not None:
                command.extend(["--max-train-samples", str(int(args.max_train_samples))])
            if args.max_val_samples is not None:
                command.extend(["--max-val-samples", str(int(args.max_val_samples))])
            if args.max_test_samples is not None:
                command.extend(["--max-test-samples", str(int(args.max_test_samples))])
            _run(command, cwd=REPO_ROOT)

    _run(
        [
            sys.executable,
            "ctsc/scripts/aggregate_Unified38_anti_shortcut.py",
            "--input-root",
            str(output_dir),
            "--output-dir",
            str(output_dir),
            "--task-names",
            *tasks,
            "--split-seeds",
            *[str(int(seed)) for seed in split_seeds],
        ],
        cwd=REPO_ROOT,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "task_count": len(tasks),
                "split_seed_count": len(split_seeds),
                "summary": str(output_dir / "Unified38_boundary_aggregate.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
