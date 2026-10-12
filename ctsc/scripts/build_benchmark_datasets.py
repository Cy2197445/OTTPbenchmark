"""Build the selected DSRL precursor files using the benchmark protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctsc.precursor.data import save_precursor_dataset
from ctsc.precursor.public_offline import PublicOfflineProtocolConfig, build_public_precursor_dataset
from ctsc.precursor.revision import file_sha256
from ctsc.scripts.run_benchmark import output_directory

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "manifests/datasets.json").read_text(encoding="utf-8"))
    by_task = {r["task_name"]: r for r in catalog["tasks"]}
    tasks = list(by_task) if args.tasks is None else list(dict.fromkeys(args.tasks))
    if not tasks or set(tasks) - set(by_task):
        raise ValueError("Unknown or empty task selection")
    output = output_directory(args.output_dir)
    for task in tasks:
        row = by_task[task]
        raw = args.raw_dir.resolve() / row["file_name"]
        if not raw.is_file() or raw.stat().st_size != row["bytes"] or file_sha256(raw) != row["sha256"]:
            raise ValueError(f"Raw dataset differs from its source manifest: {raw}")
        if (output / f"{task}.pt").exists():
            raise FileExistsError(output / f"{task}.pt")
    for task in tasks:
        raw = args.raw_dir.resolve() / by_task[task]["file_name"]
        dataset = build_public_precursor_dataset(dataset_path=raw, task_name=task,
                                                protocol=PublicOfflineProtocolConfig())
        path = save_precursor_dataset(dataset, output / f"{task}.pt")
        print(json.dumps({"task": task, "output": str(path), "samples": len(dataset["costs"])}), flush=True)


if __name__ == "__main__":
    main()
