'Build a public-offline precursor dataset from a local file or DSRL env.'

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.data import save_precursor_dataset
from ctsc.precursor.public_offline import (
    PublicOfflineProtocolConfig,
    build_public_precursor_dataset,
    summarize_public_precursor_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--group-size-episodes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = PublicOfflineProtocolConfig(
        gamma=float(args.gamma),
        group_size_episodes=int(args.group_size_episodes),
    )
    dataset = build_public_precursor_dataset(
        env_id=args.env_id,
        dataset_path=args.dataset_path,
        task_name=args.task_name,
        protocol=protocol,
    )
    output_path = save_precursor_dataset(dataset, args.output)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_public_precursor_dataset(dataset)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"dataset": str(output_path), "summary_json": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
