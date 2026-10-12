'Audit public offline tasks under the Unified38 fixed protocol.'

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.public_offline import (
    PublicOfflineProtocolConfig,
    audit_public_task_protocol,
    build_public_precursor_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--env-id", type=str, nargs="*", default=None)
    parser.add_argument("--dataset-path", type=str, nargs="*", default=None)
    parser.add_argument("--task-name", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--group-size-episodes", type=int, default=1)
    parser.add_argument("--step-q-low", type=float, default=0.75)
    parser.add_argument("--step-q-high", type=float, default=0.90)
    parser.add_argument("--outcome-q-low", type=float, default=0.75)
    parser.add_argument("--outcome-q-high", type=float, default=0.90)
    parser.add_argument("--min-class-ratio", type=float, default=0.01)
    parser.add_argument("--min-stable-tasks", type=int, default=3)
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return slug or "task"


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tasks", [])
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a list or a dict with a `tasks` list.")
    return [dict(item) for item in payload]


def _collect_task_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest:
        return _load_manifest(args.manifest)

    specs: list[dict[str, Any]] = []
    env_ids = list(args.env_id or [])
    dataset_paths = list(args.dataset_path or [])
    task_names = list(args.task_name or [])

    if env_ids:
        for env_id in env_ids:
            specs.append({"env_id": str(env_id), "task_name": str(env_id)})
        return specs

    if dataset_paths:
        if task_names and len(task_names) != len(dataset_paths):
            raise ValueError("When `--task-name` is provided with `--dataset-path`, the lengths must match.")
        for index, dataset_path in enumerate(dataset_paths):
            specs.append({
                "dataset_path": str(dataset_path),
                "task_name": str(task_names[index]) if task_names else Path(dataset_path).stem,
            })
        return specs

    raise ValueError("Provide either `--manifest`, at least one `--env-id`, or at least one `--dataset-path`.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _collect_task_specs(args)
    protocol = PublicOfflineProtocolConfig(
        gamma=float(args.gamma),
        group_size_episodes=int(args.group_size_episodes),
        step_q_low=float(args.step_q_low),
        step_q_high=float(args.step_q_high),
        outcome_q_low=float(args.outcome_q_low),
        outcome_q_high=float(args.outcome_q_high),
        min_class_ratio=float(args.min_class_ratio),
    )

    task_results: list[dict[str, Any]] = []
    for spec in specs:
        task_name = str(spec.get("task_name") or spec.get("env_id") or Path(spec["dataset_path"]).stem)
        task_dir = output_dir / _slugify(task_name)
        task_dir.mkdir(parents=True, exist_ok=True)

        dataset = build_public_precursor_dataset(
            env_id=spec.get("env_id"),
            dataset_path=spec.get("dataset_path"),
            task_name=task_name,
            protocol=protocol,
        )
        audit = audit_public_task_protocol(
            dataset,
            split_seeds=list(args.split_seeds),
            protocol=protocol,
        )
        (task_dir / "label_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        task_results.append(audit)

    stable_tasks = [item["task_name"] for item in task_results if bool(item["stable_task"])]
    summary = {
        "protocol": protocol.to_dict(),
        "task_count": len(task_results),
        "stable_task_count": len(stable_tasks),
        "stable_tasks": stable_tasks,
        "min_stable_tasks_required": int(args.min_stable_tasks),
        "gate_a_protocol_feasibility": {
            "status": "pass",
            "task_specific_overrides_used": False,
            "rule": (
                "This audit enforces a single global protocol. "
                "If later progress requires task-specific manual retuning, Gate A must be marked failed."
            ),
        },
        "gate_b_task_coverage": {
            "status": "pass" if len(stable_tasks) >= int(args.min_stable_tasks) else "fail",
            "rule": f"Require at least {int(args.min_stable_tasks)} stable tasks before advancing the Unified38 public protocol.",
        },
        "decision": (
            "advance_unified38_protocol"
            if len(stable_tasks) >= int(args.min_stable_tasks)
            else "hold_unified38_protocol"
        ),
        "tasks": task_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "decision": summary["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
