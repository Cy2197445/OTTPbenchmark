'Validate and group revision runs without mixing protocol identities.'

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.revision import aggregation_identity, assert_valid_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def aggregate_manifests(paths: list[Path]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    seen_attempts: set[tuple[str, str]] = set()
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert_valid_run_manifest(payload)
        attempt_key = (str(payload["experiment_id"]), str(payload["attempt_id"]))
        if attempt_key in seen_attempts:
            raise ValueError(f"Duplicate experiment/attempt pair: {attempt_key}")
        seen_attempts.add(attempt_key)
        protocol = payload["protocol"]
        key = aggregation_identity(protocol)
        groups.setdefault(key, []).append({
            "manifest": str(path),
            "experiment_id": str(payload["experiment_id"]),
            "attempt_id": str(payload["attempt_id"]),
            "task": str(payload["dataset"]["task"]),
            "protocol_hash": str(protocol["protocol_hash"]),
        })
    return {
        "schema_version": "ottp_revision_aggregate_v1",
        "run_count": int(sum(len(values) for values in groups.values())),
        "protocol_group_count": int(len(groups)),
        "groups": [
            {
                "aggregation_identity": list(key),
                "run_count": len(values),
                "runs": values,
            }
            for key, values in sorted(groups.items(), key=lambda item: repr(item[0]))
        ],
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    manifests = sorted(run_root.rglob("run_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No run_manifest.json files found under {run_root}")
    summary = aggregate_manifests(manifests)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "run_count": summary["run_count"], "protocol_group_count": summary["protocol_group_count"]}, indent=2))


if __name__ == "__main__":
    main()
