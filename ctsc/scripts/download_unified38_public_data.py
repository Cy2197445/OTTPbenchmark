'Download or manifest the public Unified38 datasets from the upstream DSRL card.'

from __future__ import annotations

import argparse
import json
import hashlib
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ctsc.precursor.public_source_catalog import dsrl_catalog_records
from ctsc.precursor.revision import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--manifest-output", type=str, default=None)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait-seconds", type=float, default=3.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def _download(
    url: str,
    output_path: Path,
    *,
    chunk_bytes: int,
    retries: int,
    retry_wait_seconds: float,
    timeout_seconds: float,
    expected_sha256: str = "",
    recorded_size: int = 0,
) -> dict[str, Any]:
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(int(retries) + 1):
        try:
            if expected_sha256 and output_path.is_file() and output_path.stat().st_size == recorded_size:
                if file_sha256(output_path) == expected_sha256:
                    return {"status": "cached", "output_path": str(output_path), "actual_size": recorded_size, "sha256": expected_sha256}
            with requests.get(url, stream=True, timeout=float(timeout_seconds)) as response:
                response.raise_for_status()
                expected_size = int(response.headers.get("content-length", "0") or "0")
                if not expected_sha256 and output_path.exists() and expected_size > 0 and output_path.stat().st_size == expected_size:
                    return {
                        "status": "cached",
                        "output_path": str(output_path),
                        "expected_size": expected_size,
                        "actual_size": int(output_path.stat().st_size),
                    }
                tmp_path = output_path.with_suffix(output_path.suffix + ".part")
                digest = hashlib.sha256()
                count = 0
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=int(chunk_bytes)):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)
                            count += len(chunk)
                if recorded_size and count != recorded_size:
                    raise ValueError(f"Downloaded size differs: {count} != {recorded_size}")
                if expected_sha256 and digest.hexdigest() != expected_sha256:
                    raise ValueError("Downloaded SHA256 differs from the dataset manifest")
                tmp_path.replace(output_path)
                return {
                    "status": "downloaded",
                    "output_path": str(output_path),
                    "expected_size": expected_size,
                    "actual_size": int(output_path.stat().st_size),
                    "sha256": digest.hexdigest(),
                }
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = str(exc)
            if attempt >= int(retries):
                break
            time.sleep(float(retry_wait_seconds) * float(attempt + 1))
    return {
        "status": "failed",
        "output_path": str(output_path),
        "expected_size": 0,
        "actual_size": int(output_path.stat().st_size) if output_path.exists() else 0,
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_output = Path(args.manifest_output) if args.manifest_output else output_dir / "download_log.json"
    locked = json.loads((Path(__file__).resolve().parents[2] / "manifests/datasets.json").read_text(encoding="utf-8"))
    by_task = {row["task_name"]: row for row in locked["tasks"]}
    records = dsrl_catalog_records(tuple(args.tasks) if args.tasks else None)

    results: list[dict[str, Any]] = []
    for record in records:
        url = str(record["url"])
        file_name = Path(url).name
        output_path = output_dir / file_name
        if args.manifest_only:
            payload: dict[str, Any] = {
                "status": "manifest_only",
                "output_path": str(output_path),
                "expected_size": None,
                "actual_size": int(output_path.stat().st_size) if output_path.exists() else 0,
            }
        else:
            payload = _download(
                url,
                output_path,
                chunk_bytes=int(args.chunk_bytes),
                retries=int(args.retries),
                retry_wait_seconds=float(args.retry_wait_seconds),
                timeout_seconds=float(args.timeout_seconds),
                expected_sha256=by_task[record["task_name"]]["sha256"],
                recorded_size=by_task[record["task_name"]]["bytes"],
            )
        payload.update(
            {
                "task_name": str(record["task_name"]),
                "source_group": str(record["source_group"]),
                "source_platform": str(record["source_platform"]),
                "source": str(record["source"]),
                "source_homepage": str(record["source_homepage"]),
                "license": str(record["license"]),
                "download_manifest_id": str(record["download_manifest_id"]),
                "source_url": url,
                "file_name": file_name,
            }
        )
        results.append(payload)

    failed = [row for row in results if str(row.get("status")) == "failed"]

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(
            {
                "title": "Unified38 Public Dataset Download Manifest",
                "output_dir": str(output_dir),
                "source_homepage": records[0]["source_homepage"] if records else "",
                "download_manifest_id": records[0]["download_manifest_id"] if records else "",
                "tasks": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "requested": len(results),
                "failed": len(failed),
                "manifest": str(manifest_output),
                "mode": "manifest_only" if args.manifest_only else "download",
            },
            ensure_ascii=False,
        )
    )
    if failed and not args.manifest_only:
        failed_tasks = [str(row.get("task_name")) for row in failed]
        raise SystemExit(
            json.dumps(
                {
                    "status": "download_failed",
                    "failed_tasks": failed_tasks,
                    "manifest": str(manifest_output),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
