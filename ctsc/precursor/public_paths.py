"""Resolve and verify raw files from a source catalog or a local download log."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ctsc.precursor.revision import file_sha256


def raw_dataset_path(task: str, manifest: Mapping[str, Any], raw_dir: Path, *, root: Path) -> Path:
    records = {str(row["task_name"]): row for row in manifest.get("tasks", [])}
    if task not in records:
        raise KeyError(f"Task not found in data manifest: {task}")
    row = records[task]
    if row.get("output_path"):
        path = Path(row["output_path"])
        if not path.is_absolute():
            path = root / path
    else:
        name = str(row["file_name"])
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"Invalid dataset file name: {name}")
        path = raw_dir / name
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset missing for {task}: {path}")
    if row.get("bytes") is not None and path.stat().st_size != int(row["bytes"]):
        raise ValueError(f"Dataset size mismatch: {path}")
    if row.get("sha256") and file_sha256(path) != row["sha256"]:
        raise ValueError(f"Dataset SHA256 mismatch: {path}")
    return path
