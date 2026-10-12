"""Attempt allocation for seed-study runs."""
from __future__ import annotations
from pathlib import Path
from ctsc.scripts.run_revision_cc1_cell import _is_cuda_oom

def _attempt_dir(base: Path) -> Path:
    for number in range(1, 1000):
        candidate = base / f"attempt-{number:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many Stage 7 attempts: {base}")

