"""Descriptive fixed-split training-seed statistics."""
from __future__ import annotations
import numpy as np

def _numeric_summary(values: list[float]) -> dict[str, float]:
    observed = np.asarray(values, dtype=float)
    if observed.size < 2 or not np.isfinite(observed).all():
        raise ValueError("Stability summary requires at least two finite values")
    return {"mean": float(np.mean(observed)), "median": float(np.median(observed)),
            "sample_sd": float(np.std(observed, ddof=1)),
            "q25": float(np.quantile(observed, 0.25)),
            "q75": float(np.quantile(observed, 0.75)),
            "iqr": float(np.quantile(observed, 0.75) - np.quantile(observed, 0.25)),
            "min": float(observed.min()), "max": float(observed.max())}

