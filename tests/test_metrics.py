from __future__ import annotations

import numpy as np
import pytest

from ctsc.precursor.metrics import binary_average_precision, binary_roc_auc, expected_calibration_error


def test_binary_metrics_are_exact_for_perfect_predictions() -> None:
    scores = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)

    assert binary_roc_auc(scores, labels) == pytest.approx(1.0)
    assert binary_average_precision(scores, labels) == pytest.approx(1.0)
    assert expected_calibration_error(scores, labels) == pytest.approx(0.0)


def test_binary_metrics_handle_no_positive_examples() -> None:
    scores = np.asarray([0.2, 0.4, 0.6], dtype=np.float64)
    labels = np.asarray([0, 0, 0], dtype=np.int64)

    assert binary_roc_auc(scores, labels) == pytest.approx(0.5)
    assert binary_average_precision(scores, labels) == pytest.approx(0.0)
