import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conformal import (  # noqa: E402
    conformal_threshold,
    coverage,
    expected_calibration_error,
    lac_scores,
    mean_set_size,
    prediction_sets,
)


def synthetic(n, sharpness, rng):
    """Calibrated-by-construction data: labels are drawn from the probabilities."""
    logits = rng.normal(size=(n, 4)) * sharpness
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(4, p=p) for p in probs])
    return probs, y


def test_threshold_uses_finite_sample_correction():
    scores = np.arange(1, 10) / 10  # n = 9, alpha = 0.1, rank = ceil(10 * 0.9) = 9
    assert conformal_threshold(scores, 0.1) == 0.9


def test_threshold_is_infinite_when_calibration_set_is_too_small():
    assert conformal_threshold(np.array([0.1, 0.2]), 0.1) == float("inf")


def test_marginal_coverage_holds_when_data_are_exchangeable():
    rng = np.random.default_rng(0)
    probs, y = synthetic(20000, 2.0, rng)
    threshold = conformal_threshold(lac_scores(probs[:10000], y[:10000]), 0.1)
    observed = coverage(prediction_sets(probs[10000:], threshold), y[10000:])
    assert abs(observed - 0.9) < 0.015


def test_coverage_falls_when_test_labels_shift():
    rng = np.random.default_rng(1)
    probs, y = synthetic(20000, 2.0, rng)
    threshold = conformal_threshold(lac_scores(probs[:10000], y[:10000]), 0.1)
    noisy = y[10000:].copy()
    flip = rng.random(len(noisy)) < 0.3
    noisy[flip] = rng.integers(0, 4, flip.sum())
    assert coverage(prediction_sets(probs[10000:], threshold), noisy) < 0.85


def test_set_size_shrinks_with_smaller_threshold():
    probs = np.array([[0.7, 0.2, 0.05, 0.05]])
    assert mean_set_size(prediction_sets(probs, 0.5)) == 1
    assert mean_set_size(prediction_sets(probs, 0.9)) == 2


def test_ece_is_near_zero_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(2)
    probs, y = synthetic(20000, 2.0, rng)
    assert expected_calibration_error(probs, y) < 0.02
    overconfident = np.tile([0.97, 0.01, 0.01, 0.01], (1000, 1))
    labels = rng.integers(0, 4, 1000)
    assert expected_calibration_error(overconfident, labels) > 0.5
