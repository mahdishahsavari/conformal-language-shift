"""Split conformal prediction for multiple-choice questions.

Everything here is plain NumPy and has no model dependency, so it can be unit
tested without a GPU. `probs` is always an (n_questions, n_options) array and
`y` holds the index of the correct option.
"""
from __future__ import annotations

import math

import numpy as np


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAC nonconformity score: one minus the probability of the true option."""
    return 1.0 - probs[np.arange(len(y)), y]


def aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """APS nonconformity score: total probability ranked at or above the true option.

    Not used by the pilot. Implementing it answers a question the paper needs:
    does the coverage loss under language shift depend on the score function?
    """
    # TODO(Mahdi): sort each row by descending probability, take the cumulative
    # sum, and return the cumulative value at the position of the true option.
    # Trade-off to think about: LAC gives the smallest sets but tends to
    # under-cover hard questions; APS gives larger sets with more even coverage.
    raise NotImplementedError


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of the calibration scores."""
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return math.inf
    return float(np.sort(scores)[rank - 1])


def prediction_sets(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean membership matrix: an option is kept when 1 - p <= threshold."""
    return 1.0 - probs <= threshold


def coverage(sets: np.ndarray, y: np.ndarray) -> float:
    """Share of questions whose prediction set contains the correct option."""
    return float(sets[np.arange(len(y)), y].mean())


def mean_set_size(sets: np.ndarray) -> float:
    return float(sets.sum(axis=1).mean())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Top-label ECE with equal-width bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)
