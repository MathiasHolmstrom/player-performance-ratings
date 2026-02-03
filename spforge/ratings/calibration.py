"""Calibration utilities for reference rating anchoring."""

import math
from collections.abc import Sequence


def sigmoid(coef: float, rating: float, anchor: float) -> float:
    """Compute sigmoid(coef * (rating - anchor)).

    Public helper for computing predicted performance from rating difference.
    """
    value = coef * (rating - anchor)
    return math.exp(value) / (1 + math.exp(value))


def _mean_prediction(ratings: Sequence[float], coef: float, anchor: float) -> float:
    """Compute mean prediction over all ratings."""
    if not ratings:
        raise ValueError("ratings sequence cannot be empty")
    total = sum(sigmoid(coef, r, anchor) for r in ratings)
    return total / len(ratings)


def calibrate_reference_rating(
    ratings: Sequence[float],
    coef: float,
    target_mean: float = 0.5,
    lo: float = 500.0,
    hi: float = 1500.0,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """
    Return anchor so mean(sigmoid(coef*(rating-anchor))) ~= target_mean.

    Uses bisection search. Higher anchor -> lower predictions.

    Raises ValueError if ratings is empty or target not bracketed by bounds.
    """
    if not ratings:
        raise ValueError("ratings sequence cannot be empty")

    mean_lo = _mean_prediction(ratings, coef, lo)
    mean_hi = _mean_prediction(ratings, coef, hi)

    if not (mean_hi <= target_mean <= mean_lo):
        raise ValueError(
            f"Target {target_mean} not bracketed: "
            f"mean at lo={lo} is {mean_lo:.4f}, mean at hi={hi} is {mean_hi:.4f}"
        )

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        mean_mid = _mean_prediction(ratings, coef, mid)

        if abs(mean_mid - target_mean) < tol:
            return mid

        if mean_mid > target_mean:
            lo = mid  # Need higher anchor
        else:
            hi = mid  # Need lower anchor

    return (lo + hi) / 2
