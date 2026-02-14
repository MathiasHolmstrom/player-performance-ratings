"""Tests for calibration utilities."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from spforge.data_structures import ColumnNames
from spforge.ratings import PlayerRatingGenerator
from spforge.ratings.calibration import calibrate_reference_rating, sigmoid


def test_calibration_hits_target_mean():
    """Calibrated anchor produces mean(pred) within tolerance."""
    ratings = [900.0, 950.0, 1000.0, 1050.0, 1100.0]
    coef = 0.0015
    target_mean = 0.5
    tol = 1e-4

    anchor = calibrate_reference_rating(ratings, coef, target_mean=target_mean, tol=tol)

    predictions = [sigmoid(coef, r, anchor) for r in ratings]
    actual_mean = sum(predictions) / len(predictions)

    assert abs(actual_mean - target_mean) < tol


def test_anchor_shifts_with_rating_distribution():
    """Higher ratings -> higher anchor."""
    coef = 0.0015
    target_mean = 0.5

    low_ratings = [800.0, 850.0, 900.0, 950.0, 1000.0]
    high_ratings = [1000.0, 1050.0, 1100.0, 1150.0, 1200.0]

    anchor_low = calibrate_reference_rating(low_ratings, coef, target_mean=target_mean)
    anchor_high = calibrate_reference_rating(high_ratings, coef, target_mean=target_mean)

    assert anchor_high > anchor_low


def test_monotonicity_preserved():
    """Higher rating still yields higher prediction after calibration."""
    ratings = [900.0, 950.0, 1000.0, 1050.0, 1100.0]
    coef = 0.0015

    anchor = calibrate_reference_rating(ratings, coef)

    predictions = [sigmoid(coef, r, anchor) for r in ratings]

    for i in range(len(predictions) - 1):
        assert predictions[i] < predictions[i + 1]


def test_empty_ratings_raises():
    """ValueError on empty input."""
    with pytest.raises(ValueError, match="ratings sequence cannot be empty"):
        calibrate_reference_rating([], coef=0.0015)


def test_target_not_bracketed_raises():
    """ValueError when target unreachable."""
    ratings = [1400.0, 1450.0, 1500.0]
    coef = 0.0015

    with pytest.raises(ValueError, match="Target .* not bracketed"):
        calibrate_reference_rating(
            ratings,
            coef,
            target_mean=0.1,
            lo=500.0,
            hi=1000.0,
        )


def test_different_target_means():
    """Works for targets other than 0.5."""
    ratings = [900.0, 950.0, 1000.0, 1050.0, 1100.0]
    coef = 0.0015
    tol = 1e-4

    for target_mean in [0.35, 0.45, 0.55, 0.65]:
        anchor = calibrate_reference_rating(ratings, coef, target_mean=target_mean, tol=tol)

        predictions = [sigmoid(coef, r, anchor) for r in ratings]
        actual_mean = sum(predictions) / len(predictions)

        assert abs(actual_mean - target_mean) < tol


@pytest.fixture
def calibration_cn():
    return ColumnNames(
        player_id="pid",
        team_id="tid",
        match_id="mid",
        start_date="dt",
    )


def test_ignore_opponent_calibration_applied_on_fit_transform(calibration_cn):
    """Calibration runs automatically with ignore_opponent predictor."""
    np.random.seed(42)
    n_matches = 300  # Enough to get > 100 unique players

    data = {"pid": [], "tid": [], "mid": [], "dt": [], "perf": []}

    for i in range(n_matches):
        date = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        mid = f"M{i}"
        # 4 players per match, 2 per team
        for k in range(4):
            pid = f"P{i * 4 + k}"
            tid = f"T{k // 2}"
            data["pid"].append(pid)
            data["tid"].append(tid)
            data["mid"].append(mid)
            data["dt"].append(date)
            data["perf"].append(np.random.beta(2, 2))  # Mean ~0.5

    df = pl.DataFrame(data)

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=calibration_cn,
        performance_predictor="ignore_opponent",
        auto_scale_performance=True,
    )
    gen.fit_transform(df)

    # Verify calibration happened: mean prediction should be ~0.5
    ratings = [r.rating_value for r in gen._player_off_ratings.values()]
    coef = gen._performance_predictor.coef
    anchor = gen._performance_predictor._reference_rating

    predictions = [sigmoid(coef, r, anchor) for r in ratings]
    mean_pred = sum(predictions) / len(predictions)

    assert abs(mean_pred - 0.5) < 0.02


def test_calibration_skipped_for_small_datasets(calibration_cn):
    """Calibration is skipped when fewer than 100 players."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2020-01-01"] * 4,
            "perf": [0.6, 0.4, 0.5, 0.5],
        }
    )

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=calibration_cn,
        performance_predictor="ignore_opponent",
        auto_scale_performance=False,
        start_harcoded_start_rating=1000.0,
    )
    gen.fit_transform(df)

    # Reference rating should remain at default (1000)
    assert gen._performance_predictor._reference_rating == 1000.0


def test_calibration_not_applied_to_difference_predictor(calibration_cn):
    """Difference predictor does not trigger calibration."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2020-01-01"] * 4,
            "perf": [0.6, 0.4, 0.5, 0.5],
        }
    )

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=calibration_cn,
        performance_predictor="difference",
    )
    gen.fit_transform(df)

    assert not hasattr(gen._performance_predictor, "_reference_rating")
