"""Tests to ensure PlayerRatingGenerator does not mutate input columns."""

import polars as pl
import pytest

from spforge import ColumnNames
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures


@pytest.fixture
def cn_with_projected():
    """ColumnNames with both participation_weight and projected_participation_weight."""
    return ColumnNames(
        player_id="pid",
        team_id="tid",
        match_id="mid",
        start_date="dt",
        update_match_id="mid",
        participation_weight="minutes",
        projected_participation_weight="minutes_prediction",
    )


@pytest.fixture
def fit_df():
    """Training data with minutes > 1 (will trigger auto-scaling)."""
    return pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.6, 0.4, 0.7, 0.3],
            "minutes": [30.0, 25.0, 32.0, 28.0],
            "minutes_prediction": [28.0, 24.0, 30.0, 26.0],
        }
    )


@pytest.fixture
def future_df():
    """Future prediction data with minutes > 1 (will trigger auto-scaling)."""
    return pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M2", "M2", "M2", "M2"],
            "dt": ["2024-01-02"] * 4,
            "minutes": [30.0, 25.0, 32.0, 28.0],
            "minutes_prediction": [28.0, 24.0, 30.0, 26.0],
        }
    )


def test_fit_transform_does_not_mutate_participation_weight(cn_with_projected, fit_df):
    """fit_transform should not modify the participation_weight column values."""
    # Join result with original to compare values by player_id
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn_with_projected,
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    result = gen.fit_transform(fit_df)

    # Check that each player's minutes value is preserved
    original_by_player = dict(zip(fit_df["pid"].to_list(), fit_df["minutes"].to_list()))
    result_by_player = dict(zip(result["pid"].to_list(), result["minutes"].to_list()))

    for pid, original_val in original_by_player.items():
        result_val = result_by_player[pid]
        assert result_val == original_val, (
            f"participation_weight for player {pid} was mutated. "
            f"Expected {original_val}, got {result_val}"
        )


def test_fit_transform_does_not_mutate_projected_participation_weight(cn_with_projected, fit_df):
    """fit_transform should not modify the projected_participation_weight column values."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn_with_projected,
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    result = gen.fit_transform(fit_df)

    # Check that each player's minutes_prediction value is preserved
    original_by_player = dict(zip(fit_df["pid"].to_list(), fit_df["minutes_prediction"].to_list()))
    result_by_player = dict(zip(result["pid"].to_list(), result["minutes_prediction"].to_list()))

    for pid, original_val in original_by_player.items():
        result_val = result_by_player[pid]
        assert result_val == original_val, (
            f"projected_participation_weight for player {pid} was mutated. "
            f"Expected {original_val}, got {result_val}"
        )


def test_transform_does_not_mutate_participation_weight(cn_with_projected, fit_df, future_df):
    """transform should not modify the participation_weight column values."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn_with_projected,
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    gen.fit_transform(fit_df)

    result = gen.transform(future_df)

    # Check that each player's minutes value is preserved
    original_by_player = dict(zip(future_df["pid"].to_list(), future_df["minutes"].to_list()))
    result_by_player = dict(zip(result["pid"].to_list(), result["minutes"].to_list()))

    for pid, original_val in original_by_player.items():
        result_val = result_by_player[pid]
        assert result_val == original_val, (
            f"participation_weight for player {pid} was mutated during transform. "
            f"Expected {original_val}, got {result_val}"
        )


def test_transform_does_not_mutate_projected_participation_weight(cn_with_projected, fit_df, future_df):
    """transform should not modify the projected_participation_weight column values."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn_with_projected,
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    gen.fit_transform(fit_df)

    result = gen.transform(future_df)

    # Check that each player's minutes_prediction value is preserved
    original_by_player = dict(zip(future_df["pid"].to_list(), future_df["minutes_prediction"].to_list()))
    result_by_player = dict(zip(result["pid"].to_list(), result["minutes_prediction"].to_list()))

    for pid, original_val in original_by_player.items():
        result_val = result_by_player[pid]
        assert result_val == original_val, (
            f"projected_participation_weight for player {pid} was mutated during transform. "
            f"Expected {original_val}, got {result_val}"
        )


def test_future_transform_does_not_mutate_participation_weight(cn_with_projected, fit_df, future_df):
    """future_transform should not modify the participation_weight column values."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn_with_projected,
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    gen.fit_transform(fit_df)

    original_minutes = future_df["minutes"].to_list()
    result = gen.future_transform(future_df)

    # The minutes column should have the same values as before
    result_minutes = result["minutes"].to_list()
    assert result_minutes == original_minutes, (
        f"participation_weight column was mutated during future_transform. "
        f"Expected {original_minutes}, got {result_minutes}"
    )


def test_future_transform_does_not_mutate_projected_participation_weight(cn_with_projected, fit_df, future_df):
    """future_transform should not modify the projected_participation_weight column values."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn_with_projected,
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    gen.fit_transform(fit_df)

    original_minutes_pred = future_df["minutes_prediction"].to_list()
    result = gen.future_transform(future_df)

    # The minutes_prediction column should have the same values as before
    result_minutes_pred = result["minutes_prediction"].to_list()
    assert result_minutes_pred == original_minutes_pred, (
        f"projected_participation_weight column was mutated during future_transform. "
        f"Expected {original_minutes_pred}, got {result_minutes_pred}"
    )


def test_multiple_transforms_do_not_compound_scaling(cn_with_projected, fit_df, future_df):
    """Multiple transform calls should not compound the scaling effect."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn_with_projected,
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    gen.fit_transform(fit_df)

    # Call transform multiple times
    result1 = gen.transform(future_df)
    result2 = gen.transform(result1)
    result3 = gen.transform(result2)

    # After 3 transforms, each player's values should still be the same as original
    original_by_player = dict(zip(future_df["pid"].to_list(), future_df["minutes_prediction"].to_list()))
    final_by_player = dict(zip(result3["pid"].to_list(), result3["minutes_prediction"].to_list()))

    for pid, original_val in original_by_player.items():
        final_val = final_by_player[pid]
        assert final_val == original_val, (
            f"Multiple transforms compounded the scaling for player {pid}. "
            f"Expected {original_val}, got {final_val}"
        )
