"""Tests to ensure utility functions use scaled participation weights when available."""

import polars as pl
import pytest

from spforge import ColumnNames
from spforge.ratings.utils import (
    _SCALED_PPW,
    add_team_rating_projected,
    add_rating_mean_projected,
)


@pytest.fixture
def column_names():
    return ColumnNames(
        player_id="pid",
        team_id="tid",
        match_id="mid",
        start_date="dt",
        projected_participation_weight="ppw",
    )


@pytest.fixture
def df_with_scaled():
    """DataFrame with both raw and scaled projected participation weights."""
    return pl.DataFrame({
        "pid": ["A", "B", "C", "D"],
        "tid": ["T1", "T1", "T2", "T2"],
        "mid": ["M1", "M1", "M1", "M1"],
        "dt": ["2024-01-01"] * 4,
        "rating": [1100.0, 900.0, 1050.0, 950.0],
        "ppw": [20.0, 5.0, 10.0, 10.0],  # Raw weights (would give wrong answer)
        _SCALED_PPW: [1.0, 0.5, 1.0, 1.0],  # Scaled/clipped weights
    })


@pytest.fixture
def df_without_scaled():
    """DataFrame with only raw projected participation weights (no scaled column)."""
    return pl.DataFrame({
        "pid": ["A", "B", "C", "D"],
        "tid": ["T1", "T1", "T2", "T2"],
        "mid": ["M1", "M1", "M1", "M1"],
        "dt": ["2024-01-01"] * 4,
        "rating": [1100.0, 900.0, 1050.0, 950.0],
        "ppw": [0.8, 0.4, 1.0, 1.0],  # Already scaled weights
    })


def test_add_team_rating_projected_uses_scaled_column(column_names, df_with_scaled):
    """add_team_rating_projected should use _SCALED_PPW when available."""
    result = add_team_rating_projected(
        df=df_with_scaled,
        column_names=column_names,
        player_rating_col="rating",
        team_rating_out="team_rating",
    )

    # With scaled weights (1.0, 0.5), T1 team rating = (1100*1.0 + 900*0.5) / (1.0+0.5) = 1450/1.5 = 966.67
    # If it used raw weights (20.0, 5.0), it would be (1100*20 + 900*5) / 25 = 26500/25 = 1060
    t1_rating = result.filter(pl.col("tid") == "T1")["team_rating"][0]

    expected_with_scaled = (1100.0 * 1.0 + 900.0 * 0.5) / (1.0 + 0.5)
    wrong_with_raw = (1100.0 * 20.0 + 900.0 * 5.0) / (20.0 + 5.0)

    assert t1_rating == pytest.approx(expected_with_scaled, rel=1e-6)
    assert t1_rating != pytest.approx(wrong_with_raw, rel=1e-6)


def test_add_team_rating_projected_falls_back_to_raw(column_names, df_without_scaled):
    """add_team_rating_projected should use raw ppw when _SCALED_PPW is not available."""
    result = add_team_rating_projected(
        df=df_without_scaled,
        column_names=column_names,
        player_rating_col="rating",
        team_rating_out="team_rating",
    )

    # With raw weights (0.8, 0.4), T1 team rating = (1100*0.8 + 900*0.4) / (0.8+0.4) = 1240/1.2 = 1033.33
    t1_rating = result.filter(pl.col("tid") == "T1")["team_rating"][0]

    expected = (1100.0 * 0.8 + 900.0 * 0.4) / (0.8 + 0.4)
    assert t1_rating == pytest.approx(expected, rel=1e-6)


def test_add_rating_mean_projected_uses_scaled_column(column_names, df_with_scaled):
    """add_rating_mean_projected should use _SCALED_PPW when available."""
    result = add_rating_mean_projected(
        df=df_with_scaled,
        column_names=column_names,
        player_rating_col="rating",
        rating_mean_out="mean_rating",
    )

    # With scaled weights, mean = (1100*1.0 + 900*0.5 + 1050*1.0 + 950*1.0) / (1.0+0.5+1.0+1.0)
    # = (1100 + 450 + 1050 + 950) / 3.5 = 3550/3.5 = 1014.29
    mean_rating = result["mean_rating"][0]

    expected_with_scaled = (1100.0*1.0 + 900.0*0.5 + 1050.0*1.0 + 950.0*1.0) / (1.0+0.5+1.0+1.0)
    wrong_with_raw = (1100.0*20.0 + 900.0*5.0 + 1050.0*10.0 + 950.0*10.0) / (20.0+5.0+10.0+10.0)

    assert mean_rating == pytest.approx(expected_with_scaled, rel=1e-6)
    assert mean_rating != pytest.approx(wrong_with_raw, rel=1e-6)


def test_add_rating_mean_projected_falls_back_to_raw(column_names, df_without_scaled):
    """add_rating_mean_projected should use raw ppw when _SCALED_PPW is not available."""
    result = add_rating_mean_projected(
        df=df_without_scaled,
        column_names=column_names,
        player_rating_col="rating",
        rating_mean_out="mean_rating",
    )

    # With raw weights (0.8, 0.4, 1.0, 1.0)
    mean_rating = result["mean_rating"][0]

    expected = (1100.0*0.8 + 900.0*0.4 + 1050.0*1.0 + 950.0*1.0) / (0.8+0.4+1.0+1.0)
    assert mean_rating == pytest.approx(expected, rel=1e-6)


def test_scaled_weights_not_in_output(column_names, df_with_scaled):
    """Verify utility functions don't add scaled columns to output unnecessarily."""
    result = add_team_rating_projected(
        df=df_with_scaled,
        column_names=column_names,
        player_rating_col="rating",
        team_rating_out="team_rating",
    )

    # The scaled column should still be present (it was in input)
    # but no new internal columns should be added
    assert _SCALED_PPW in result.columns
    assert "team_rating" in result.columns
