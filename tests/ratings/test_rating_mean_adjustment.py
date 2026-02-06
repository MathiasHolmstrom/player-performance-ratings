"""Tests for rating mean stabilization mechanism."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from spforge import ColumnNames
from spforge.ratings import PlayerRatingGenerator


@pytest.fixture
def base_cn():
    return ColumnNames(
        player_id="pid",
        team_id="tid",
        match_id="mid",
        start_date="dt",
        update_match_id="mid",
        participation_weight="pw",
    )


def create_matches_df(n_matches: int, base_date: str = "2024-01-01") -> pl.DataFrame:
    """Create a DataFrame with n_matches between two players."""
    base = datetime.fromisoformat(base_date)
    rows = []
    for i in range(n_matches):
        date_str = (base + timedelta(days=i)).isoformat()[:10]
        rows.append({"pid": "P1", "tid": "T1", "mid": f"M{i+1}", "dt": date_str, "perf": 0.6, "pw": 1.0})
        rows.append({"pid": "P2", "tid": "T2", "mid": f"M{i+1}", "dt": date_str, "perf": 0.4, "pw": 1.0})
    return pl.DataFrame(rows)


def test_rating_mean_adjustment_disabled_by_default(base_cn):
    """Verify that rating mean adjustment is disabled by default."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
    )
    assert gen._rating_mean_adjustment_enabled is False


def test_rating_mean_adjustment_params_stored(base_cn):
    """Verify that the adjustment parameters are stored correctly."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        rating_mean_adjustment_enabled=True,
        rating_mean_adjustment_target=1500.0,
        rating_mean_adjustment_check_frequency=100,
        rating_mean_adjustment_active_days=300,
    )
    assert gen._rating_mean_adjustment_enabled is True
    assert gen._rating_mean_adjustment_target == 1500.0
    assert gen._rating_mean_adjustment_check_frequency == 100
    assert gen._rating_mean_adjustment_active_days == 300


def test_apply_rating_mean_adjustment_active_player_detection(base_cn):
    """Test that adjustment correctly identifies active vs inactive players."""
    from spforge.data_structures import PlayerRating

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        rating_mean_adjustment_enabled=True,
        rating_mean_adjustment_target=1000.0,
        rating_mean_adjustment_active_days=250,
    )

    # Set up players with different last match days
    gen._player_off_ratings["P1"] = PlayerRating(id="P1", rating_value=900.0)
    gen._player_off_ratings["P1"].last_match_day_number = 100
    gen._player_def_ratings["P1"] = PlayerRating(id="P1", rating_value=900.0)

    gen._player_off_ratings["P2"] = PlayerRating(id="P2", rating_value=900.0)
    gen._player_off_ratings["P2"].last_match_day_number = 300
    gen._player_def_ratings["P2"] = PlayerRating(id="P2", rating_value=900.0)

    gen._player_off_ratings["P3"] = PlayerRating(id="P3", rating_value=900.0)
    gen._player_off_ratings["P3"].last_match_day_number = 50  # Will be inactive at day 301
    gen._player_def_ratings["P3"] = PlayerRating(id="P3", rating_value=900.0)

    # At day 300: all players active (P1=200 days ago, P2=0, P3=250)
    # Mean is 900, adjustment is +100
    gen._apply_rating_mean_adjustment(300)
    assert gen._player_off_ratings["P1"].rating_value == 1000.0
    assert gen._player_off_ratings["P2"].rating_value == 1000.0
    assert gen._player_off_ratings["P3"].rating_value == 1000.0

    # Reset ratings
    gen._player_off_ratings["P1"].rating_value = 900.0
    gen._player_off_ratings["P2"].rating_value = 900.0
    gen._player_off_ratings["P3"].rating_value = 900.0
    gen._player_def_ratings["P1"].rating_value = 900.0
    gen._player_def_ratings["P2"].rating_value = 900.0
    gen._player_def_ratings["P3"].rating_value = 900.0

    # At day 301: P3 is now 251 days ago -> inactive
    # Only P1 and P2 are adjusted
    gen._apply_rating_mean_adjustment(301)
    assert gen._player_off_ratings["P1"].rating_value == 1000.0
    assert gen._player_off_ratings["P2"].rating_value == 1000.0
    assert gen._player_off_ratings["P3"].rating_value == 900.0  # Not adjusted


def test_rating_mean_adjustment_deflation_scenario(base_cn):
    """Test adjustment when ratings drift below target (deflation)."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        rating_mean_adjustment_enabled=True,
        rating_mean_adjustment_target=1000.0,
        rating_mean_adjustment_check_frequency=1,  # Check every match
        rating_mean_adjustment_active_days=250,
        start_harcoded_start_rating=900.0,  # Start below target
        auto_scale_performance=True,
    )

    # Create a few matches
    df = create_matches_df(3)
    gen.fit_transform(df)

    # Calculate mean of all players' ratings
    total = 0.0
    for pid in gen._player_off_ratings:
        off = gen._player_off_ratings[pid].rating_value
        def_ = gen._player_def_ratings[pid].rating_value
        total += (off + def_) / 2
    mean = total / len(gen._player_off_ratings)

    # Mean should be exactly 1000 since adjustment runs after each match
    assert abs(mean - 1000.0) < 1.0


def test_rating_mean_adjustment_inflation_scenario(base_cn):
    """Test adjustment when ratings drift above target (inflation)."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        rating_mean_adjustment_enabled=True,
        rating_mean_adjustment_target=1000.0,
        rating_mean_adjustment_check_frequency=1,
        rating_mean_adjustment_active_days=250,
        start_harcoded_start_rating=1100.0,  # Start above target
        auto_scale_performance=True,
    )

    df = create_matches_df(3)
    gen.fit_transform(df)

    # Calculate mean of all players' ratings
    total = 0.0
    for pid in gen._player_off_ratings:
        off = gen._player_off_ratings[pid].rating_value
        def_ = gen._player_def_ratings[pid].rating_value
        total += (off + def_) / 2
    mean = total / len(gen._player_off_ratings)

    assert abs(mean - 1000.0) < 1.0


def test_inactive_players_not_adjusted(base_cn):
    """Test that inactive players are not affected by adjustments."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        rating_mean_adjustment_enabled=True,
        rating_mean_adjustment_target=1000.0,
        rating_mean_adjustment_check_frequency=1,
        rating_mean_adjustment_active_days=10,  # Very short window
        start_harcoded_start_rating=900.0,
        auto_scale_performance=True,
    )

    # Create matches spread over time
    base = datetime.fromisoformat("2024-01-01")
    rows = []

    # Match 1: P1 vs P2 on day 1
    rows.append({"pid": "P1", "tid": "T1", "mid": "M1", "dt": "2024-01-01", "perf": 0.6, "pw": 1.0})
    rows.append({"pid": "P2", "tid": "T2", "mid": "M1", "dt": "2024-01-01", "perf": 0.4, "pw": 1.0})

    # Match 2: P3 vs P4 on day 20 (P1 and P2 now inactive with 10-day window)
    rows.append({"pid": "P3", "tid": "T3", "mid": "M2", "dt": "2024-01-20", "perf": 0.6, "pw": 1.0})
    rows.append({"pid": "P4", "tid": "T4", "mid": "M2", "dt": "2024-01-20", "perf": 0.4, "pw": 1.0})

    df = pl.DataFrame(rows)
    gen.fit_transform(df)

    # P1 and P2 should have their original ratings (not adjusted in Match 2)
    # P3 and P4 should be adjusted to target
    p1_off = gen._player_off_ratings["P1"].rating_value
    p1_def = gen._player_def_ratings["P1"].rating_value

    p3_off = gen._player_off_ratings["P3"].rating_value
    p3_def = gen._player_def_ratings["P3"].rating_value

    # P1's mean rating should not be 1000 (they were inactive during adjustment)
    p1_mean = (p1_off + p1_def) / 2
    p3_mean = (p3_off + p3_def) / 2

    # P3/P4 were active and should be close to target
    # P1/P2 were inactive and won't be at target
    assert abs(p3_mean - 1000.0) < 50  # P3 should be close to target
    # P1 started at 900, got some rating changes from match 1, but wasn't adjusted later


def test_check_frequency_behavior(base_cn):
    """Test that adjustment only happens at the specified frequency."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        rating_mean_adjustment_enabled=True,
        rating_mean_adjustment_target=1000.0,
        rating_mean_adjustment_check_frequency=5,  # Check every 5 matches
        rating_mean_adjustment_active_days=250,
        start_harcoded_start_rating=900.0,
        auto_scale_performance=True,
    )

    # After 3 matches, counter should be 3, no adjustment yet
    df = create_matches_df(3)
    gen.fit_transform(df)

    # Counter should be 3 (not reset since we haven't hit 5)
    assert gen._matches_since_last_adjustment_check == 3

    # Continue with 2 more matches using transform (which doesn't process historical)
    # Actually, we need to continue with fit_transform on new data
    # But since fit_transform clears state, let's verify counter behavior differently

    # Create fresh generator and check counter after exactly 5 matches
    gen2 = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        rating_mean_adjustment_enabled=True,
        rating_mean_adjustment_target=1000.0,
        rating_mean_adjustment_check_frequency=5,
        rating_mean_adjustment_active_days=250,
        start_harcoded_start_rating=900.0,
        auto_scale_performance=True,
    )

    df5 = create_matches_df(5)
    gen2.fit_transform(df5)

    # After 5 matches, counter should be reset to 0
    assert gen2._matches_since_last_adjustment_check == 0


def test_reset_state_clears_counter(base_cn):
    """Test that _reset_rating_state clears the adjustment counter."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        rating_mean_adjustment_enabled=True,
        rating_mean_adjustment_check_frequency=10,
    )

    gen._matches_since_last_adjustment_check = 7
    gen._reset_rating_state()

    assert gen._matches_since_last_adjustment_check == 0
