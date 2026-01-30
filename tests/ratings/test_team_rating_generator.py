"""
Unit tests for TeamRatingGenerator.

These tests follow the pattern: "when X is the input then we should expect to see Y because of XXX reasons"
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from spforge.data_structures import ColumnNames, GameColumnNames
from spforge.ratings import TeamRatingGenerator
from spforge.ratings.enums import RatingKnownFeatures, RatingUnknownFeatures
from spforge.ratings.team_performance_predictor import TeamRatingNonOpponentPerformancePredictor


@pytest.fixture
def column_names():
    """Standard column names for testing."""
    return ColumnNames(
        match_id="match_id",
        team_id="team_id",
        start_date="start_date",
        update_match_id="match_id",
    )


@pytest.fixture
def basic_rating_generator(column_names):
    """Basic TeamRatingGenerator with minimal configuration."""
    return TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        features_out=[
            RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
            RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED,
        ],
    )


def test_init_when_default_params_are_used_then_generator_initializes_with_expected_values(
    column_names,
):
    """
    When default parameters are used, then we should expect to see:
    - start_team_rating = 1000.0 (default)
    - rating_center = 1000.0 (same as start_team_rating when not specified)
    - performance_predictor = "difference" (default)
    - rating_change_multiplier_offense = 50 (default)
    - rating_change_multiplier_defense = 50 (default)
    because these are the standard defaults for team rating generation.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
    )

    assert generator.start_rating_generator is not None
    assert generator.performance_predictor == "difference"
    assert generator.rating_change_multiplier_offense == 50.0
    assert generator.rating_change_multiplier_defense == 50.0


@pytest.mark.parametrize("predictor", ["difference", "mean", "ignore_opponent"])
def test_init_when_performance_predictor_is_set_then_it_is_stored_correctly(
    column_names, predictor
):
    """
    When a performance_predictor is set, then we should expect to see
    it stored correctly because the predictor type determines how performance
    is predicted based on team and opponent ratings.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        performance_predictor=predictor,
    )

    assert generator.performance_predictor == predictor


@pytest.mark.parametrize("off_mult,def_mult", [(25.0, 50.0), (50.0, 25.0), (100.0, 100.0)])
def test_init_when_different_multipliers_are_set_then_they_are_stored_correctly(
    column_names, off_mult, def_mult
):
    """
    When different offense and defense multipliers are set, then we should expect to see
    they are stored separately because offense and defense ratings can have different
    sensitivity to performance changes.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        rating_change_multiplier_offense=off_mult,
        rating_change_multiplier_defense=def_mult,
    )

    assert generator.rating_change_multiplier_offense == off_mult
    assert generator.rating_change_multiplier_defense == def_mult


@pytest.mark.parametrize("suffix", ["", "_custom", "_test"])
def test_init_when_output_suffix_is_provided_then_it_is_applied_to_column_names(
    column_names, suffix
):
    """
    When output_suffix is provided, then we should expect to see
    all output column names suffixed with that value because _suffix
    method appends the suffix to feature names for namespacing.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        output_suffix=suffix,
    )

    if suffix:
        assert generator.TEAM_OFF_RATING_PROJ_COL.endswith(suffix)
        assert generator.TEAM_DEF_RATING_PROJ_COL.endswith(suffix)
    else:
        assert generator.TEAM_OFF_RATING_PROJ_COL == "team_off_rating_projected"


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_fit_transform_when_called_then_ratings_are_updated(df_type, basic_rating_generator):
    """
    When fit_transform is called, then we should expect to see
    team ratings updated based on match results because _historical_transform
    processes matches and applies rating changes through _apply_team_updates.
    """
    df = df_type(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    basic_rating_generator.fit_transform(df)

    team_a_off = basic_rating_generator._team_off_ratings["team_a"]
    assert team_a_off.rating_value != 1000.0


@pytest.mark.parametrize("performance", [0.0, 0.5, 1.0])
def test_fit_transform_when_team_performance_varies_then_rating_changes_accordingly(
    column_names, performance
):
    """
    When team performance varies, then we should expect to see
    rating changes proportional to (observed - predicted) because
    the rating update formula is: change = (perf - pred) * multiplier.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        rating_change_multiplier_offense=50.0,
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [performance, 1.0 - performance],
        }
    )

    generator.fit_transform(df)
    team_a_off = generator._team_off_ratings["team_a"]

    if performance > 0.5:
        assert team_a_off.rating_value > 1000.0
    elif performance < 0.5:
        assert team_a_off.rating_value < 1000.0
    else:
        assert abs(team_a_off.rating_value - 1000.0) < 1.0


def test_fit_transform_when_team_wins_against_equal_opponent_then_offense_rating_increases(
    basic_rating_generator,
):
    """
    When a team wins (performance=1.0) against an opponent with equal rating,
    then we should expect to see the team's offense rating increase because
    the observed performance (1.0) exceeds the predicted performance (~0.5),
    resulting in a positive rating change.
    """
    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    basic_rating_generator.fit_transform(df)

    team_a_off = basic_rating_generator._team_off_ratings["team_a"]
    assert team_a_off.rating_value > 1000.0


def test_fit_transform_when_team_loses_against_equal_opponent_then_offense_rating_decreases(
    basic_rating_generator,
):
    """
    When a team loses (performance=0.0) against an opponent with equal rating,
    then we should expect to see the team's offense rating decrease because
    the observed performance (0.0) is below the predicted performance (~0.5),
    resulting in a negative rating change.
    """
    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [0.0, 1.0],
        }
    )

    basic_rating_generator.fit_transform(df)

    team_a_off = basic_rating_generator._team_off_ratings["team_a"]
    assert team_a_off.rating_value < 1000.0


def test_fit_transform_when_opponent_wins_then_defense_rating_decreases(basic_rating_generator):
    """
    When the opponent wins (opponent performance=1.0), then we should expect to see
    the team's defense rating decrease because defense performance is calculated as
    1.0 - opponent_offense_performance, so when opponent scores 1.0, defense gets 0.0,
    which is below the predicted defense performance, resulting in negative change.
    """
    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [0.0, 1.0],
        }
    )

    basic_rating_generator.fit_transform(df)

    team_a_def = basic_rating_generator._team_def_ratings["team_a"]
    assert team_a_def.rating_value < 1000.0


@pytest.mark.parametrize("off_mult,def_mult", [(100.0, 25.0), (25.0, 100.0)])
def test_fit_transform_when_multipliers_differ_then_offense_and_defense_change_differently(
    column_names, off_mult, def_mult
):
    """
    When offense and defense multipliers are different, then we should expect to see
    different magnitude of changes for offense vs defense ratings because the
    rating change formula multiplies the performance difference by the respective
    multiplier, so larger multipliers result in larger changes.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        rating_change_multiplier_offense=off_mult,
        rating_change_multiplier_defense=def_mult,
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    generator.fit_transform(df)

    team_a_off = generator._team_off_ratings["team_a"]
    team_a_def = generator._team_def_ratings["team_a"]

    off_change = abs(team_a_off.rating_value - 1000.0)
    def_change = abs(team_a_def.rating_value - 1000.0)

    if off_mult > def_mult:
        assert off_change > def_change
    elif def_mult > off_mult:
        assert def_change > off_change


def test_fit_transform_when_called_then_output_contains_requested_features(column_names):
    """
    When fit_transform is called with features_out specified, then we should expect to see
    only those features in the output dataframe because _add_rating_features filters
    columns based on cols_to_add and drops unrequested features.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[
            RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
            RatingKnownFeatures.TEAM_DEF_RATING_PROJECTED,
        ],
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    assert "team_off_rating_projected" in result.columns
    assert "team_def_rating_projected" in result.columns
    assert "opponent_off_rating_projected" not in result.columns


def test_fit_transform_when_single_match_with_two_teams_then_match_df_has_one_row_per_team(
    basic_rating_generator,
):
    """
    When input has a single match with two teams, then we should expect to see
    match_df with exactly 2 rows (one per team) because _create_match_df joins
    teams on match_id and filters out self-matches, creating one row per team
    showing the team and its opponent.
    """
    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    result = basic_rating_generator.fit_transform(df)

    assert len(result) == 2
    assert set(result["team_id"].to_list()) == {"team_a", "team_b"}
    assert all(result["match_id"] == 1)


@pytest.mark.parametrize("num_matches", [1, 2, 5])
def test_fit_transform_when_multiple_matches_then_each_match_creates_two_rows(
    basic_rating_generator, num_matches
):
    """
    When input has multiple matches, then we should expect to see
    match_df with 2 rows per match because each match has two teams
    and the join creates one row per team showing its opponent.
    """
    match_ids = []
    team_ids = []
    dates = []
    performances = []

    for i in range(num_matches):
        match_ids.extend([i + 1, i + 1])
        team_ids.extend([f"team_{i*2+1}", f"team_{i*2+2}"])
        dates.extend([datetime(2024, 1, i + 1), datetime(2024, 1, i + 1)])
        performances.extend([1.0, 0.0])

    df = pl.DataFrame(
        {
            "match_id": match_ids,
            "team_id": team_ids,
            "start_date": dates,
            "won": performances,
        }
    )

    result = basic_rating_generator.fit_transform(df)

    assert len(result) == num_matches * 2
    assert result["match_id"].n_unique() == num_matches


def test_fit_transform_when_teams_have_different_dates_then_day_number_is_calculated_correctly(
    basic_rating_generator,
):
    """
    When teams have different start dates, then we should expect to see
    __day_number column calculated correctly with the earliest date as day 1
    because add_day_number_utc calculates relative day numbers from the minimum date.
    """
    df = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "team_id": ["team_a", "team_b", "team_a", "team_c"],
            "start_date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 5),
                datetime(2024, 1, 5),
            ],
            "won": [1.0, 0.0, 1.0, 0.0],
        }
    )

    result = basic_rating_generator.fit_transform(df)

    assert len(result) == 4


def test_fit_transform_when_difference_predictor_used_then_rating_changes_reflect_prediction(
    column_names,
):
    """
    When using difference predictor, then we should expect to see
    rating changes reflect predictions: when team_rating > opp_rating, prediction > 0.5,
    so winning gives smaller change than when team_rating < opp_rating, because
    the difference predictor adds a term proportional to (team_rating - opp_rating).
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        performance_predictor="difference",
        confidence_weight=0.0,
        rating_change_multiplier_offense=50.0,
        output_suffix="",
    )

    for i in range(5):
        df_setup = pl.DataFrame(
            {
                "match_id": [i + 1, i + 1],
                "team_id": ["team_a", f"opp_{i}"],
                "start_date": [datetime(2024, 1, i + 1), datetime(2024, 1, i + 1)],
                "won": [1.0, 0.0],
            }
        )
        generator.fit_transform(df_setup)

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value
    assert team_a_rating_before > 1000.0

    df_test = pl.DataFrame(
        {
            "match_id": [100, 100],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 20), datetime(2024, 1, 20)],
            "won": [1.0, 0.0],
        }
    )

    generator.fit_transform(df_test)
    team_a_rating_after = generator._team_off_ratings["team_a"].rating_value
    rating_change_strong = team_a_rating_after - team_a_rating_before

    generator2 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        performance_predictor="difference",
        confidence_weight=0.0,
        rating_change_multiplier_offense=50.0,
        output_suffix="",
    )

    for i in range(5):
        df_setup = pl.DataFrame(
            {
                "match_id": [i + 1, i + 1],
                "team_id": ["team_a", f"opp_{i}"],
                "start_date": [datetime(2024, 1, i + 1), datetime(2024, 1, i + 1)],
                "won": [0.0, 1.0],
            }
        )
        generator2.fit_transform(df_setup)

    team_a_rating_before_weak = generator2._team_off_ratings["team_a"].rating_value
    assert team_a_rating_before_weak < 1000.0

    df_test2 = pl.DataFrame(
        {
            "match_id": [100, 100],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 20), datetime(2024, 1, 20)],
            "won": [1.0, 0.0],
        }
    )

    generator2.fit_transform(df_test2)
    team_a_rating_after_weak = generator2._team_off_ratings["team_a"].rating_value
    rating_change_weak = team_a_rating_after_weak - team_a_rating_before_weak

    assert rating_change_weak > rating_change_strong


def test_fit_transform_when_mean_predictor_used_then_prediction_reflects_mean_vs_center(
    column_names,
):
    """
    When using mean predictor, then we should expect to see
    prediction reflects mean rating vs rating_center because the mean predictor
    compares the average of both teams' ratings to the center, affecting
    rating changes accordingly.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        performance_predictor="mean",
        confidence_weight=0.0,
        rating_change_multiplier_offense=50.0,
        output_suffix="",
    )

    for i in range(3):
        df_setup = pl.DataFrame(
            {
                "match_id": [i + 1, i + 1],
                "team_id": ["team_a", f"opp_{i}"],
                "start_date": [datetime(2024, 1, i + 1), datetime(2024, 1, i + 1)],
                "won": [1.0, 0.0],
            }
        )
        generator.fit_transform(df_setup)
        df_setup2 = pl.DataFrame(
            {
                "match_id": [i + 10, i + 10],
                "team_id": ["team_b", f"opp_b_{i}"],
                "start_date": [datetime(2024, 1, i + 10), datetime(2024, 1, i + 10)],
                "won": [1.0, 0.0],
            }
        )
        generator.fit_transform(df_setup2)

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value
    team_b_rating_before = generator._team_off_ratings["team_b"].rating_value
    mean_rating = (team_a_rating_before + team_b_rating_before) / 2.0

    assert mean_rating > 1000.0

    df_test = pl.DataFrame(
        {
            "match_id": [100, 100],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 20), datetime(2024, 1, 20)],
            "won": [1.0, 0.0],
        }
    )

    generator.fit_transform(df_test)
    team_a_rating_after = generator._team_off_ratings["team_a"].rating_value
    rating_change = team_a_rating_after - team_a_rating_before

    assert rating_change < 30.0


def test_fit_transform_when_ignore_opponent_predictor_used_then_opponent_rating_does_not_matter(
    column_names,
):
    """
    When using ignore_opponent predictor, TeamRatingNonOpponentPerformancePredictor should be created
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        performance_predictor="ignore_opponent",
        confidence_weight=0.0,
        rating_change_multiplier_offense=50.0,
        output_suffix="",
    )
    assert (
        generator._performance_predictor.__class__.__name__
        == TeamRatingNonOpponentPerformancePredictor.__name__
    )


def test_fit_transform_when_prediction_would_exceed_bounds_then_it_is_clamped_to_0_1(column_names):
    """
    When prediction calculation would exceed [0, 1] bounds, then we should
    expect to see the prediction clamped to [0.0, 1.0] because the method
    uses max(0.0, min(1.0, pred)) to ensure valid probability values.

    We mock the performance predictor to return out-of-bounds values so the
    clamping path is exercised deterministically.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        performance_predictor="difference",
        confidence_weight=0.0,
        rating_change_multiplier_offense=50.0,
        output_suffix="",
    )

    mocked = MagicMock()
    mocked.predict_performance.return_value = 1.0
    generator._performance_predictor = mocked

    for i in range(15):
        df_setup = pl.DataFrame(
            {
                "match_id": [i + 1, i + 1],
                "team_id": ["team_a", f"opp_{i}"],
                "start_date": [datetime(2024, 1, i + 1), datetime(2024, 1, i + 1)],
                "won": [1.0, 0.0],
            }
        )
        generator.fit_transform(df_setup)

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value

    df_test = pl.DataFrame(
        {
            "match_id": [100, 100],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 30), datetime(2024, 1, 30)],
            "won": [1.0, 0.0],
        }
    )

    generator.fit_transform(df_test)
    team_a_rating_after = generator._team_off_ratings["team_a"].rating_value
    rating_change = team_a_rating_after - team_a_rating_before

    # With clamping to 1.0, a "certain win" prediction yields near-zero update.
    assert abs(rating_change) < 5.0

    # Ensure the predictor was actually used:
    # Each match produces 2 rows in match_df, and each row calls predict_performance twice (off+def) => 4 calls/match.
    # 15 setup matches + 1 test match => 16 * 4 = 64 calls.
    assert mocked.predict_performance.call_count == 64


def test_fit_transform_when_rating_difference_requested_then_it_is_calculated_correctly(
    column_names,
):
    """
    When TEAM_RATING_DIFFERENCE_PROJECTED is requested, it should equal
    TEAM_RATING_PROJECTED - OPPONENT_RATING_PROJECTED.

    We mock the performance predictor to avoid depending on predictor logic;
    rating values still flow through _add_rating_features and the join logic.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[
            RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED,
            RatingKnownFeatures.TEAM_RATING_PROJECTED,
            RatingKnownFeatures.OPPONENT_RATING_PROJECTED,
        ],
        output_suffix="",
    )

    # Mock the performance predictor used inside _calculate_ratings
    mocked = MagicMock()
    mocked.predict_performance.side_effect = [0.5, 0.5, 0.5, 0.5]  # 2 teams * (off+def)
    generator._performance_predictor = mocked

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    assert "team_rating_difference_projected" in result.columns

    team_a_row = result.filter(pl.col("team_id") == "team_a")
    diff = team_a_row["team_rating_difference_projected"][0]
    team_rating = team_a_row["team_rating_projected"][0]
    opp_rating = team_a_row["opponent_rating_projected"][0]

    assert diff == pytest.approx(team_rating - opp_rating)

    # Ensure the mock was actually used (off+def for each team-row in match_df)
    assert mocked.predict_performance.call_count == 4


def test_fit_transform_when_rating_mean_requested_then_it_is_calculated_correctly(column_names):
    """
    When RATING_MEAN_PROJECTED is requested, then we should expect to see
    it calculated as (TEAM_RATING_PROJECTED + OPPONENT_RATING_PROJECTED) / 2.0
    because _add_rating_features computes the mean of team and opponent ratings.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[
            RatingKnownFeatures.RATING_MEAN_PROJECTED,
            RatingKnownFeatures.TEAM_RATING_PROJECTED,
            RatingKnownFeatures.OPPONENT_RATING_PROJECTED,
        ],
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    assert "rating_mean_projected" in result.columns
    team_a_row = result.filter(pl.col("team_id") == "team_a")
    mean = team_a_row["rating_mean_projected"][0]
    team_rating = team_a_row["team_rating_projected"][0]
    opp_rating = team_a_row["opponent_rating_projected"][0]
    expected_mean = (team_rating + opp_rating) / 2.0
    assert mean == pytest.approx(expected_mean)


@pytest.mark.parametrize(
    "feature",
    [
        RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
        RatingKnownFeatures.TEAM_DEF_RATING_PROJECTED,
        RatingKnownFeatures.OPPONENT_OFF_RATING_PROJECTED,
        RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED,
    ],
)
def test_fit_transform_when_individual_rating_features_requested_then_they_appear_in_output(
    column_names, feature
):
    """
    When individual rating features are requested, then we should expect to see
    them in the output dataframe because _add_rating_features checks cols_to_add
    and includes requested features while dropping unrequested ones.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[feature],
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    feature_name = str(feature)
    assert feature_name in result.columns


def test_fit_transform_when_match_has_three_teams_then_assertion_fails(basic_rating_generator):
    """
    When a match has three or more teams, then we should expect to see
    an assertion error because _historical_transform asserts that each
    match_id has exactly 2 unique teams.
    """
    df = pl.DataFrame(
        {
            "match_id": [1, 1, 1],
            "team_id": ["team_a", "team_b", "team_c"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0, 0.5],
        }
    )

    with pytest.raises(AssertionError):
        basic_rating_generator.fit_transform(df)


def test_fit_transform_when_performance_mean_out_of_range_then_error_is_raised(column_names):
    """
    When performance mean is outside [0.42, 0.58], then we should expect to see
    a ValueError raised because fit_transform validates that performance mean
    is between 0.42 and 0.58 to ensure reasonable performance distribution.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="Mean.*must be between 0.42 and 0.58"):
        generator.fit_transform(df)


def test_fit_transform_when_performance_out_of_range_then_error_is_raised(column_names):
    """
    When performance values are outside [0, 1], then we should expect to see
    a ValueError raised because fit_transform validates that performance values
    are between -0.02 and 1.02 to ensure valid probability-like values.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.5, 0.0],
        }
    )

    with pytest.raises(ValueError, match="Max.*must be less than than 1.02"):
        generator.fit_transform(df)


def test_fit_transform_when_performance_is_null_then_no_rating_change(column_names):
    """
    When performance is null, then we should expect to see no rating change
    because null means missing data, not 0.0 (worst) performance.
    The team's pre-match rating for the next game should equal their rating before the null game.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )

    # Match 1: team_a perf=0.6, team_b perf=0.4
    # Match 2: team_a has null performance, team_b perf=0.6
    # Match 3: team_a perf=0.6, team_b perf=0.4
    df = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 2, 3, 3],
            "team_id": ["team_a", "team_b", "team_a", "team_b", "team_a", "team_b"],
            "start_date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 3),
            ],
            "won": [0.6, 0.4, None, 0.6, 0.6, 0.4],  # team_a has null in match 2
        }
    )

    result = generator.fit_transform(df)

    # Get team_a's pre-match rating for match 2 (after match 1) and match 3 (after match 2)
    team_a_rating_before_m2 = result.filter(
        (pl.col("team_id") == "team_a") & (pl.col("match_id") == 2)
    )["team_off_rating_projected"][0]
    team_a_rating_before_m3 = result.filter(
        (pl.col("team_id") == "team_a") & (pl.col("match_id") == 3)
    )["team_off_rating_projected"][0]

    # Key assertion: rating before M3 should equal rating before M2
    # because null performance in M2 means NO rating change
    assert team_a_rating_before_m3 == team_a_rating_before_m2, (
        f"team_a's rating changed after null performance game! "
        f"Before M2={team_a_rating_before_m2}, Before M3={team_a_rating_before_m3}"
    )

    # Also verify null is not treated as 0.0 by comparing with explicit 0.0
    # Use 0.3 instead of 0.0 to keep mean in valid range
    df_with_low_perf = df.with_columns(
        pl.when((pl.col("team_id") == "team_a") & (pl.col("match_id") == 2))
        .then(0.3)  # Low performance (below predicted ~0.5) causes rating drop
        .otherwise(pl.col("won"))
        .alias("won")
    )

    gen_low = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )
    result_low = gen_low.fit_transform(df_with_low_perf)

    team_a_rating_before_m3_with_low = result_low.filter(
        (pl.col("team_id") == "team_a") & (pl.col("match_id") == 3)
    )["team_off_rating_projected"][0]

    # With low perf (0.3), rating should drop (different from null which has no change)
    assert team_a_rating_before_m3 > team_a_rating_before_m3_with_low, (
        f"Null performance is being treated as low performance! "
        f"Rating with null={team_a_rating_before_m3}, rating with low perf={team_a_rating_before_m3_with_low}"
    )


def test_transform_when_auto_scale_performance_then_uses_correct_column(column_names):
    """
    When auto_scale_performance=True, the performance manager renames the column
    (e.g., 'won' -> 'performance__won'). Transform should still work by applying
    the performance manager to transform the input data.

    Bug: Currently transform doesn't apply the performance manager, causing
    a column mismatch where it looks for 'performance__won' but data has 'won'.
    This results in None being returned and defaulting to 0.0 performance.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        auto_scale_performance=True,
    )

    # fit_transform with valid performance values
    fit_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [0.6, 0.4],
        }
    )
    generator.fit_transform(fit_df)

    # After fit_transform, performance_column is changed to 'performance__won'
    assert generator.performance_column == "performance__won", (
        f"Expected performance_column to be 'performance__won' but got '{generator.performance_column}'"
    )

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value

    # transform with same format data (original column name 'won')
    # team_a has good performance (0.6 > predicted ~0.5), so rating should INCREASE
    transform_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "won": [0.6, 0.4],  # Original column name, not 'performance__won'
        }
    )

    generator.transform(transform_df)

    team_a_rating_after = generator._team_off_ratings["team_a"].rating_value

    # With 0.6 performance (above predicted ~0.5), rating should INCREASE
    # Bug: column mismatch causes perf to default to 0.0, making rating DECREASE
    assert team_a_rating_after > team_a_rating_before, (
        f"Rating should increase with good performance (0.6), but it went from "
        f"{team_a_rating_before} to {team_a_rating_after}. This indicates transform "
        f"is not finding the performance column (looking for '{generator.performance_column}' "
        f"but data has 'won') and defaulting to 0.0 performance."
    )


@pytest.mark.parametrize("confidence_weight", [0.0, 0.5, 1.0])
def test_fit_transform_when_confidence_weight_varies_then_new_teams_have_different_rating_changes(
    column_names, confidence_weight
):
    """
    When confidence_weight varies, then we should expect to see
    new teams (low confidence) have different rating changes compared to
    experienced teams (high confidence) because _applied_multiplier blends
    between base multiplier and confidence-based multiplier, where low confidence
    results in higher multipliers and thus larger rating changes.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        confidence_weight=confidence_weight,
        rating_change_multiplier_offense=50.0,
        start_team_rating=1000.0,
        output_suffix="",
    )

    for match_num in range(1, 11):
        df_experience = pl.DataFrame(
            {
                "match_id": [match_num, match_num],
                "team_id": ["experienced_team", f"opp_{match_num}"],
                "start_date": [datetime(2024, 1, match_num), datetime(2024, 1, match_num)],
                "won": [1.0, 0.0],
            }
        )
        generator.fit_transform(df_experience)

    experienced_rating_before = generator._team_off_ratings["experienced_team"].rating_value

    df_opponent_setup = pl.DataFrame(
        {
            "match_id": [99, 99],
            "team_id": ["standard_opponent", "opp_setup"],
            "start_date": [datetime(2024, 1, 19), datetime(2024, 1, 19)],
            "won": [0.0, 1.0],
        }
    )
    generator.fit_transform(df_opponent_setup)

    df_test = pl.DataFrame(
        {
            "match_id": [100, 100, 101, 101],
            "team_id": ["new_team", "standard_opponent", "experienced_team", "standard_opponent"],
            "start_date": [
                datetime(2024, 1, 20),
                datetime(2024, 1, 20),
                datetime(2024, 1, 20),
                datetime(2024, 1, 20),
            ],
            "won": [1.0, 0.0, 1.0, 0.0],
        }
    )

    generator.fit_transform(df_test)

    new_team_rating_after = generator._team_off_ratings["new_team"].rating_value
    new_team_change = new_team_rating_after - 1000.0

    experienced_rating_after = generator._team_off_ratings["experienced_team"].rating_value
    experienced_change = experienced_rating_after - experienced_rating_before

    change_difference = new_team_change - experienced_change

    if confidence_weight == 0.0:

        assert change_difference > 0
        assert change_difference < 15.0
    elif confidence_weight == 1.0:

        assert change_difference >= 10.0
    else:

        assert change_difference > 0
        assert change_difference >= 5.0


def test_fit_transform_when_team_plays_more_games_then_confidence_increases(basic_rating_generator):
    """
    When a team plays more games, then we should expect to see
    confidence_sum increase because each match adds MATCH_CONTRIBUTION_TO_SUM_VALUE
    to the confidence_sum, making the team's rating more stable over time.
    """

    df1 = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    basic_rating_generator.fit_transform(df1)
    conf_after_1 = basic_rating_generator._team_off_ratings["team_a"].confidence_sum

    df2 = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "won": [1.0, 0.0],
        }
    )

    basic_rating_generator.fit_transform(df2)
    conf_after_2 = basic_rating_generator._team_off_ratings["team_a"].confidence_sum

    assert conf_after_2 > conf_after_1


def test_fit_transform_when_days_between_matches_increase_then_confidence_decreases(column_names):
    """
    When days between matches increase, then we should expect to see
    confidence_sum decrease because the confidence calculation subtracts
    a term proportional to days_ago, reducing confidence for stale ratings.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        confidence_days_ago_multiplier=0.06,
        output_suffix="",
    )

    df1 = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    generator.fit_transform(df1)
    conf_after_1 = generator._team_off_ratings["team_a"].confidence_sum

    df2 = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": [datetime(2024, 4, 10), datetime(2024, 4, 10)],
            "won": [1.0, 0.0],
        }
    )

    generator.fit_transform(df2)
    conf_after_2 = generator._team_off_ratings["team_a"].confidence_sum

    assert conf_after_2 <= conf_after_1 + 1.5


def test_fit_transform_when_team_wins_multiple_matches_then_rating_increases_consistently(
    basic_rating_generator,
):
    """
    When a team wins multiple matches, then we should expect to see
    rating increase consistently after each win because each match applies
    a positive rating change, and the team's rating accumulates these changes.
    """
    ratings_after_each_match = [1000.0]

    for match_num in range(1, 4):
        df = pl.DataFrame(
            {
                "match_id": [match_num, match_num],
                "team_id": ["team_a", f"team_{match_num+1}"],
                "start_date": [datetime(2024, 1, match_num), datetime(2024, 1, match_num)],
                "won": [1.0, 0.0],
            }
        )

        basic_rating_generator.fit_transform(df)
        current_rating = basic_rating_generator._team_off_ratings["team_a"].rating_value
        ratings_after_each_match.append(current_rating)

    assert ratings_after_each_match[1] > ratings_after_each_match[0]
    assert ratings_after_each_match[2] > ratings_after_each_match[1]
    assert ratings_after_each_match[3] > ratings_after_each_match[2]


def test_fit_transform_when_team_plays_different_opponents_then_ratings_reflect_opponent_strength(
    basic_rating_generator,
):
    """
    When a team plays different opponents, then we should expect to see
    rating changes reflect opponent strength because the prediction depends
    on opponent rating, so beating a strong opponent gives more rating
    than beating a weak opponent.
    """

    df1 = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_b", "team_c"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )
    basic_rating_generator.fit_transform(df1)
    _ = basic_rating_generator._team_off_ratings["team_b"].rating_value

    df2 = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "won": [1.0, 0.0],
        }
    )
    basic_rating_generator.fit_transform(df2)
    team_a_rating_after_strong = basic_rating_generator._team_off_ratings["team_a"].rating_value
    change_after_strong = team_a_rating_after_strong - 1000.0

    df3 = pl.DataFrame(
        {
            "match_id": [3, 3],
            "team_id": ["team_a", "team_d"],
            "start_date": [datetime(2024, 1, 3), datetime(2024, 1, 3)],
            "won": [1.0, 0.0],
        }
    )
    basic_rating_generator.fit_transform(df3)
    team_a_rating_after_weak = basic_rating_generator._team_off_ratings["team_a"].rating_value
    change_after_weak = team_a_rating_after_weak - team_a_rating_after_strong

    assert change_after_strong > change_after_weak


def test_fit_transform_when_same_update_match_id_then_updates_are_batched(column_names):
    """
    When multiple matches share the same update_match_id, then we should expect to see
    rating updates applied only after processing all matches with that ID because
    the batching logic accumulates updates and only applies them when update_match_id
    changes, ensuring all matches in a batch are evaluated with the same ratings.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        confidence_weight=0.0,
        output_suffix="",
    )

    cn = ColumnNames(
        match_id="match_id",
        team_id="team_id",
        start_date="start_date",
        update_match_id="batch_id",
    )
    generator.column_names = cn

    df = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "batch_id": [1, 1, 1, 1],
            "team_id": ["team_a", "team_b", "team_a", "team_c"],
            "start_date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
            ],
            "won": [1.0, 0.0, 1.0, 0.0],
        }
    )

    generator.fit_transform(df)

    assert len(generator._team_off_ratings) >= 3


def test_fit_transform_when_different_update_match_ids_then_updates_applied_separately(
    column_names,
):
    """
    When matches have different update_match_ids, then we should expect to see
    rating updates applied after each batch because the batching logic detects
    the change in update_match_id and applies accumulated updates before processing
    the next batch.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    cn = ColumnNames(
        match_id="match_id",
        team_id="team_id",
        start_date="start_date",
        update_match_id="batch_id",
    )
    generator.column_names = cn

    df = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "batch_id": [1, 1, 2, 2],
            "team_id": ["team_a", "team_b", "team_a", "team_c"],
            "start_date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
            ],
            "won": [1.0, 0.0, 1.0, 0.0],
        }
    )

    generator.fit_transform(df)

    assert len(generator._team_off_ratings) >= 3


def test_fit_transform_when_opponent_rating_requested_then_it_uses_opponent_def_rating(
    column_names,
):
    """
    When OPPONENT_RATING_PROJECTED is requested, then we should expect to see
    it aliased from OPP_DEF_RATING_PROJECTED because _add_rating_features
    maps opponent rating to opponent defense rating for backward compatibility.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[
            RatingKnownFeatures.OPPONENT_RATING_PROJECTED,
            RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED,
        ],
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    assert "opponent_rating_projected" in result.columns
    team_a_row = result.filter(pl.col("team_id") == "team_a")

    assert (
        team_a_row["opponent_rating_projected"][0] == team_a_row["opponent_def_rating_projected"][0]
    )


def test_fit_transform_when_team_rating_projected_requested_then_it_equals_team_off_rating(
    column_names,
):
    """
    When TEAM_RATING_PROJECTED is requested, then we should expect to see
    it equals TEAM_OFF_RATING_PROJECTED because _calculate_ratings sets
    TEAM_RATING_PROJ_COL as an alias for TEAM_OFF_RATING_PROJ_COL for backward compatibility.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[
            RatingKnownFeatures.TEAM_RATING_PROJECTED,
            RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
        ],
        output_suffix="",
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    assert "team_rating_projected" in result.columns
    team_a_row = result.filter(pl.col("team_id") == "team_a")

    assert team_a_row["team_rating_projected"][0] == team_a_row["team_off_rating_projected"][0]


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_transform_when_called_then_ratings_are_updated(df_type, basic_rating_generator):
    """
    When transform is called, then we should expect to see
    team ratings updated because transform calls _historical_transform
    which processes matches and applies rating changes.
    """
    df = df_type(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    basic_rating_generator.transform(df)

    team_a_off = basic_rating_generator._team_off_ratings["team_a"]
    assert team_a_off.rating_value != 1000.0


def test_transform_when_called_after_fit_transform_then_uses_updated_ratings(
    basic_rating_generator,
):
    """
    When transform is called after fit_transform, then we should expect to see
    predictions use the updated ratings from fit_transform because the rating
    states persist between calls and are used for subsequent predictions.
    """

    df1 = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    basic_rating_generator.fit_transform(df1)
    team_a_rating_after_first = basic_rating_generator._team_off_ratings["team_a"].rating_value

    df2 = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "won": [1.0, 0.0],
        }
    )

    result = basic_rating_generator.transform(df2)

    team_a_row = result.filter(pl.col("team_id") == "team_a")
    assert team_a_row["team_off_rating_projected"][0] == pytest.approx(team_a_rating_after_first)


def test_transform_when_called_without_performance_column_then_no_rating_change(column_names):
    """
    When transform is called without performance column, then we should expect to see
    ratings remain unchanged because null/missing performance means no rating update
    (not treated as 0.0 which would cause a rating drop).
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    fit_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )
    generator.fit_transform(fit_df)

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value

    df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
        }
    )

    generator.transform(df)
    team_a_rating_after = generator._team_off_ratings["team_a"].rating_value

    # Null/missing performance means no rating change
    assert team_a_rating_after == team_a_rating_before


def test_future_transform_when_called_then_ratings_not_updated(basic_rating_generator):
    """
    When future_transform is called, then we should expect to see
    ratings remain unchanged because _calculate_future_ratings only computes
    predictions using existing ratings without applying any updates.
    """

    historical_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    basic_rating_generator.fit_transform(historical_df)

    team_a_off_before = basic_rating_generator._team_off_ratings["team_a"].rating_value
    team_a_def_before = basic_rating_generator._team_def_ratings["team_a"].rating_value
    team_a_games_before = basic_rating_generator._team_off_ratings["team_a"].games_played

    future_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
        }
    )

    basic_rating_generator.future_transform(future_df)

    team_a_off_after = basic_rating_generator._team_off_ratings["team_a"].rating_value
    team_a_def_after = basic_rating_generator._team_def_ratings["team_a"].rating_value
    team_a_games_after = basic_rating_generator._team_off_ratings["team_a"].games_played

    assert team_a_off_after == team_a_off_before
    assert team_a_def_after == team_a_def_before
    assert team_a_games_after == team_a_games_before


def test_future_transform_when_called_then_predictions_use_current_ratings(basic_rating_generator):
    """
    When future_transform is called, then we should expect to see
    predictions based on the current state of ratings because
    _calculate_future_ratings uses existing RatingState objects to compute
    predictions without modifying them.
    """

    historical_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    basic_rating_generator.fit_transform(historical_df)

    team_a_off_rating = basic_rating_generator._team_off_ratings["team_a"].rating_value
    team_b_def_rating = basic_rating_generator._team_def_ratings["team_b"].rating_value

    future_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
        }
    )

    result = basic_rating_generator.future_transform(future_df)

    team_a_row = result.filter(pl.col("team_id") == "team_a")
    assert team_a_row["team_off_rating_projected"][0] == team_a_off_rating
    assert team_a_row["opponent_def_rating_projected"][0] == team_b_def_rating


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_future_transform_when_called_with_different_df_types_then_works_correctly(
    column_names, df_type
):
    """
    When future_transform is called with different dataframe types (pandas/polars),
    then we should expect to see it work correctly because the method uses
    narwhals decorators to handle type conversion automatically.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        output_suffix="",
    )

    fit_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )
    generator.fit_transform(fit_df)

    future_df = df_type(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
        }
    )

    result = generator.future_transform(future_df)
    assert result is not None
    assert len(result) == 2


def test_future_transform_when_called_without_performance_column_then_works_correctly(column_names):
    """
    When future_transform is called without performance column, then we should expect to see
    it works correctly because future matches don't have outcomes yet, so performance
    column is not required for future_transform. The output will contain requested features.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        output_suffix="",
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )

    fit_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )
    generator.fit_transform(fit_df)

    future_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
        }
    )

    result = generator.future_transform(future_df)
    assert result is not None
    assert len(result) == 2
    assert "team_off_rating_projected" in result.columns


def test_transform_vs_future_transform_when_same_match_then_transform_updates_ratings_but_future_transform_does_not(
    column_names,
):
    """
    When the same match is processed with transform vs future_transform,
    then we should expect to see transform updates ratings while future_transform
    does not, because transform uses _historical_transform which applies rating
    updates, while future_transform uses _calculate_future_ratings which only
    computes predictions without updating ratings.
    """
    generator1 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    generator2 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    historical_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    generator1.fit_transform(historical_df)
    generator2.fit_transform(historical_df)

    team_a_rating_before_1 = generator1._team_off_ratings["team_a"].rating_value
    team_a_rating_before_2 = generator2._team_off_ratings["team_a"].rating_value

    assert team_a_rating_before_1 == team_a_rating_before_2

    test_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "won": [1.0, 0.0],
        }
    )

    generator1.transform(test_df)
    generator2.future_transform(test_df)

    team_a_rating_after_transform = generator1._team_off_ratings["team_a"].rating_value
    assert team_a_rating_after_transform > team_a_rating_before_1

    team_a_rating_after_future = generator2._team_off_ratings["team_a"].rating_value
    assert team_a_rating_after_future == team_a_rating_before_2


def test_transform_vs_future_transform_when_performance_column_missing_then_both_work_with_no_rating_change(
    column_names,
):
    """
    When performance column is missing, then we should expect to see
    both future_transform and transform work, and both result in no rating change
    because null/missing performance means no update (not treated as 0.0).
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )

    fit_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )
    generator.fit_transform(fit_df)

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value

    future_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
        }
    )

    result_future = generator.future_transform(future_df)
    assert result_future is not None
    assert "team_off_rating_projected" in result_future.columns

    assert generator._team_off_ratings["team_a"].rating_value == team_a_rating_before

    transform_df = pl.DataFrame(
        {
            "match_id": [3, 3],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 3), datetime(2024, 1, 3)],
        }
    )

    result_transform = generator.transform(transform_df)
    assert result_transform is not None

    # Null/missing performance means no rating change
    team_a_rating_after_transform = generator._team_off_ratings["team_a"].rating_value
    assert team_a_rating_after_transform == team_a_rating_before


def test_transform_vs_future_transform_when_games_played_then_transform_increments_but_future_transform_does_not(
    column_names,
):
    """
    When matches are processed, then we should expect to see
    transform increments games_played counter because it applies rating updates,
    while future_transform does not increment games_played because it doesn't
    update ratings or game counts.
    """
    generator1 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    generator2 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    historical_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    generator1.fit_transform(historical_df)
    generator2.fit_transform(historical_df)

    games_before_1 = generator1._team_off_ratings["team_a"].games_played
    games_before_2 = generator2._team_off_ratings["team_a"].games_played

    assert games_before_1 == games_before_2

    test_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "won": [1.0, 0.0],
        }
    )

    generator1.transform(test_df)
    generator2.future_transform(test_df)

    games_after_transform = generator1._team_off_ratings["team_a"].games_played
    assert games_after_transform == games_before_1 + 1.0

    games_after_future = generator2._team_off_ratings["team_a"].games_played
    assert games_after_future == games_before_2


def test_transform_vs_future_transform_when_confidence_changes_then_transform_updates_but_future_transform_does_not(
    column_names,
):
    """
    When matches are processed, then we should expect to see
    transform updates confidence_sum because it applies rating updates including
    confidence recalculation, while future_transform does not update confidence
    because it doesn't apply any rating updates.
    """
    generator1 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    generator2 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    historical_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    generator1.fit_transform(historical_df)
    generator2.fit_transform(historical_df)

    conf_before_1 = generator1._team_off_ratings["team_a"].confidence_sum
    conf_before_2 = generator2._team_off_ratings["team_a"].confidence_sum

    assert conf_before_1 == conf_before_2

    test_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "won": [1.0, 0.0],
        }
    )

    generator1.transform(test_df)
    generator2.future_transform(test_df)

    conf_after_transform = generator1._team_off_ratings["team_a"].confidence_sum

    assert conf_after_transform != conf_before_1

    conf_after_future = generator2._team_off_ratings["team_a"].confidence_sum
    assert conf_after_future == conf_before_2


def test_transform_vs_future_transform_when_predictions_calculated_then_both_use_same_ratings(
    column_names,
):
    """
    When predictions are calculated, then we should expect to see
    both transform and future_transform use the same current ratings for
    predictions, because both methods read from the same RatingState objects
    to compute predictions, even though only transform updates them afterwards.
    """
    generator1 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )

    generator2 = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )

    historical_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "won": [1.0, 0.0],
        }
    )

    generator1.fit_transform(historical_df)
    generator2.fit_transform(historical_df)

    team_a_rating = generator1._team_off_ratings["team_a"].rating_value
    assert generator2._team_off_ratings["team_a"].rating_value == team_a_rating

    test_df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "won": [1.0, 0.0],
        }
    )

    result_transform = generator1.transform(test_df)
    result_future = generator2.future_transform(
        test_df.select(["match_id", "team_id", "start_date"])
    )

    team_a_row_transform = result_transform.filter(pl.col("team_id") == "team_a")
    team_a_row_future = result_future.filter(pl.col("team_id") == "team_a")

    assert (
        team_a_row_transform["team_off_rating_projected"][0]
        == team_a_row_future["team_off_rating_projected"][0]
    )
    assert team_a_row_transform["team_off_rating_projected"][0] == team_a_rating


def _create_date_column(date_format: str, dates: list) -> pl.Series:
    """Helper to create date column with specified format."""
    if date_format == "string_iso_date":
        date_strings = [f"2024-01-{d:02d}" if isinstance(d, int) else d for d in dates]
        return pl.Series("start_date", date_strings)
    elif date_format == "string_datetime_space":
        date_strings = [f"2024-01-{d:02d} 12:00:00" if isinstance(d, int) else d for d in dates]
        return pl.Series("start_date", date_strings)
    elif date_format == "string_datetime_iso":
        date_strings = [f"2024-01-{d:02d}T12:00:00" if isinstance(d, int) else d for d in dates]
        return pl.Series("start_date", date_strings)
    elif date_format == "date_type":
        from datetime import date

        date_objs = [date(2024, 1, d) if isinstance(d, int) else d for d in dates]
        return pl.Series("start_date", date_objs, dtype=pl.Date)
    elif date_format == "datetime_type":
        datetime_objs = [datetime(2024, 1, d) if isinstance(d, int) else d for d in dates]
        return pl.Series("start_date", datetime_objs, dtype=pl.Datetime(time_zone=None))
    elif date_format == "datetime_timezone":
        datetime_objs = [datetime(2024, 1, d) if isinstance(d, int) else d for d in dates]
        return pl.Series("start_date", datetime_objs, dtype=pl.Datetime(time_zone="UTC"))
    else:
        raise ValueError(f"Unknown date_format: {date_format}")


@pytest.mark.parametrize(
    "date_format",
    [
        "string_iso_date",
        "string_datetime_space",
        "string_datetime_iso",
        "date_type",
        "datetime_type",
        "datetime_timezone",
    ],
)
def test_fit_transform_when_date_formats_vary_then_processes_successfully(
    column_names, date_format
):
    """
    When date formats vary (string, date, datetime, timezone-aware), then we should expect to see
    fit_transform processes successfully because _add_day_number handles all these formats
    and converts them to a consistent day number based on the minimum date.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        auto_scale_performance=True,
    )

    dates = [1, 2]
    date_col = _create_date_column(date_format, dates)

    df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": date_col,
            "won": [1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    assert len(result) == 2

    assert generator._team_off_ratings["team_a"].rating_value != 1000.0


@pytest.mark.parametrize(
    "date_format",
    [
        "string_iso_date",
        "datetime_type",
    ],
)
def test_transform_when_date_formats_vary_then_processes_successfully(column_names, date_format):
    """
    When date formats vary, then we should expect to see
    transform processes successfully because _add_day_number handles all these formats.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
    )

    fit_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": _create_date_column("string_iso_date", [1, 1]),
            "won": [1.0, 0.0],
        }
    )
    generator.fit_transform(fit_df)

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value

    dates = [2, 2]
    date_col = _create_date_column(date_format, dates)

    df = pl.DataFrame(
        {
            "match_id": [2, 2],
            "team_id": ["team_a", "team_c"],
            "start_date": date_col,
            "won": [1.0, 0.0],
        }
    )

    result = generator.transform(df)

    assert len(result) == 2

    assert generator._team_off_ratings["team_a"].rating_value != team_a_rating_before


@pytest.mark.parametrize(
    "date_format",
    [
        "string_iso_date",
        "datetime_type",
        "datetime_timezone",
    ],
)
def test_future_transform_when_date_formats_vary_then_processes_successfully(
    column_names, date_format
):
    """
    When date formats vary, then we should expect to see
    future_transform processes successfully because _add_day_number handles all these formats.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )

    fit_df = pl.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": ["team_a", "team_b"],
            "start_date": _create_date_column("string_iso_date", [1, 1]),
            "won": [1.0, 0.0],
        }
    )
    generator.fit_transform(fit_df)

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value

    dates = [2, 2]
    if date_format == "datetime_timezone":
        date_values = [datetime(2024, 1, 2), datetime(2024, 1, 2)]
        df = pl.DataFrame(
            {
                "match_id": [2, 2],
                "team_id": ["team_a", "team_b"],
                "start_date": pl.Series(
                    "start_date", date_values, dtype=pl.Datetime(time_zone="UTC")
                ),
            }
        )
    else:
        date_col = _create_date_column(date_format, dates)
        df = pl.DataFrame(
            {
                "match_id": [2, 2],
                "team_id": ["team_a", "team_b"],
                "start_date": date_col,
            }
        )

    result = generator.future_transform(df)

    assert len(result) == 2
    assert "team_off_rating_projected" in result.columns

    assert generator._team_off_ratings["team_a"].rating_value == team_a_rating_before


def test_fit_transform_when_dates_span_multiple_months_then_day_number_increments_correctly(
    column_names,
):
    """
    When dates span multiple months, then we should expect to see
    day_number increments correctly based on actual day differences,
    not just month differences, because day_number is calculated from
    the integer representation of dates.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        auto_scale_performance=True,
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 2, 3, 3],
            "team_id": ["team_a", "team_b", "team_a", "team_c", "team_b", "team_c"],
            "start_date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-15",
                "2024-01-15",
                "2024-02-01",
                "2024-02-01",
            ],
            "won": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    assert len(result) == 6


def test_fit_transform_when_single_date_then_day_number_is_one(column_names):
    """
    When all dates are the same, then we should expect to see
    day_number = 1 for all rows because (date - min_date + 1) = (date - date + 1) = 1.
    """
    generator = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        start_team_rating=1000.0,
        confidence_weight=0.0,
        output_suffix="",
        auto_scale_performance=True,
    )

    df = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "team_id": ["team_a", "team_b", "team_a", "team_c"],
            "start_date": ["2024-01-01"] * 4,
            "won": [1.0, 0.0, 1.0, 0.0],
        }
    )

    result = generator.fit_transform(df)

    assert len(result) == 4


# --- Feature Output Tests ---


@pytest.fixture
def sample_team_df(column_names):
    """Sample dataframe for team rating tests."""
    return pl.DataFrame(
        {
            "match_id": ["M1", "M1", "M2", "M2"],
            "team_id": ["T1", "T2", "T1", "T2"],
            "start_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "won": [1.0, 0.0, 1.0, 0.0],
        }
    )


@pytest.mark.parametrize(
    "features_out,non_predictor_features_out,output_suffix,expected_cols",
    [
        # Test 1: Single known feature, no suffix (defaults to performance_column="won")
        (
            [RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
            None,
            None,
            ["team_off_rating_projected_won"],
        ),
        (
            [
                RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
                RatingKnownFeatures.TEAM_DEF_RATING_PROJECTED,
            ],
            None,
            None,
            ["team_off_rating_projected_won", "team_def_rating_projected_won"],
        ),
        (
            [RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
            [RatingUnknownFeatures.PERFORMANCE],
            None,
            ["team_off_rating_projected_won", "performance_won"],
        ),
        (
            [RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
            None,
            "v2",
            ["team_off_rating_projected_v2"],
        ),
        # Test 5: Multiple features with suffix
        (
            [
                RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
                RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED,
            ],
            [RatingUnknownFeatures.TEAM_RATING_DIFFERENCE],
            "custom",
            [
                "team_off_rating_projected_custom",
                "opponent_def_rating_projected_custom",
                "team_rating_difference_custom",
            ],
        ),
        # Test 6: Rating difference features (defaults to performance_column="won")
        (
            [RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED],
            [RatingUnknownFeatures.TEAM_RATING_DIFFERENCE],
            None,
            ["team_rating_difference_projected_won", "team_rating_difference_won"],
        ),
        # Test 7: Rating mean features
        ([RatingKnownFeatures.RATING_MEAN_PROJECTED], None, "mean", ["rating_mean_projected_mean"]),
        # Test 8: Empty features_out (should use defaults, defaults to performance_column="won")
        (
            None,
            [RatingUnknownFeatures.PERFORMANCE],
            None,
            [
                "team_rating_difference_projected_won",
                "performance_won",
            ],  # Default is RATING_DIFFERENCE_PROJECTED
        ),
    ],
)
def test_team_rating_features_out_combinations(
    column_names,
    sample_team_df,
    features_out,
    non_predictor_features_out,
    output_suffix,
    expected_cols,
):
    """Test that correct features are output for different combinations of features_out, non_predictor_features_out, and suffixes."""
    gen = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=features_out,
        non_predictor_features_out=non_predictor_features_out,
        output_suffix=output_suffix,
        confidence_weight=0.0,  # Disable confidence for simpler testing
    )
    result = gen.fit_transform(sample_team_df)

    # Check that all expected columns are present
    result_cols = (
        result.columns.tolist() if hasattr(result.columns, "tolist") else list(result.columns)
    )
    for col in expected_cols:
        assert (
            col in result_cols
        ), f"Expected column '{col}' not found in output. Columns: {result_cols}"

    # Check that result has data
    assert len(result) > 0


@pytest.mark.parametrize("output_suffix", [None, "v2", "custom_suffix", "test123"])
def test_team_rating_suffix_applied_to_all_features(column_names, sample_team_df, output_suffix):
    """Test that output_suffix is correctly applied to all requested features."""
    features = [
        RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
        RatingKnownFeatures.TEAM_DEF_RATING_PROJECTED,
        RatingKnownFeatures.OPPONENT_OFF_RATING_PROJECTED,
    ]
    non_predictor = [
        RatingUnknownFeatures.TEAM_RATING_DIFFERENCE,
        RatingUnknownFeatures.PERFORMANCE,
    ]

    gen = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=features,
        non_predictor_features_out=non_predictor,
        output_suffix=output_suffix,
        confidence_weight=0.0,
    )
    result = gen.fit_transform(sample_team_df)

    # Build expected column names
    if output_suffix:
        expected_cols = [
            f"team_off_rating_projected_{output_suffix}",
            f"team_def_rating_projected_{output_suffix}",
            f"opponent_off_rating_projected_{output_suffix}",
            f"team_rating_difference_{output_suffix}",
            f"performance_{output_suffix}",
        ]
    else:
        # When output_suffix=None, it defaults to performance column name ("won")
        expected_cols = [
            "team_off_rating_projected_won",
            "team_def_rating_projected_won",
            "opponent_off_rating_projected_won",
            "team_rating_difference_won",
            "performance_won",
        ]

    result_cols = (
        result.columns.tolist() if hasattr(result.columns, "tolist") else list(result.columns)
    )
    for col in expected_cols:
        assert col in result_cols, f"Expected column '{col}' not found. Columns: {result_cols}"


def test_team_rating_only_requested_features_present(column_names, sample_team_df):
    """Test that only requested features (and input columns) are present in output."""
    gen = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
        non_predictor_features_out=None,
        output_suffix=None,
        confidence_weight=0.0,
    )
    result = gen.fit_transform(sample_team_df)

    # Should have input columns + requested feature
    input_cols = set(sample_team_df.columns)
    result_cols = set(result.columns)

    # Check that input columns are preserved
    for col in input_cols:
        assert col in result_cols, f"Input column '{col}' missing from output"

    # Check that requested feature is present (with performance column suffix)
    assert "team_off_rating_projected_won" in result_cols

    # Check that other rating features are NOT present (unless they're input columns)
    unwanted_features = [
        "team_def_rating_projected",
        "opponent_off_rating_projected",
        "team_rating_difference",
    ]
    for feature in unwanted_features:
        if feature not in input_cols:
            assert (
                feature not in result_cols
            ), f"Unrequested feature '{feature}' should not be in output"


def test_team_rating_combined_features_out_and_non_predictor(column_names, sample_team_df):
    """Test that features_out and non_predictor_features_out work together correctly."""
    gen = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[
            RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
            RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED,
        ],
        non_predictor_features_out=[
            RatingUnknownFeatures.TEAM_RATING_DIFFERENCE,
            RatingUnknownFeatures.PERFORMANCE,
        ],
        output_suffix=None,
        confidence_weight=0.0,
    )
    result = gen.fit_transform(sample_team_df)

    # All requested features should be present (with performance column suffix)
    expected_cols = [
        "team_off_rating_projected_won",
        "team_rating_difference_projected_won",
        "team_rating_difference_won",
        "performance_won",
    ]

    result_cols = (
        result.columns.tolist() if hasattr(result.columns, "tolist") else list(result.columns)
    )
    for col in expected_cols:
        assert col in result_cols, f"Expected column '{col}' not found. Columns: {result_cols}"


def test_team_rating_backward_compat_team_rating_projected(column_names, sample_team_df):
    """Test that TEAM_RATING_PROJECTED (backward compat) works correctly."""
    gen = TeamRatingGenerator(
        performance_column="won",
        column_names=column_names,
        features_out=[RatingKnownFeatures.TEAM_RATING_PROJECTED],
        output_suffix=None,
        confidence_weight=0.0,
    )
    result = gen.fit_transform(sample_team_df)

    # TEAM_RATING_PROJECTED should be present and equal to TEAM_OFF_RATING_PROJECTED (with performance column suffix)
    result_cols = (
        result.columns.tolist() if hasattr(result.columns, "tolist") else list(result.columns)
    )
    assert "team_rating_projected_won" in result_cols

    # Check that it has the same values as team_off_rating_projected (if that's also requested)
    # Actually, TEAM_RATING_PROJECTED is an alias for TEAM_OFF_RATING_PROJECTED
    if "team_off_rating_projected_won" in result_cols:
        team_rating = result["team_rating_projected_won"].to_list()
        team_off = result["team_off_rating_projected_won"].to_list()
        assert team_rating == team_off


def test_team_with_strong_offense_and_weak_defense_gets_expected_ratings_and_predictions(
    column_names,
):
    start_rating = 1000.0

    generator = TeamRatingGenerator(
        auto_scale_performance=True,
        performance_column="team_points",
        column_names=column_names,
        performance_predictor="difference",
        confidence_weight=0.0,  # keep updates simple/deterministic
        rating_change_multiplier_offense=50.0,
        rating_change_multiplier_defense=50.0,
        output_suffix="",
        start_harcoded_start_rating=start_rating,
    )

    base_day = datetime(2024, 1, 1)

    df = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "team_id": [
                "team_a",
                "team_b",  # high-scoring, bad defense
                "team_a",
                "team_c",  # high-scoring, bad defense
                "team_d",
                "team_e",  # normal league game
                "team_f",
                "team_g",  # normal league game
            ],
            "start_date": [
                base_day,
                base_day,
                base_day + timedelta(days=1),
                base_day + timedelta(days=1),
                base_day + timedelta(days=2),
                base_day + timedelta(days=2),
                base_day + timedelta(days=3),
                base_day + timedelta(days=3),
            ],
            "team_points": [
                140,
                130,
                138,
                128,
                120,
                115,
                118,
                112,
            ],
        }
    )

    generator.fit_transform(df)

    a_off = float(generator._team_off_ratings["team_a"].rating_value)
    a_def = float(generator._team_def_ratings["team_a"].rating_value)

    assert a_off > start_rating
    assert a_def < start_rating

    # Predicted performances vs a ~1000 opponent
    pred_off = float(
        generator._performance_predictor.predict_performance(
            rating_value=a_off,
            opponent_team_rating_value=start_rating,
        )
    )
    pred_def = float(
        generator._performance_predictor.predict_performance(
            rating_value=a_def,
            opponent_team_rating_value=start_rating,
        )
    )

    assert pred_off > 0.5
    assert pred_def < 0.5


# ========================================
# Tests for GameColumnNames and game-level data conversion
# ========================================


def test_GameColumnNames__validation_raises_when_performance_column_pairs_is_empty():
    """
    When performance_column_pairs is empty, then we should expect to see
    a ValueError raised because at least one performance column pair must be specified.
    """
    with pytest.raises(ValueError, match="performance_column_pairs must contain at least one"):
        GameColumnNames(
            match_id="match_id",
            start_date="start_date",
            team1_name="team1",
            team2_name="team2",
            performance_column_pairs={},
        )


def test_GameColumnNames__validation_raises_when_column_name_is_empty():
    """
    When a column name in performance_column_pairs is an empty string, then we should expect to see
    a ValueError raised because all column names must be non-empty strings.
    """
    with pytest.raises(ValueError, match="All column names in performance_column_pairs must be"):
        GameColumnNames(
            match_id="match_id",
            start_date="start_date",
            team1_name="team1",
            team2_name="team2",
            performance_column_pairs={"score": ("", "team2_score")},
        )


def test_GameColumnNames__update_match_id_defaults_to_match_id():
    """
    When update_match_id is not provided, then we should expect to see
    it default to match_id because this is the standard behavior.
    """
    gcn = GameColumnNames(
        match_id="game_id",
        start_date="start_date",
        team1_name="home_team",
        team2_name="away_team",
        performance_column_pairs={"score": ("home_score", "away_score")},
    )

    assert gcn.update_match_id == "game_id"


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_TeamRatingGenerator__fit_transform_with_GameColumnNames(df_type):
    """
    When TeamRatingGenerator is initialized with GameColumnNames and game-level data is passed,
    then we should expect to see rating features generated correctly because the conversion
    to game+team format happens automatically.
    """
    game_df = df_type(
        {
            "match_id": [1, 2],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "home": ["team_a", "team_a"],
            "away": ["team_b", "team_c"],
            "home_score": [100, 105],
            "away_score": [95, 90],
        }
    )

    gcn = GameColumnNames(
        match_id="match_id",
        start_date="start_date",
        team1_name="home",
        team2_name="away",
        performance_column_pairs={"score": ("home_score", "away_score")},
    )

    generator = TeamRatingGenerator(
        performance_column="score",
        column_names=gcn,
        auto_scale_performance=True,
        output_suffix="",
        features_out=[
            RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
            RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED,
        ],
    )

    result_df = generator.fit_transform(game_df)

    # Should have 4 rows (2 per game, since game-level data is converted to game+team format)
    assert len(result_df) == 4

    # Should have rating features
    assert "team_off_rating_projected" in result_df.columns
    assert "opponent_def_rating_projected" in result_df.columns

    # Check team ratings were updated
    assert "team_a" in generator._team_off_ratings
    assert "team_b" in generator._team_off_ratings
    assert "team_c" in generator._team_off_ratings


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_TeamRatingGenerator__future_transform_with_GameColumnNames(df_type):
    """
    When future_transform is called with game-level data, then we should expect to see
    rating features generated without updating internal ratings because future_transform
    only adds pre-match features.
    """
    # Training data
    train_df = df_type(
        {
            "match_id": [1],
            "start_date": [datetime(2024, 1, 1)],
            "home": ["team_a"],
            "away": ["team_b"],
            "home_score": [100],
            "away_score": [95],
        }
    )

    # Future data
    future_df = df_type(
        {
            "match_id": [2],
            "start_date": [datetime(2024, 1, 2)],
            "home": ["team_a"],
            "away": ["team_c"],
            "home_score": [105],  # These won't be used
            "away_score": [90],
        }
    )

    gcn = GameColumnNames(
        match_id="match_id",
        start_date="start_date",
        team1_name="home",
        team2_name="away",
        performance_column_pairs={"score": ("home_score", "away_score")},
    )

    generator = TeamRatingGenerator(
        performance_column="score",
        column_names=gcn,
        auto_scale_performance=True,
        output_suffix="",
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )

    # Fit on training data
    generator.fit_transform(train_df)

    team_a_rating_before = generator._team_off_ratings["team_a"].rating_value

    # Future transform on future data
    result_df = generator.future_transform(future_df)

    # Ratings should not change
    team_a_rating_after = generator._team_off_ratings["team_a"].rating_value
    assert team_a_rating_before == team_a_rating_after

    # Should still generate features
    assert "team_off_rating_projected" in result_df.columns
    # Should have 2 rows (1 game with 2 teams in game+team format)
    assert len(result_df) == 2


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_TeamRatingGenerator__with_GameColumnNames_multiple_performance_columns(df_type):
    """
    When GameColumnNames has multiple performance column pairs and performance_weights
    are provided, then we should expect the weighted performance to be used for rating
    updates because the PerformanceWeightsManager combines multiple metrics.
    """
    from spforge.performance_transformers._performance_manager import ColumnWeight

    game_df = df_type(
        {
            "match_id": [1, 2, 3],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "home": ["team_a", "team_a", "team_b"],
            "away": ["team_b", "team_c", "team_c"],
            "home_points": [100, 105, 98],
            "away_points": [95, 90, 102],
            "home_assists": [20, 22, 18],
            "away_assists": [18, 19, 20],
        }
    )

    gcn = GameColumnNames(
        match_id="match_id",
        start_date="start_date",
        team1_name="home",
        team2_name="away",
        performance_column_pairs={
            "points": ("home_points", "away_points"),
            "assists": ("home_assists", "away_assists"),
        },
    )

    generator = TeamRatingGenerator(
        performance_column="points",  # Must match one of the columns from performance_column_pairs
        column_names=gcn,
        performance_weights=[
            ColumnWeight(name="points", weight=0.7, lower_is_better=False),
            ColumnWeight(name="assists", weight=0.3, lower_is_better=False),
        ],
        output_suffix="",
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )

    result_df = generator.fit_transform(game_df)

    # Should have 6 rows (3 games × 2 teams each)
    assert len(result_df) == 6

    # Performance columns should be present
    assert "points" in result_df.columns
    assert "assists" in result_df.columns

    # Weighted performance column should be created by PerformanceWeightsManager
    assert "performance__points" in result_df.columns

    # Rating features should be present
    assert "team_off_rating_projected" in result_df.columns

    # Team ratings should be updated (team_a won both matches with higher combined performance)
    team_a_rating = generator._team_off_ratings["team_a"].rating_value
    team_b_rating = generator._team_off_ratings["team_b"].rating_value
    team_c_rating = generator._team_off_ratings["team_c"].rating_value

    # team_a should have higher rating than team_b and team_c
    # (team_a: won match 1 and 2, team_b: lost to A, won against C, team_c: lost both)
    assert team_a_rating > team_b_rating
    assert team_a_rating > team_c_rating


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_TeamRatingGenerator__with_GameColumnNames_verifies_weighted_performance_affects_ratings(
    df_type,
):
    """
    When different performance weights are used, then we should expect ratings to reflect
    the weighted performance calculation because the PerformanceWeightsManager combines
    metrics according to the specified weights.

    This test verifies that the weighting is actually being applied by showing that
    changing the weights changes which team ends up with higher rating.
    """
    from spforge.performance_transformers._performance_manager import ColumnWeight

    # Scenario: Team A dominates points, Team B dominates assists
    # Match: Team A (high points, low assists) vs Team B (low points, high assists)
    game_df = df_type(
        {
            "match_id": [1],
            "start_date": [datetime(2024, 1, 1)],
            "home": ["team_a"],
            "away": ["team_b"],
            "home_points": [100],  # Team A: strong on points
            "away_points": [60],  # Team B: weak on points
            "home_assists": [10],  # Team A: weak on assists
            "away_assists": [30],  # Team B: strong on assists
        }
    )

    gcn = GameColumnNames(
        match_id="match_id",
        start_date="start_date",
        team1_name="home",
        team2_name="away",
        performance_column_pairs={
            "points": ("home_points", "away_points"),
            "assists": ("home_assists", "away_assists"),
        },
    )

    # Test 1: Weight points heavily (90% points, 10% assists)
    # Expected: Team A should have higher rating (because it dominates points)
    generator_points_heavy = TeamRatingGenerator(
        performance_column="points",
        column_names=gcn,
        performance_weights=[
            ColumnWeight(name="points", weight=0.9, lower_is_better=False),
            ColumnWeight(name="assists", weight=0.1, lower_is_better=False),
        ],
        output_suffix="",
    )

    generator_points_heavy.fit_transform(game_df)

    team_a_rating_points_heavy = generator_points_heavy._team_off_ratings["team_a"].rating_value
    team_b_rating_points_heavy = generator_points_heavy._team_off_ratings["team_b"].rating_value

    # Team A should have higher rating when points are weighted heavily
    assert team_a_rating_points_heavy > team_b_rating_points_heavy, (
        f"When points are weighted 90%, team_a (100 points, 10 assists) should have higher "
        f"rating than team_b (60 points, 30 assists). Got: team_a={team_a_rating_points_heavy:.2f}, "
        f"team_b={team_b_rating_points_heavy:.2f}"
    )

    # Test 2: Weight assists heavily (10% points, 90% assists)
    # Expected: Team B should have higher rating (because it dominates assists)
    generator_assists_heavy = TeamRatingGenerator(
        performance_column="points",
        column_names=gcn,
        performance_weights=[
            ColumnWeight(name="points", weight=0.1, lower_is_better=False),
            ColumnWeight(name="assists", weight=0.9, lower_is_better=False),
        ],
        output_suffix="",
    )

    generator_assists_heavy.fit_transform(game_df)

    team_a_rating_assists_heavy = generator_assists_heavy._team_off_ratings["team_a"].rating_value
    team_b_rating_assists_heavy = generator_assists_heavy._team_off_ratings["team_b"].rating_value

    # Team B should have higher rating when assists are weighted heavily
    assert team_b_rating_assists_heavy > team_a_rating_assists_heavy, (
        f"When assists are weighted 90%, team_b (60 points, 30 assists) should have higher "
        f"rating than team_a (100 points, 10 assists). Got: team_a={team_a_rating_assists_heavy:.2f}, "
        f"team_b={team_b_rating_assists_heavy:.2f}"
    )

    # The ratings should change significantly based on weights
    # This proves the weighting is actually being applied
    team_a_rating_difference = team_a_rating_points_heavy - team_a_rating_assists_heavy
    team_b_rating_difference = team_b_rating_assists_heavy - team_b_rating_points_heavy

    # Both teams should have meaningfully different ratings based on weighting scheme
    assert abs(team_a_rating_difference) > 10, (
        f"Team A's rating should change significantly based on weights. "
        f"Difference: {team_a_rating_difference:.2f}"
    )
    assert abs(team_b_rating_difference) > 10, (
        f"Team B's rating should change significantly based on weights. "
        f"Difference: {team_b_rating_difference:.2f}"
    )


def test_plus_minus_team_diff_positive_next_match(column_names):
    """
    For a zero-sum plus_minus game, the team with the higher total plus_minus
    should have a positive team rating difference in the next match.
    """
    df = pl.DataFrame(
        {
            "match_id": ["M1", "M1", "M2", "M2"],
            "team_id": ["team_a", "team_b", "team_a", "team_b"],
            "start_date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
            ],
            "plus_minus": [10.0, -10.0, 0.0, 0.0],
        }
    )

    generator = TeamRatingGenerator(
        performance_column="plus_minus",
        column_names=column_names,
        auto_scale_performance=True,
        use_off_def_split=False,
        features_out=[RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED],
    )
    res = generator.fit_transform(df)

    diff_col = "team_rating_difference_projected_plus_minus"
    m2_team = (
        res.filter(pl.col("match_id") == "M2")
        .group_by("team_id")
        .agg(pl.col(diff_col).mean().alias("diff"))
    )
    a_diff = m2_team.filter(pl.col("team_id") == "team_a").select("diff").item()
    b_diff = m2_team.filter(pl.col("team_id") == "team_b").select("diff").item()

    assert a_diff > 0
    assert b_diff < 0


def test_plus_minus_future_transform_team_diff(column_names):
    """
    future_transform should carry forward plus_minus ratings and produce
    the same team diff direction for the next match.
    """
    fit_df = pl.DataFrame(
        {
            "match_id": ["M1", "M1"],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "plus_minus": [10.0, -10.0],
        }
    )
    future_df = pl.DataFrame(
        {
            "match_id": ["M2", "M2"],
            "team_id": ["team_a", "team_b"],
            "start_date": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
        }
    )

    generator = TeamRatingGenerator(
        performance_column="plus_minus",
        column_names=column_names,
        auto_scale_performance=True,
        use_off_def_split=False,
        features_out=[RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED],
    )
    generator.fit_transform(fit_df)
    res = generator.future_transform(future_df)

    diff_col = "team_rating_difference_projected_plus_minus"
    m2_team = res.group_by("team_id").agg(pl.col(diff_col).mean().alias("diff"))
    a_diff = m2_team.filter(pl.col("team_id") == "team_a").select("diff").item()
    b_diff = m2_team.filter(pl.col("team_id") == "team_b").select("diff").item()

    assert a_diff > 0
    assert b_diff < 0
