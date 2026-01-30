from datetime import datetime, timedelta

import polars as pl
import pytest

from spforge import ColumnNames
from spforge.data_structures import RatingState
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures, RatingUnknownFeatures


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


@pytest.fixture
def sample_df(base_cn):
    """A standard 1-match, 2-team, 4-player setup."""
    return pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.6, 0.4, 0.7, 0.3],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def sequential_df(base_cn):
    """A sequential 3-match setup for testing rating evolution."""
    return pl.DataFrame(
        {
            "pid": ["P1", "P2", "P1", "P2", "P1", "P2"],
            "tid": ["T1", "T2", "T1", "T2", "T1", "T2"],
            "mid": ["M1", "M1", "M2", "M2", "M3", "M3"],
            "dt": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "perf": [0.8, 0.2, 0.8, 0.2, 0.8, 0.2],
            "pw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.mark.parametrize("perf_val", [-0.1, 1.1])
def test_fit_transform_performance_bounds_validation(base_cn, sample_df, perf_val):
    """Check that the generator raises ValueError when performance is outside [0, 1]."""
    df = sample_df.with_columns(pl.lit(perf_val).alias("perf"))
    gen = PlayerRatingGenerator(performance_column="perf", column_names=base_cn)
    with pytest.raises(ValueError):
        gen.fit_transform(df)


def test_fit_transform_updates_internal_state(base_cn, sample_df):
    """Verify that fit_transform populates the internal rating dictionaries."""
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    assert len(gen._player_off_ratings) == 0

    gen.fit_transform(sample_df)

    assert "P1" in gen._player_off_ratings
    assert "P1" in gen._player_def_ratings

    assert gen._player_off_ratings["P1"].rating_value > 0


def test_fit_transform_participation_weight_scaling(base_cn):
    """Test that a player with lower participation weight receives a smaller rating update."""
    df = pl.DataFrame(
        {
            "pid": ["Full", "Half", "Opp1", "Opp2"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.8, 0.8, 0.2, 0.2],
            "pw": [1.0, 0.5, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    gen.fit_transform(df)

    full_rating = gen._player_off_ratings["Full"].rating_value
    half_rating = gen._player_off_ratings["Half"].rating_value

    assert full_rating > half_rating
    assert half_rating > 0


def test_fit_transform_batch_update_logic(base_cn):
    """Test that ratings do not update between matches if update_match_id is the same."""

    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P1", "P2"],
            "tid": ["T1", "T2", "T1", "T2"],
            "mid": ["M1", "M1", "M2", "M2"],
            "update_id": ["Batch1", "Batch1", "Batch1", "Batch1"],
            "dt": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "perf": [0.9, 0.1, 0.9, 0.1],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    from dataclasses import replace

    cn = replace(base_cn, update_match_id="update_id")
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=cn, auto_scale_performance=True
    )
    output = gen.fit_transform(df)

    assert len(output) >= 2

    assert len(gen._player_off_ratings) > 0


@pytest.mark.parametrize(
    "feature",
    [
        RatingKnownFeatures.PLAYER_OFF_RATING,
        RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
        RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED,
    ],
)
def test_fit_transform_requested_features_presence(base_cn, sample_df, feature):
    """Verify that specific requested features appear in the resulting DataFrame."""
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, features_out=[feature]
    )
    res = gen.fit_transform(sample_df)

    expected_col = f"{feature}_perf"
    assert expected_col in res.columns


def test_future_transform_no_state_mutation(base_cn, sample_df):
    """Ensure that calling future_transform does not alter the model's internal ratings."""
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    gen.fit_transform(sample_df)

    off_state_before = gen._player_off_ratings["P1"].rating_value

    future_df = sample_df.with_columns(pl.lit("M-FUTURE").alias("mid"))
    gen.future_transform(future_df)

    off_state_after = gen._player_off_ratings["P1"].rating_value
    assert off_state_before == off_state_after


def test_future_transform_cold_start_player(base_cn, sample_df):
    """Check that future_transform handles players not seen during fit_transform."""
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    gen.fit_transform(sample_df)

    new_player_df = pl.DataFrame(
        {
            "pid": ["P99", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M2", "M2", "M2", "M2"],
            "dt": ["2024-01-05"] * 4,
            "pw": [1.0] * 4,
        }
    )

    res = gen.future_transform(new_player_df)

    assert "P99" in res["pid"].to_list()

    assert len(res) >= 2


def test_transform_is_identical_to_future_transform(base_cn, sample_df):
    """Verify that the standard transform() call redirects to future_transform logic."""
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )

    gen.fit_transform(sample_df)

    res_transform = gen.transform(sample_df)
    res_future = gen.future_transform(sample_df)

    assert len(res_transform) == len(res_future)

    assert set(res_transform.columns) == set(res_future.columns)


def test_fit_transform_offense_defense_independence(base_cn):
    """Verify that Offense and Defense ratings update based on different logic."""

    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.8, 0.7, 0.8, 0.7],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    gen.fit_transform(df)

    off_rating = gen._player_off_ratings["P1"].rating_value
    assert off_rating > 0.0

    def_rating = gen._player_def_ratings["P1"].rating_value

    assert def_rating < off_rating or def_rating <= 0.0


def test_plus_minus_does_not_split_off_def(base_cn):
    """
    plus_minus represents overall impact. For a single game where team totals
    are equal/opposite, offense and defense ratings should not diverge.
    """
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "plus_minus": [8.0, 2.0, -5.0, -5.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="plus_minus",
        column_names=base_cn,
        auto_scale_performance=True,
        use_off_def_split=False,
    )
    gen.fit_transform(df)

    off_rating = gen._player_off_ratings["P1"].rating_value
    def_rating = gen._player_def_ratings["P1"].rating_value

    assert abs(off_rating - def_rating) < 1e-9


def test_plus_minus_team_diff_positive_next_match(base_cn):
    """
    For a zero-sum plus_minus game, the team with the higher total plus_minus
    should have a positive team rating difference in the next match.
    """
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4", "P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2", "T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1", "M2", "M2", "M2", "M2"],
            "dt": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
            "plus_minus": [8.0, 7.0, -6.0, -9.0, 0.0, 0.0, 0.0, 0.0],
            "pw": [1.0] * 8,
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="plus_minus",
        column_names=base_cn,
        auto_scale_performance=True,
        use_off_def_split=False,
    )
    res = gen.fit_transform(df)

    diff_col = "team_rating_difference_projected_plus_minus"
    m2_team = res.filter(pl.col("mid") == "M2").group_by("tid").agg(
        pl.col(diff_col).mean().alias("diff")
    )
    t1_diff = m2_team.filter(pl.col("tid") == "T1").select("diff").item()
    t2_diff = m2_team.filter(pl.col("tid") == "T2").select("diff").item()

    assert t1_diff > 0
    assert t2_diff < 0


def test_plus_minus_future_transform_team_diff(base_cn):
    """
    future_transform should carry forward plus_minus ratings and produce
    the same team diff direction for the next match.
    """
    fit_df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "plus_minus": [8.0, 7.0, -6.0, -9.0],
            "pw": [1.0] * 4,
        }
    )
    future_df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M2", "M2", "M2", "M2"],
            "dt": ["2024-01-02"] * 4,
            "pw": [1.0] * 4,
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="plus_minus",
        column_names=base_cn,
        auto_scale_performance=True,
        use_off_def_split=False,
        features_out=[RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED],
    )
    gen.fit_transform(fit_df)
    res = gen.future_transform(future_df)

    diff_col = "team_rating_difference_projected_plus_minus"
    m2_team = res.group_by("tid").agg(pl.col(diff_col).mean().alias("diff"))
    t1_diff = m2_team.filter(pl.col("tid") == "T1").select("diff").item()
    t2_diff = m2_team.filter(pl.col("tid") == "T2").select("diff").item()

    assert t1_diff > 0
    assert t2_diff < 0


def _create_date_column_player(date_format: str, dates: list) -> pl.Series:
    """Helper to create date column with specified format for player tests."""
    if date_format == "string_iso_date":
        return pl.Series("dt", dates)
    elif date_format == "string_datetime_space":
        date_strings = [
            f"{d} 12:00:00" if isinstance(d, str) and " " not in d else d for d in dates
        ]
        return pl.Series("dt", date_strings)
    elif date_format == "date_type":
        from datetime import date

        date_objs = [
            date(2024, 1, int(d.split("-")[2])) if isinstance(d, str) and "-" in d else d
            for d in dates
        ]
        return pl.Series("dt", date_objs, dtype=pl.Date)
    elif date_format == "datetime_type":
        from datetime import datetime

        datetime_objs = [
            datetime(2024, 1, int(d.split("-")[2])) if isinstance(d, str) and "-" in d else d
            for d in dates
        ]
        return pl.Series("dt", datetime_objs, dtype=pl.Datetime(time_zone=None))
    else:
        raise ValueError(f"Unknown date_format: {date_format}")


@pytest.mark.parametrize(
    "date_format",
    [
        "string_iso_date",
        "string_datetime_space",
        "date_type",
        "datetime_type",
    ],
)
def test_fit_transform_when_date_formats_vary_then_processes_successfully_player(
    base_cn, date_format
):
    """
    When date formats vary for player rating generator, then we should expect to see
    fit_transform processes successfully because _add_day_number handles all these formats.
    """
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
    )

    dates = ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]
    date_col = _create_date_column_player(date_format, dates)

    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": date_col,
            "perf": [0.6, 0.4, 0.7, 0.3],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = gen.fit_transform(df)

    assert len(result) >= 2

    assert len(gen._player_off_ratings) > 0


@pytest.mark.parametrize(
    "date_format",
    [
        "string_iso_date",
        "datetime_type",
    ],
)
def test_future_transform_when_date_formats_vary_then_processes_successfully_player(
    base_cn, date_format
):
    """
    When date formats vary for player rating generator, then we should expect to see
    future_transform processes successfully because _add_day_number handles all these formats.
    """
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
    )

    fit_df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": _create_date_column_player("string_iso_date", ["2024-01-01"] * 4),
            "perf": [0.6, 0.4, 0.7, 0.3],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gen.fit_transform(fit_df)

    p1_rating_before = gen._player_off_ratings["P1"].rating_value

    dates = ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02"]
    date_col = _create_date_column_player(date_format, dates)

    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M2", "M2", "M2", "M2"],
            "dt": date_col,
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = gen.future_transform(df)

    assert len(result) >= 2

    assert gen._player_off_ratings["P1"].rating_value == p1_rating_before


def test_init_custom_suffix_applied_to_all_features(base_cn):
    """Verify that the output_suffix is appended to all requested feature columns."""
    gen = PlayerRatingGenerator(
        performance_column="goals",
        column_names=base_cn,
        output_suffix="v2",
        features_out=[
            RatingKnownFeatures.PLAYER_OFF_RATING,
            RatingKnownFeatures.TEAM_DEF_RATING_PROJECTED,
        ],
    )
    # The generator uses performance_column as a default suffix if output_suffix is None,
    # but if output_suffix is provided, it should respect it.
    assert gen.PLAYER_OFF_RATING_COL == "player_off_rating_v2"
    assert gen.TEAM_DEF_RATING_PROJ_COL == "team_def_rating_projected_v2"


def test_init_multi_performance_weights(base_cn):
    """Verify initialization when multiple performance columns are weighted."""
    weights = [{"col": "goals", "weight": 0.7}, {"col": "assists", "weight": 0.3}]
    gen = PlayerRatingGenerator(
        performance_column="ignored", performance_weights=weights, column_names=base_cn
    )
    # Internally it should have a PerformanceManager
    assert gen.performance_manager is not None


# --- fit_transform Tests ---


def test_fit_transform_zero_participation_weight(base_cn):
    """A player with 0 participation weight should experience no rating change."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P_Opp"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01"],
            "perf": [1.0, 0.0],
            "pw": [0.0, 1.0],  # P1 played 0 minutes
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_harcoded_start_rating=0.0,  # Set start rating to 0.0 for this test
    )
    gen.fit_transform(df)

    # Rating should remain at the start value (0.0) since participation weight is 0
    # (no rating change occurs when participation weight is 0)
    assert gen._player_off_ratings["P1"].rating_value == 0.0


def test_fit_transform_scales_participation_weight_by_fit_quantile(base_cn):
    """Participation weight ratio should reflect scaling by the fit 99th percentile."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "O1", "O2"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.9, 0.9, 0.1, 0.1],
            "pw": [10.0, 20.0, 10.0, 10.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_harcoded_start_rating=1000.0,
        scale_participation_weights=True,
    )
    gen.fit_transform(df)

    start_rating = 1000.0
    p1_change = gen._player_off_ratings["P1"].rating_value - start_rating
    p2_change = gen._player_off_ratings["P2"].rating_value - start_rating

    q = df["pw"].quantile(0.99, "linear")
    expected_ratio = min(1.0, 10.0 / q) / min(1.0, 20.0 / q)

    assert p1_change / p2_change == pytest.approx(expected_ratio, rel=1e-6)


def test_fit_transform_auto_scales_participation_weight_when_out_of_bounds(base_cn):
    """Automatically enable scaling when participation weights exceed [0, 1]."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "O1", "O2"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.9, 0.9, 0.1, 0.1],
            "pw": [10.0, 20.0, 10.0, 10.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_harcoded_start_rating=1000.0,
    )
    gen.fit_transform(df)

    start_rating = 1000.0
    p1_change = gen._player_off_ratings["P1"].rating_value - start_rating
    p2_change = gen._player_off_ratings["P2"].rating_value - start_rating

    q = df["pw"].quantile(0.99, "linear")
    expected_ratio = min(1.0, 10.0 / q) / min(1.0, 20.0 / q)

    assert gen.scale_participation_weights is True
    assert p1_change / p2_change == pytest.approx(expected_ratio, rel=1e-6)


def test_fit_transform_auto_scale_logs_warning_when_out_of_bounds(base_cn, caplog):
    """Auto-scaling should emit a warning when participation weights exceed [0, 1]."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "O1", "O2"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.9, 0.9, 0.1, 0.1],
            "pw": [10.0, 20.0, 10.0, 10.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_harcoded_start_rating=1000.0,
    )
    with caplog.at_level("WARNING"):
        gen.fit_transform(df)

    assert any(
        "Auto-scaling participation weights" in record.message for record in caplog.records
    )


def test_future_transform_scales_projected_participation_weight_by_fit_quantile():
    """Future projected participation weights should scale with fit quantile and be clipped."""
    cn = ColumnNames(
        player_id="pid",
        team_id="tid",
        match_id="mid",
        start_date="dt",
        update_match_id="mid",
        participation_weight="pw",
        projected_participation_weight="ppw",
    )
    fit_df = pl.DataFrame(
        {
            "pid": ["A", "B", "C", "D"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.9, 0.1, 0.9, 0.1],
            "pw": [10.0, 10.0, 10.0, 10.0],
            "ppw": [10.0, 10.0, 10.0, 10.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn,
        auto_scale_performance=True,
        scale_participation_weights=True,
        features_out=[RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
    )
    gen.fit_transform(fit_df)

    future_df = pl.DataFrame(
        {
            "pid": ["A", "B", "C", "D"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M2", "M2", "M2", "M2"],
            "dt": ["2024-01-02"] * 4,
            "pw": [10.0, 10.0, 10.0, 10.0],
            "ppw": [20.0, 5.0, 10.0, 10.0],
        }
    )
    res = gen.future_transform(future_df)

    a_rating = gen._player_off_ratings["A"].rating_value
    b_rating = gen._player_off_ratings["B"].rating_value
    w_a = min(1.0, max(0.0, 20.0 / 10.0))
    w_b = min(1.0, max(0.0, 5.0 / 10.0))
    expected_team_off = (a_rating * w_a + b_rating * w_b) / (w_a + w_b)

    team_off_col = "team_off_rating_projected_perf"
    actual_team_off = (
        res.filter(pl.col("tid") == "T1").select(team_off_col).unique().item()
    )

    assert actual_team_off == pytest.approx(expected_team_off, rel=1e-6)


def test_fit_transform_sequential_rating_evolution(base_cn, sequential_df):
    """Ratings should change monotonically if a player performs consistently above average."""
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    gen.fit_transform(sequential_df)

    # After processing 3 matches where P1 consistently performs well (0.8),
    # the final rating should be higher than the initial rating (1000)
    final_rating = gen._player_off_ratings["P1"].rating_value
    assert final_rating > 1000.0  # Should have increased from default start rating

    # Also verify that confidence increased
    assert gen._player_off_ratings["P1"].confidence_sum > 1.0
    assert gen._player_off_ratings["P1"].games_played == 3.0


def test_fit_transform_confidence_decay_over_time(base_cn):
    """Players who haven't played for a long time should have their confidence sum decreased."""
    # Match 1: Initial play
    # Match 2: Play after 200 days (exceeds default confidence_max_days=120)
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P1", "P2"],
            "tid": ["T1", "T2", "T1", "T2"],
            "mid": ["M1", "M1", "M2", "M2"],
            "dt": ["2024-01-01", "2024-01-01", "2024-10-01", "2024-10-01"],
            "perf": [0.5, 0.5, 0.8, 0.2],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        confidence_days_ago_multiplier=1.0,  # Aggressive decay
    )
    gen.fit_transform(df)

    # After Match 1, confidence_sum should be 1.0.
    # Before Match 2, the 200+ days gap should trigger decay in _post_match_confidence_sum.
    # We check internal state
    assert gen._player_off_ratings["P1"].confidence_sum < 2.0


def test_fit_transform_null_performance_handling(base_cn, sample_df):
    """Rows with null performance should be handled or skipped without crashing."""
    df_with_null = sample_df.with_columns(
        pl.when(pl.col("pid") == "P1").then(None).otherwise(pl.col("perf")).alias("perf")
    )
    gen = PlayerRatingGenerator(performance_column="perf", column_names=base_cn)

    # Depending on implementation, it might skip P1 or treat as 0.
    # The key is that the generator shouldn't crash.
    res = gen.fit_transform(df_with_null)
    assert len(res) == 4


def test_fit_transform_null_performance__no_rating_change(base_cn):
    """Players with null performance should have zero rating change, not be treated as 0.0 perf."""
    # Match 1: Both players have performance (P1=0.6, P2=0.4)
    # Match 2: P1 has null performance, P2 has 0.6
    # Match 3: Both players have performance again
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P1", "P2", "P1", "P2"],
            "tid": ["T1", "T2", "T1", "T2", "T1", "T2"],
            "mid": ["M1", "M1", "M2", "M2", "M3", "M3"],
            "dt": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "perf": [0.6, 0.4, None, 0.6, 0.6, 0.4],  # P1 has null in M2
            "pw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    result = gen.fit_transform(df)

    # Get P1's pre-match rating for M2 (after M1) and M3 (after M2 with null perf)
    p1_rating_before_m2 = result.filter(
        (pl.col("pid") == "P1") & (pl.col("mid") == "M2")
    )["player_off_rating_perf"][0]
    p1_rating_before_m3 = result.filter(
        (pl.col("pid") == "P1") & (pl.col("mid") == "M3")
    )["player_off_rating_perf"][0]

    # Key assertion: P1's rating before M3 should equal rating before M2
    # because null performance in M2 means NO rating change
    assert p1_rating_before_m3 == p1_rating_before_m2, (
        f"P1's rating changed after null performance game! "
        f"Before M2={p1_rating_before_m2}, Before M3={p1_rating_before_m3}"
    )

    # Also verify null is not treated as 0.0 by comparing with explicit 0.0
    df_with_zero = df.with_columns(
        pl.when((pl.col("pid") == "P1") & (pl.col("mid") == "M2"))
        .then(0.0)
        .otherwise(pl.col("perf"))
        .alias("perf")
    )

    gen_zero = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    result_zero = gen_zero.fit_transform(df_with_zero)

    p1_rating_before_m3_with_zero = result_zero.filter(
        (pl.col("pid") == "P1") & (pl.col("mid") == "M3")
    )["player_off_rating_perf"][0]

    # With 0.0 perf, rating should drop (different from null)
    assert p1_rating_before_m3 > p1_rating_before_m3_with_zero, (
        f"Null performance is being treated as 0.0! "
        f"Rating with null={p1_rating_before_m3}, rating with 0.0={p1_rating_before_m3_with_zero}"
    )


def test_fit_transform_null_performance__still_outputs_player_rating(base_cn):
    """Players with null performance should still have their pre-match rating in output."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.6, None, 0.4, 0.5],  # P2 has null performance
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    result = gen.fit_transform(df)

    # P2 should still be in output with their pre-match rating
    assert len(result) == 4
    p2_row = result.filter(pl.col("pid") == "P2")
    assert len(p2_row) == 1
    assert "player_off_rating_perf" in result.columns
    # P2's rating should be the start rating (1000.0) since they're new and had no update
    assert p2_row["player_off_rating_perf"][0] == 1000.0


def test_transform_null_performance__no_rating_change(base_cn):
    """In transform (historical), null performance should result in no rating change."""
    # First fit with some data
    fit_df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01"],
            "perf": [0.6, 0.4],
            "pw": [1.0, 1.0],
        }
    )

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    gen.fit_transform(fit_df)

    p1_rating_before = gen._player_off_ratings["P1"].rating_value

    # Now transform with P1 having null performance
    transform_df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-02", "2024-01-02"],
            "perf": [None, 0.6],  # P1 has null
            "pw": [1.0, 1.0],
        }
    )

    gen.transform(transform_df)

    p1_rating_after = gen._player_off_ratings["P1"].rating_value

    # P1's rating should not change significantly (only confidence decay, not performance-based)
    # Since null perf means no rating change from performance
    assert abs(p1_rating_after - p1_rating_before) < 0.01, (
        f"P1's rating changed significantly with null performance: "
        f"before={p1_rating_before}, after={p1_rating_after}"
    )


def test_future_transform_null_performance__outputs_projections(base_cn):
    """In future_transform, null performance should still output rating projections."""
    # First fit with some data
    fit_df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01"],
            "perf": [0.6, 0.4],
            "pw": [1.0, 1.0],
        }
    )

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    gen.fit_transform(fit_df)

    p1_rating_before = gen._player_off_ratings["P1"].rating_value

    # Future transform (no performance needed, but if null it shouldn't affect anything)
    future_df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-02", "2024-01-02"],
            "pw": [1.0, 1.0],
            # No perf column - this is a future match
        }
    )

    result = gen.future_transform(future_df)

    # Should output projections for all players
    assert len(result) == 2
    assert "player_off_rating_perf" in result.columns

    # Ratings should NOT be updated (future_transform doesn't update state)
    assert gen._player_off_ratings["P1"].rating_value == p1_rating_before


# --- transform & future_transform Tests ---


def test_transform_error_before_fit(base_cn, sample_df):
    """Calling transform before fit/fit_transform should raise an error or return start ratings."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        features_out=[
            RatingKnownFeatures.PLAYER_OFF_RATING
        ],  # Explicitly request player_off_rating
    )
    # If the model isn't fitted, it hasn't seen any players.
    # It should still work but return default start ratings for everyone.
    res = gen.transform(sample_df)
    assert "player_off_rating_perf" in res.columns
    # Default start rating is 1000.0, not 0.0
    assert res["player_off_rating_perf"][0] == 1000.0


def test_future_transform_extreme_rating_differences(base_cn):
    """Verify predictions stay within [0, 1] even with massive rating gaps."""
    from spforge.data_structures import PlayerRating
    from spforge.ratings import RatingUnknownFeatures

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        non_predictor_features_out=[
            RatingUnknownFeatures.PLAYER_PREDICTED_OFF_PERFORMANCE
        ],  # Request prediction column
    )

    gen._player_off_ratings["GOD"] = PlayerRating(id="GOD", rating_value=100.0, confidence_sum=30)
    gen._player_def_ratings["GOD"] = PlayerRating(id="GOD", rating_value=100.0, confidence_sum=30)
    gen._player_off_ratings["NOOB"] = PlayerRating(
        id="NOOB", rating_value=-100.0, confidence_sum=30
    )
    gen._player_def_ratings["NOOB"] = PlayerRating(
        id="NOOB", rating_value=-100.0, confidence_sum=30
    )

    future_df = pl.DataFrame(
        {
            "pid": ["GOD", "NOOB"],
            "tid": ["T1", "T2"],
            "mid": ["M-FUTURE", "M-FUTURE"],  # Both players in same match
            "dt": ["2025-01-01", "2025-01-01"],
            "pw": [1.0, 1.0],
        }
    )

    res = gen.future_transform(future_df)
    pred_col = "player_predicted_off_performance_perf"  # Correct column name with suffix

    # Predictions should be clipped or sigmoid-like between 0 and 1
    assert 0.0 <= res.filter(pl.col("pid") == "GOD")[pred_col][0] <= 1.0
    assert 0.0 <= res.filter(pl.col("pid") == "NOOB")[pred_col][0] <= 1.0


# --- Multiple Call / State Persistence Tests ---


def test_fit_transform_multiple_calls_persistence(base_cn):
    """Calling fit_transform twice should result in additive updates to the state."""
    df1 = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01"],
            "perf": [0.8, 0.2],
            "pw": [1.0, 1.0],
        }
    )
    df2 = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-02", "2024-01-02"],
            "perf": [0.8, 0.2],
            "pw": [1.0, 1.0],
        }
    )

    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )

    gen.fit_transform(df1)
    rating_after_1 = gen._player_off_ratings["P1"].rating_value

    gen.fit_transform(df2)
    rating_after_2 = gen._player_off_ratings["P1"].rating_value

    assert rating_after_2 > rating_after_1
    assert gen._player_off_ratings["P1"].confidence_sum > 1.0


def test_fit_transform_auto_scale_false(base_cn):
    """Verify behavior when auto_scale_performance is False."""
    # Performance is already 0-1, but auto_scale=False should skip the min/max adjustment.
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01"],
            "perf": [0.9, 0.1],
            "pw": [1.0, 1.0],
        }
    )
    gen_no_scale = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=False
    )
    gen_no_scale.fit_transform(df)

    # If it didn't scale, the 0.9 should be compared directly against the default prediction (0.5).
    assert gen_no_scale._player_off_ratings["P1"].rating_value > 0


# --- Team-Changing Behavior Tests ---


def test_fit_transform_team_change_tracking(base_cn):
    """Verify that most_recent_team_id is updated when a player changes teams."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P1", "P3"],  # P1 changes from T1 to T2
            "tid": ["T1", "T2", "T2", "T1"],
            "mid": ["M1", "M1", "M2", "M2"],
            "dt": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "perf": [0.6, 0.4, 0.7, 0.3],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    gen.fit_transform(df)

    # After M1, P1 should be on T1
    # After M2, P1 should be on T2
    assert gen._player_off_ratings["P1"].most_recent_team_id == "T2"
    assert gen._player_def_ratings["P1"].most_recent_team_id == "T2"


def test_fit_transform_multiple_team_changes(base_cn):
    """Verify team tracking works correctly with multiple team changes."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P1", "P3", "P1", "P4"],  # P1: T1 -> T2 -> T3
            "tid": ["T1", "T2", "T2", "T1", "T3", "T1"],
            "mid": ["M1", "M1", "M2", "M2", "M3", "M3"],
            "dt": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "perf": [0.6, 0.4, 0.7, 0.3, 0.8, 0.2],
            "pw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    gen.fit_transform(df)

    assert gen._player_off_ratings["P1"].most_recent_team_id == "T3"
    assert gen._player_def_ratings["P1"].most_recent_team_id == "T3"


def test_fit_transform_team_change_in_same_batch(base_cn):
    """Verify team change tracking when update_match_id groups multiple matches."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P1", "P3"],
            "tid": ["T1", "T2", "T2", "T1"],  # P1 changes teams
            "mid": ["M1", "M1", "M2", "M2"],
            "update_id": ["Batch1", "Batch1", "Batch1", "Batch1"],  # Same batch
            "dt": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "perf": [0.6, 0.4, 0.7, 0.3],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    from dataclasses import replace

    cn = replace(base_cn, update_match_id="update_id")
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=cn, auto_scale_performance=True
    )
    gen.fit_transform(df)

    # P1 should end up on T2 (last team in the batch)
    assert gen._player_off_ratings["P1"].most_recent_team_id == "T2"
    assert gen._player_def_ratings["P1"].most_recent_team_id == "T2"


# NOTE: team_id_change_confidence_sum_decrease parameter exists but is not currently used in the code.
# The logic to decrease confidence when team changes is missing from _apply_player_updates.
# This test documents the expected behavior if/when the feature is implemented.
def test_fit_transform_team_change_confidence_decrease_not_implemented(base_cn):
    """Document that team change confidence decrease is not currently implemented."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P1", "P3"],  # P1 changes from T1 to T2
            "tid": ["T1", "T2", "T2", "T1"],
            "mid": ["M1", "M1", "M2", "M2"],
            "dt": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "perf": [0.6, 0.4, 0.7, 0.3],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        team_id_change_confidence_sum_decrease=5.0,  # Set but not used
    )
    gen.fit_transform(df)

    # Currently, confidence_sum increases normally regardless of team change
    # If implemented, confidence_sum should decrease by team_id_change_confidence_sum_decrease
    # when team changes from previous most_recent_team_id
    p1_confidence = gen._player_off_ratings["P1"].confidence_sum
    # This test just verifies the parameter exists and team_id is tracked
    assert gen._player_off_ratings["P1"].most_recent_team_id == "T2"
    assert p1_confidence > 0  # Confidence exists, but team change doesn't affect it currently


# --- Start Rating Generator Tests ---


def test_fit_transform_hardcoded_start_rating(base_cn):
    """When start_harcoded_start_rating is set, all new players get that rating."""
    hardcoded_rating = 1500.0
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_harcoded_start_rating=hardcoded_rating,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )

    # Test with future_transform to see the start rating before updates
    future_df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01"] * 2,
            "pw": [1.0, 1.0],
        }
    )
    res = gen.future_transform(future_df)

    # New players should get the hardcoded start rating
    p1_rating = res.filter(pl.col("pid") == "P1")["player_off_rating_perf"][0]
    p2_rating = res.filter(pl.col("pid") == "P2")["player_off_rating_perf"][0]
    assert p1_rating == hardcoded_rating
    assert p2_rating == hardcoded_rating

    # Also test fit_transform - ratings will start at hardcoded value then update
    df = pl.DataFrame(
        {
            "pid": ["P3", "P4"],
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-02"] * 2,
            "perf": [0.6, 0.4],
            "pw": [1.0, 1.0],
        }
    )
    gen.fit_transform(df)

    # After match, ratings will have updated from hardcoded start
    # But we can verify they started from hardcoded by checking they're close
    # (they'll be updated based on performance)
    assert gen._player_off_ratings["P3"].rating_value != 1000.0  # Not default
    assert gen._player_def_ratings["P3"].rating_value != 1000.0  # Not default


def test_future_transform_hardcoded_start_rating(base_cn, sample_df):
    """Hardcoded start rating works in future_transform for unseen players."""
    hardcoded_rating = 1200.0
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_harcoded_start_rating=hardcoded_rating,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING, RatingKnownFeatures.PLAYER_DEF_RATING],
    )
    gen.fit_transform(sample_df)

    # New player in future_transform should get hardcoded rating
    # Need both teams to have players for the match structure
    future_df = pl.DataFrame(
        {
            "pid": ["P99", "P100"],  # Both are new
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-05", "2024-01-05"],
            "pw": [1.0, 1.0],
        }
    )
    res = gen.future_transform(future_df)

    p99_row = res.filter(pl.col("pid") == "P99")
    assert len(p99_row) > 0
    assert p99_row["player_off_rating_perf"][0] == hardcoded_rating
    assert p99_row["player_def_rating_perf"][0] == hardcoded_rating


def test_fit_transform_default_league_rating(base_cn):
    """When no league data exists, new players get default league rating (1000)."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01"],
            "perf": [0.6, 0.4],
            "pw": [1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=base_cn, auto_scale_performance=True
    )
    gen.fit_transform(df)

    # Default league rating starts at 1000 (DEFAULT_START_RATING)
    # But after the match, ratings get updated based on performance
    # P1 performed 0.6 (above average), so rating should increase from 1000
    assert gen._player_off_ratings["P1"].rating_value > 1000.0
    assert gen._player_def_ratings["P1"].rating_value > 1000.0


def test_fit_transform_league_specific_start_rating(base_cn):
    """League-specific start ratings via start_league_ratings parameter."""
    # Use separate matches for each team pair to ensure all players are processed
    # (The code processes one team pair per match due to .unique(match_id))
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],  # P4 is opponent for P3
            "tid": ["T1", "T2", "T3", "T4"],
            "mid": ["M1", "M1", "M2", "M2"],  # Two matches: M1 (P1 vs P2), M2 (P3 vs P4)
            "dt": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
            "perf": [0.6, 0.4, 0.5, 0.5],
            "pw": [1.0, 1.0, 1.0, 1.0],
            "league": ["NBA", "NBA", "G-League", "G-League"],
        }
    )
    from dataclasses import replace

    cn = replace(base_cn, league="league")
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn,
        auto_scale_performance=True,
        start_league_ratings={"NBA": 1100.0, "G-League": 900.0},
    )
    gen.fit_transform(df)

    # P1 and P2 are in NBA, P3 is in G-League
    # After match, ratings are updated from start ratings
    # P1 performed 0.6 (above average), so rating increases from 1100
    # P2 performed 0.4 (below average), so rating decreases from 1100
    # P3 performed 0.5 (average), so rating stays close to 900
    assert gen._player_off_ratings["P1"].rating_value > 1100.0  # Updated from NBA start
    assert gen._player_off_ratings["P2"].rating_value < 1100.0  # Updated from NBA start
    # P3 should have started at 900 and updated (could go up or down depending on opponent)
    assert gen._player_off_ratings["P3"].rating_value != 1000.0  # Not default


def test_fit_transform_league_quantile_calculation(base_cn):
    """League quantile calculation when enough players exist."""
    # Create enough players to exceed min_count_for_percentiles (default 50)
    # For testing, we'll use a lower threshold
    player_ids = [f"P{i}" for i in range(60)]
    team_ids = [f"T{(i % 2) + 1}" for i in range(60)]
    match_ids = ["M1"] * 60
    dates = ["2024-01-01"] * 60
    # Create varied performances to establish a distribution
    performances = [0.3 + (i % 10) * 0.05 for i in range(60)]

    df = pl.DataFrame(
        {
            "pid": player_ids,
            "tid": team_ids,
            "mid": match_ids,
            "dt": dates,
            "perf": performances,
            "pw": [1.0] * 60,
            "league": ["NBA"] * 60,
        }
    )
    from dataclasses import replace

    cn = replace(base_cn, league="league")

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn,
        auto_scale_performance=True,
        start_league_quantile=0.2,  # 20th percentile
        start_min_count_for_percentiles=50,  # Threshold met
    )
    gen.fit_transform(df)

    # After processing, new players should use quantile-based rating
    # Since all 60 players are new, they'll all get start ratings
    # The quantile will be calculated from the ratings after first match
    # For a new player in a subsequent match, it should use the quantile
    # Need at least 2 players for transformer (different performance values)
    df2 = pl.DataFrame(
        {
            "pid": ["P99", "P100"],  # Add opponent for match
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-02", "2024-01-02"],
            "perf": [0.5, 0.6],  # Different values for transformer
            "pw": [1.0, 1.0],
            "league": ["NBA", "NBA"],
        }
    )
    gen.fit_transform(df2)

    # P99 should get a quantile-based start rating
    assert gen._player_off_ratings["P99"].rating_value is not None


def test_fit_transform_league_max_days_ago_filtering(base_cn):
    """Filtering by start_max_days_ago_league_entities (only recent players count)."""
    df1 = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01"],
            "perf": [0.6, 0.4],
            "pw": [1.0, 1.0],
            "league": ["NBA", "NBA"],
        }
    )
    from dataclasses import replace

    cn = replace(base_cn, league="league")

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn,
        auto_scale_performance=True,
        start_max_days_ago_league_entities=30,  # Only players from last 30 days
    )
    gen.fit_transform(df1)

    # Add a new player after 200 days (beyond the threshold)
    # Need at least 2 players for transformer (different performance values)
    df2 = pl.DataFrame(
        {
            "pid": ["P3", "P4"],  # Add opponent for match
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-07-20", "2024-07-20"],  # ~200 days later
            "perf": [0.5, 0.6],  # Different values for transformer
            "pw": [1.0, 1.0],
            "league": ["NBA", "NBA"],
        }
    )
    gen.fit_transform(df2)

    # P3 should get default league rating since P1 and P2 are too old
    # (their ratings won't be included in quantile calculation)
    # But after the match, rating gets updated from start value
    # P3 performed 0.5 (average), so rating should be close to 1000
    assert abs(gen._player_off_ratings["P3"].rating_value - 1000.0) < 50.0  # Close to default


def test_fit_transform_team_based_start_rating(base_cn):
    """Team-based start rating when start_team_weight > 0."""
    # First, establish some players on a team
    df1 = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3", "P4"],
            "tid": ["T1", "T1", "T2", "T2"],
            "mid": ["M1", "M1", "M1", "M1"],
            "dt": ["2024-01-01"] * 4,
            "perf": [0.8, 0.7, 0.3, 0.2],  # T1 performs well, T2 poorly
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_team_weight=0.5,  # 50% team, 50% league
        start_team_rating_subtract=80.0,
        start_min_match_count_team_rating=2,  # Need 2+ games per team
    )
    gen.fit_transform(df1)

    # Add a new player to T1 (which has good players)
    df2 = pl.DataFrame(
        {
            "pid": ["P5", "P1", "P2"],  # P5 is new on T1
            "tid": ["T1", "T1", "T2"],
            "mid": ["M2", "M2", "M2"],
            "dt": ["2024-01-02"] * 3,
            "perf": [0.5, 0.8, 0.3],
            "pw": [1.0, 1.0, 1.0],
        }
    )
    gen.fit_transform(df2)

    # P5 should get a blended rating: league_rating * 0.5 + team_rating * 0.5
    # Team rating = (P1_rating + P2_rating) / 2 - 80
    p5_rating = gen._player_off_ratings["P5"].rating_value
    assert p5_rating != 1000.0  # Should be different from default
    assert p5_rating > 0  # Should be positive


def test_fit_transform_team_based_start_rating_empty_team(base_cn):
    """Empty team (no existing players) falls back to league rating."""
    df = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3"],  # T1 has P1, P2; T2 has P3 (opponent)
            "tid": ["T1", "T1", "T2"],
            "mid": ["M1", "M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "perf": [0.6, 0.4, 0.5],  # Different values to avoid transformer error
            "pw": [1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_team_weight=0.5,  # Team weight > 0 but team is empty (no existing players)
    )
    gen.fit_transform(df)

    # Should fall back to league rating (default 1000) since team is empty
    # But after match, ratings get updated
    assert gen._player_off_ratings["P1"].rating_value > 1000.0  # Updated from start


def test_fit_transform_team_based_start_rating_min_match_count(base_cn):
    """start_min_match_count_team_rating threshold works."""
    # Team with only 1 game (below threshold of 2)
    df1 = pl.DataFrame(
        {
            "pid": ["P1", "P2", "P3"],  # T1 has P1, P2; T2 has P3 (opponent)
            "tid": ["T1", "T1", "T2"],
            "mid": ["M1", "M1", "M1"],
            "dt": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "perf": [0.8, 0.2, 0.5],  # Different values to avoid transformer error
            "pw": [1.0, 1.0, 1.0],
        }
    )
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_team_weight=0.5,
        start_min_match_count_team_rating=2,  # Need 2+ games
    )
    gen.fit_transform(df1)

    # Add new player to T1 (team only has 1 game total, below threshold)
    df2 = pl.DataFrame(
        {
            "pid": ["P4", "P5"],  # P4 is new to T1, P5 is opponent
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-02"] * 2,
            "perf": [0.5, 0.6],  # Different values to avoid transformer error
            "pw": [1.0, 1.0],
        }
    )
    gen.fit_transform(df2)

    # P4 should get league rating only (team weight = 0 because threshold not met)
    # After match, rating gets updated from start value
    assert gen._player_off_ratings["P4"].rating_value != 1000.0  # Updated from start
    # Should be close to 1000 since team-based rating is not used (threshold not met)
    assert abs(gen._player_off_ratings["P4"].rating_value - 1000.0) < 50.0


def test_fit_transform_league_change_tracking(base_cn):
    """NOTE: update_players_to_leagues exists but is not currently called automatically.
    This test documents the expected behavior if/when the feature is fully implemented."""
    df1 = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01"] * 2,
            "perf": [0.6, 0.4],
            "pw": [1.0, 1.0],
            "league": ["NBA", "NBA"],
        }
    )
    from dataclasses import replace

    cn = replace(base_cn, league="league")

    gen = PlayerRatingGenerator(
        performance_column="perf", column_names=cn, auto_scale_performance=True
    )
    gen.fit_transform(df1)

    # Currently, update_players_to_leagues is not called automatically,
    # so league tracking doesn't happen. The method exists in StartRatingGenerator
    # but needs to be integrated into the rating update flow.
    # This test verifies the method exists and can be called manually if needed.
    assert hasattr(gen.start_rating_generator, "update_players_to_leagues")

    # League-specific start ratings still work via start_league_ratings parameter
    gen2 = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn,
        auto_scale_performance=True,
        start_league_ratings={"NBA": 1100.0, "G-League": 900.0},
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )

    # Check start rating before match using future_transform
    future_df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-02"] * 2,
            "league": ["NBA"] * 2,
        }
    )
    res = gen2.future_transform(future_df)
    start_rating = res["player_off_rating_perf"][0]
    assert start_rating == 1100.0  # NBA start rating

    # After fit_transform, rating will be updated from start
    gen2.fit_transform(df1)
    assert gen2._player_off_ratings["P1"].rating_value != 1000.0  # Not default
    assert gen2._player_off_ratings["P1"].rating_value > 1100.0  # Updated from NBA start


def test_fit_transform_multiple_league_changes(base_cn):
    """NOTE: League change tracking via update_players_to_leagues is not currently automatic.
    This test verifies that league-specific start ratings work correctly across different leagues.
    """
    from dataclasses import replace

    cn = replace(base_cn, league="league")

    # Test that players get appropriate start ratings based on their current league
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=cn,
        auto_scale_performance=True,
        start_league_ratings={"NBA": 1100.0, "G-League": 900.0, "EuroLeague": 950.0},
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )

    # P1 starts in NBA
    df1 = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-01"] * 2,
            "perf": [0.6, 0.4],
            "pw": [1.0, 1.0],
            "league": ["NBA", "NBA"],
        }
    )
    # Check start rating before match
    future_df = pl.DataFrame(
        {
            "pid": ["P1", "P2"],
            "tid": ["T1", "T2"],
            "mid": ["M1", "M1"],
            "dt": ["2024-01-02"] * 2,
            "league": ["NBA"] * 2,
        }
    )
    gen.future_transform(future_df)

    gen.fit_transform(df1)
    # After match, rating updated from NBA start (1100)
    assert gen._player_off_ratings["P1"].rating_value > 1100.0

    # Check P3's start rating before match (new player in G-League)
    future_df2 = pl.DataFrame(
        {
            "pid": ["P2", "P3"],
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-02"] * 2,
            "league": ["G-League"] * 2,
        }
    )
    res2 = gen.future_transform(future_df2)
    p2_rating = res2["player_off_rating_perf"][0]
    p3_start = res2["player_off_rating_perf"][1]
    assert p2_rating > p3_start
    assert p3_start == 900.0  # G-League start rating
    assert p2_rating < 1100


@pytest.mark.parametrize(
    "features_out,non_predictor_features_out,output_suffix,expected_cols",
    [
        # Test 1: Single known feature, no suffix (defaults to performance_column="perf")
        ([RatingKnownFeatures.PLAYER_OFF_RATING], None, None, ["player_off_rating_perf"]),
        # Test 2: Multiple known features, no suffix (defaults to performance_column="perf")
        (
            [RatingKnownFeatures.PLAYER_OFF_RATING, RatingKnownFeatures.PLAYER_DEF_RATING],
            None,
            None,
            ["player_off_rating_perf", "player_def_rating_perf"],
        ),
        # Test 3: Known and unknown features, no suffix (defaults to performance_column="perf")
        (
            [RatingKnownFeatures.PLAYER_OFF_RATING],
            [RatingUnknownFeatures.PLAYER_PREDICTED_OFF_PERFORMANCE],
            None,
            ["player_off_rating_perf", "player_predicted_off_performance_perf"],
        ),
        # Test 4: Single known feature with suffix
        ([RatingKnownFeatures.PLAYER_OFF_RATING], None, "v2", ["player_off_rating_v2"]),
        # Test 5: Multiple features with suffix
        (
            [RatingKnownFeatures.PLAYER_OFF_RATING, RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED],
            [RatingUnknownFeatures.PLAYER_PREDICTED_OFF_PERFORMANCE],
            "custom",
            [
                "player_off_rating_custom",
                "team_off_rating_projected_custom",
                "player_predicted_off_performance_custom",
            ],
        ),
        # Test 6: Team-level features (defaults to performance_column="perf")
        (
            [
                RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
                RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED,
            ],
            [RatingUnknownFeatures.TEAM_RATING],
            None,
            [
                "team_off_rating_projected_perf",
                "opponent_def_rating_projected_perf",
                "team_rating_perf",
            ],
        ),
        # Test 7: Rating difference features
        (
            [RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED],
            [RatingUnknownFeatures.TEAM_RATING_DIFFERENCE],
            "diff",
            ["team_rating_difference_projected_diff", "team_rating_difference_diff"],
        ),
        # Test 8: Empty features_out (should use defaults, defaults to performance_column="perf")
        (
            None,
            [RatingUnknownFeatures.PLAYER_PREDICTED_OFF_PERFORMANCE],
            None,
            [
                "team_rating_difference_projected_perf",
                "player_predicted_off_performance_perf",
            ],  # Default is RATING_DIFFERENCE_PROJECTED
        ),
    ],
)
def test_player_rating_features_out_combinations(
    base_cn, sample_df, features_out, non_predictor_features_out, output_suffix, expected_cols
):
    """Test that correct features are output for different combinations of features_out, non_predictor_features_out, and suffixes."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        features_out=features_out,
        non_predictor_features_out=non_predictor_features_out,
        output_suffix=output_suffix,
    )
    result = gen.fit_transform(sample_df)

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
def test_player_rating_suffix_applied_to_all_features(base_cn, sample_df, output_suffix):
    """Test that output_suffix is correctly applied to all requested features."""
    features = [
        RatingKnownFeatures.PLAYER_OFF_RATING,
        RatingKnownFeatures.PLAYER_DEF_RATING,
        RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
    ]
    non_predictor = [
        RatingUnknownFeatures.PLAYER_PREDICTED_OFF_PERFORMANCE,
        RatingUnknownFeatures.TEAM_RATING,
    ]

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        features_out=features,
        non_predictor_features_out=non_predictor,
        output_suffix=output_suffix,
    )
    result = gen.fit_transform(sample_df)

    # Build expected column names
    if output_suffix:
        expected_cols = [
            f"player_off_rating_{output_suffix}",
            f"player_def_rating_{output_suffix}",
            f"team_off_rating_projected_{output_suffix}",
            f"player_predicted_off_performance_{output_suffix}",
            f"team_rating_{output_suffix}",
        ]
    else:
        # When output_suffix=None, it defaults to performance column name ("perf")
        expected_cols = [
            "player_off_rating_perf",
            "player_def_rating_perf",
            "team_off_rating_projected_perf",
            "player_predicted_off_performance_perf",
            "team_rating_perf",
        ]

    result_cols = (
        result.columns.tolist() if hasattr(result.columns, "tolist") else list(result.columns)
    )
    for col in expected_cols:
        assert col in result_cols, f"Expected column '{col}' not found. Columns: {result_cols}"


def test_player_rating_only_requested_features_present(base_cn, sample_df):
    """Test that only requested features (and input columns) are present in output."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
        non_predictor_features_out=None,
        output_suffix=None,
    )
    result = gen.fit_transform(sample_df)

    # Should have input columns + requested feature
    input_cols = set(sample_df.columns)
    result_cols = set(result.columns)

    # Check that input columns are preserved
    for col in input_cols:
        assert col in result_cols, f"Input column '{col}' missing from output"

    # Check that requested feature is present (with performance column suffix)
    assert "player_off_rating_perf" in result_cols

    # Check that other rating features are NOT present (unless they're input columns)
    unwanted_features = [
        "player_def_rating",
        "team_off_rating_projected",
        "player_predicted_off_performance",
    ]
    for feature in unwanted_features:
        if feature not in input_cols:
            assert (
                feature not in result_cols
            ), f"Unrequested feature '{feature}' should not be in output"


def test_player_rating_team_with_strong_offense_and_weak_defense_gets_expected_ratings_and_predictions(
    base_cn,
):
    start_rating = 1000.0

    generator = PlayerRatingGenerator(
        auto_scale_performance=True,
        performance_column="team_points",
        column_names=base_cn,
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
            "mid": [
                1,
                1,
                1,
                1,  # team_a vs team_b (high scoring)
                2,
                2,
                2,
                2,  # team_a vs team_c (high scoring)
                3,
                3,
                3,
                3,  # team_b vs team_c (normal)
            ],
            "tid": [
                "team_a",
                "team_a",
                "team_b",
                "team_b",
                "team_a",
                "team_a",
                "team_c",
                "team_c",
                "team_b",
                "team_b",
                "team_c",
                "team_c",
            ],
            "pid": [
                "a_1",
                "a_2",
                "b_1",
                "b_2",
                "a_1",
                "a_2",
                "c_1",
                "c_2",
                "b_1",
                "b_2",
                "c_1",
                "c_2",
            ],
            "dt": [
                base_day,
                base_day,
                base_day,
                base_day,
                base_day + timedelta(days=1),
                base_day + timedelta(days=1),
                base_day + timedelta(days=1),
                base_day + timedelta(days=1),
                base_day + timedelta(days=2),
                base_day + timedelta(days=2),
                base_day + timedelta(days=2),
                base_day + timedelta(days=2),
            ],
            "team_points": [
                140,
                140,
                130,
                130,
                138,
                138,
                128,
                128,
                115,
                115,
                120,
                120,
            ],
        }
    )

    generator.fit_transform(df)

    a_off = float(generator._player_off_ratings["a_1"].rating_value)
    a_def = float(generator._player_def_ratings["a_1"].rating_value)
    assert float(generator._player_off_ratings["a_1"].rating_value) == float(
        generator._player_off_ratings["a_2"].rating_value
    )
    assert float(generator._player_def_ratings["a_1"].rating_value) == float(
        generator._player_def_ratings["a_2"].rating_value
    )

    assert a_off > start_rating
    assert a_def < start_rating


def test_fit_transform__player_rating_difference_from_team_projected_feature(base_cn, sample_df):
    """PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED computes player_off_rating - team_off_rating_projected."""
    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        features_out=[
            RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED,
            RatingKnownFeatures.PLAYER_OFF_RATING,
            RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
        ],
    )
    result = gen.fit_transform(sample_df)

    diff_col = "player_rating_difference_from_team_projected_perf"
    player_col = "player_off_rating_perf"
    team_col = "team_off_rating_projected_perf"

    assert diff_col in result.columns
    assert player_col in result.columns
    assert team_col in result.columns

    for row in result.iter_rows(named=True):
        expected = row[player_col] - row[team_col]
        assert row[diff_col] == pytest.approx(expected, rel=1e-9)


def test_fit_transform__start_league_quantile_uses_existing_player_ratings(base_cn):
    """
    Bug reproduction: start_league_quantile should use percentile of existing player
    ratings for new players, but update_players_to_leagues is never called so
    _league_player_ratings stays empty and all new players get default rating.

    Expected: New player P_NEW should start at 5th percentile of existing ratings (~920)
    Actual: New player starts at default 1000 because _league_player_ratings is empty
    """
    import numpy as np

    num_existing_players = 60
    player_ids = [f"P{i}" for i in range(num_existing_players)]
    team_ids = [f"T{i % 2 + 1}" for i in range(num_existing_players)]

    df1 = pl.DataFrame(
        {
            "pid": player_ids,
            "tid": team_ids,
            "mid": ["M1"] * num_existing_players,
            "dt": ["2024-01-01"] * num_existing_players,
            "perf": [0.3 + (i % 10) * 0.07 for i in range(num_existing_players)],
            "pw": [1.0] * num_existing_players,
        }
    )

    gen = PlayerRatingGenerator(
        performance_column="perf",
        column_names=base_cn,
        auto_scale_performance=True,
        start_league_quantile=0.05,
        start_min_count_for_percentiles=50,
        features_out=[RatingKnownFeatures.PLAYER_OFF_RATING],
    )
    gen.fit_transform(df1)

    existing_ratings = [
        gen._player_off_ratings[pid].rating_value for pid in player_ids
    ]
    expected_quantile_rating = np.percentile(existing_ratings, 5)

    srg = gen.start_rating_generator
    assert len(srg._league_player_ratings.get(None, [])) >= 50, (
        f"Expected _league_player_ratings to have >=50 entries but got "
        f"{len(srg._league_player_ratings.get(None, []))}. "
        "update_players_to_leagues is never called."
    )

    df2 = pl.DataFrame(
        {
            "pid": ["P_NEW", "P0"],
            "tid": ["T1", "T2"],
            "mid": ["M2", "M2"],
            "dt": ["2024-01-02", "2024-01-02"],
            "pw": [1.0, 1.0],
        }
    )
    result = gen.future_transform(df2)

    new_player_start_rating = result.filter(pl.col("pid") == "P_NEW")[
        "player_off_rating_perf"
    ][0]

    assert new_player_start_rating == pytest.approx(expected_quantile_rating, rel=0.1), (
        f"New player should start at 5th percentile ({expected_quantile_rating:.1f}) "
        f"but got {new_player_start_rating:.1f}. "
        "start_league_quantile has no effect because update_players_to_leagues is never called."
    )
