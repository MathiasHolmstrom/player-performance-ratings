import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from player_performance_ratings.data_structures import (
    Match,
    MatchPlayer,
    MatchPerformance,
    MatchTeam,
    PlayerRating,
    ColumnNames,
)
from player_performance_ratings.ratings import UpdateRatingGenerator
from player_performance_ratings.ratings.enums import (
    RatingKnownFeatures,
    RatingUnknownFeatures,
)
from player_performance_ratings.ratings.rating_calculators import (
    MatchRatingGenerator,
    StartRatingGenerator,
)


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )




@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_opponent_adjusted_rating_generator_with_projected_performance(df):
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )

    rating_generator = UpdateRatingGenerator(
        unknown_features_out=[RatingUnknownFeatures.TEAM_RATING],
        features_out=[RatingKnownFeatures.TEAM_RATING_PROJECTED],
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(harcoded_start_rating=1000),
        ),
    )

    data = df(
        {
            column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
            column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2],
            column_names.player_id: [1, 2, 3, 4, 1, 2, 3, 4],
            column_names.start_date: [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
            rating_generator.performance_column: [
                1.0,
                0.5,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            column_names.projected_participation_weight: [
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                1,
                0.6,
                0.6,
            ],
            column_names.participation_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    df_with_ratings = rating_generator.generate_historical(
        df=data,
        column_names=column_names,
    )

    if isinstance(df_with_ratings, pl.DataFrame):
        df_with_ratings = df_with_ratings.to_pandas()

    assert (
        df_with_ratings[RatingKnownFeatures.TEAM_RATING_PROJECTED].iloc[4]
        == df_with_ratings[RatingKnownFeatures.TEAM_RATING_PROJECTED].iloc[5]
    )
    assert (
        df_with_ratings[RatingKnownFeatures.TEAM_RATING_PROJECTED].iloc[6]
        == df_with_ratings[RatingKnownFeatures.TEAM_RATING_PROJECTED][7]
    )
    assert (
        df_with_ratings[RatingKnownFeatures.TEAM_RATING_PROJECTED].iloc[4]
        < df_with_ratings[RatingUnknownFeatures.TEAM_RATING][4]
    )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_update_rating_generator_generate_historical(df):
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )
    rating_generator = UpdateRatingGenerator(
        features_out=[
            RatingKnownFeatures.TEAM_RATING_PROJECTED,
            RatingKnownFeatures.PLAYER_RATING,
            RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED,
            RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED,
            RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED,
        ],
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(harcoded_start_rating=1000),
        ),
        unknown_features_out=[RatingUnknownFeatures.PLAYER_RATING_CHANGE],
    )
    data = df(
        {
            column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
            column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2],
            column_names.player_id: [1, 2, 3, 4, 1, 2, 3, 4],
            column_names.start_date: [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
            rating_generator.performance_column: [
                1.0,
                0.5,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            column_names.projected_participation_weight: [
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                1,
                0.6,
                0.6,
            ],
            column_names.participation_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    ratings_df = rating_generator.generate_historical(
        df=data, column_names=column_names
    )
    cols = [
        *data.columns,
        *rating_generator.unknown_features_out,
        *rating_generator._features_out,
    ]
    for col in cols:
        assert col in ratings_df.columns

    assert len(ratings_df.columns) == len(cols)


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_update_rating_generator_historical_and_future(df):
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )
    rating_generator = UpdateRatingGenerator(
        features_out=[
            RatingKnownFeatures.TEAM_RATING_PROJECTED,
            RatingKnownFeatures.PLAYER_RATING,
            RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED,
            RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED,
            RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED,
        ],
        unknown_features_out=[RatingUnknownFeatures.PLAYER_RATING_CHANGE],
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(harcoded_start_rating=1000),
        ),
    )
    historical_df = df(
        {
            column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
            column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2],
            column_names.player_id: [1, 2, 3, 4, 1, 2, 3, 4],
            column_names.start_date: [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
            rating_generator.performance_column: [
                1.0,
                0.5,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            column_names.projected_participation_weight: [
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                1,
                0.6,
                0.6,
            ],
            column_names.participation_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    future_df = df(
        {
            column_names.match_id: [3, 3, 3, 3],
            column_names.team_id: [1, 1, 2, 2],
            column_names.player_id: [1, 2, 3, 4],
            column_names.start_date: [
                pd.to_datetime("2022-01-05"),
                pd.to_datetime("2022-01-05"),
                pd.to_datetime("2022-01-05"),
                pd.to_datetime("2022-01-05"),
            ],
            column_names.projected_participation_weight: [1.0, 1.0, 1.0, 1.0],
        }
    )

    _ = rating_generator.generate_historical(
        df=historical_df, column_names=column_names
    )
    player_ratings = rating_generator.player_ratings
    future_df_with_ratings = rating_generator.generate_future(df=future_df)

    player_rating_1 = player_ratings[1].rating_value
    player_rating_2 = player_ratings[2].rating_value
    player_rating_3 = player_ratings[3].rating_value
    player_rating_4 = player_ratings[4].rating_value

    team_rating1 = player_rating_1 * 0.5 + player_rating_2 * 0.5
    team_rating2 = player_rating_3 * 0.5 + player_rating_4 * 0.5

    expected_future_ratings = {
        RatingKnownFeatures.TEAM_RATING_PROJECTED: [
            team_rating1,
            team_rating1,
            team_rating2,
            team_rating2,
        ],
        RatingKnownFeatures.PLAYER_RATING: [
            player_rating_1,
            player_rating_2,
            player_rating_3,
            player_rating_4,
        ],
        RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED: [
            team_rating1 - team_rating2,
            team_rating1 - team_rating2,
            team_rating2 - team_rating1,
            team_rating2 - team_rating1,
        ],
        RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED: [
            player_rating_1 - team_rating2,
            player_rating_2 - team_rating2,
            player_rating_3 - team_rating1,
            player_rating_4 - team_rating1,
        ],
        RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED: [
            player_rating_1 - team_rating1,
            player_rating_2 - team_rating1,
            player_rating_3 - team_rating2,
            player_rating_4 - team_rating2,
        ],
        RatingUnknownFeatures.PLAYER_RATING_CHANGE: [np.nan, np.nan, np.nan, np.nan],
    }
    if isinstance(future_df, pl.DataFrame):
        expected_future_df = future_df.hstack(pl.from_dict(expected_future_ratings))
        assert_frame_equal(
            future_df_with_ratings,
            expected_future_df.select(future_df_with_ratings.columns),
            check_dtype=False,
        )
    else:
        expected_future_df = future_df.assign(**expected_future_ratings)
        pd.testing.assert_frame_equal(
            future_df_with_ratings,
            expected_future_df,
            check_dtype=False,
            check_like=True,
        )

@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_update_rating_generator_stores_correct(df):
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = UpdateRatingGenerator()
    historical_df1 = df(
        {
            column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
            column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2],
            column_names.player_id: [1, 2, 3, 4, 1, 2, 3, 4],
            column_names.start_date: [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
            rating_generator.performance_column: [1.0, 1.0, 0, 0, 1.0, 1.0, 0, 0],
        }
    )

    historical_df2 = df(
        {
            column_names.match_id: [ 2, 2, 2, 2, 3, 3, 3, 3],
            column_names.team_id: [1, 1, 2, 2, 1, 1, 3,3],
            column_names.player_id: [ 1, 2, 3, 4, 1,2,5,6],
            column_names.start_date: [
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-03"),
                pd.to_datetime("2021-01-03"),
                pd.to_datetime("2021-01-03"),
                pd.to_datetime("2021-01-03"),
            ],
            rating_generator.performance_column: [1.0,1.0,0,0, 1.0,1.0,0,0],
        }
    )

    hist_ratings1 = rating_generator.generate_historical(historical_df1, column_names=column_names)

    if isinstance(hist_ratings1, pl.DataFrame):
        expected_rating_difference_game2 = hist_ratings1.filter(pl.col(column_names.match_id)==2)[rating_generator.features_out[0]].head(1).item()
    else:
        expected_rating_difference_game2 = hist_ratings1[hist_ratings1[column_names.match_id]==2][rating_generator.features_out[0]].iloc[0]

    hist_ratings2 = rating_generator.generate_historical(historical_df2, column_names=column_names)
    if isinstance(hist_ratings1, pl.DataFrame):
        assert hist_ratings2[rating_generator.features_out[0]].head(1).item() == expected_rating_difference_game2
    else:
        assert hist_ratings2[rating_generator.features_out[0]].iloc[0] == expected_rating_difference_game2

    for f in rating_generator._features_out:
        assert f in hist_ratings1.columns
    for f in rating_generator.unknown_features_out:
        assert f in hist_ratings2.columns

    if isinstance(historical_df1, pl.DataFrame):
        hist_ratings = pl.concat([hist_ratings1, hist_ratings2]).unique([column_names.match_id, column_names.player_id])
        hist_ratings = hist_ratings.sort([column_names.start_date, column_names.match_id, column_names.team_id ,column_names.player_id])

        assert_frame_equal(rating_generator.historical_df[rating_generator.features_out], hist_ratings[rating_generator.features_out])
    else:
        hist_ratings = pd.concat([hist_ratings1, hist_ratings2]).drop_duplicates([column_names.match_id, column_names.player_id])
        pd.testing.assert_frame_equal(rating_generator.historical_df[rating_generator.features_out].reset_index(drop=True), hist_ratings[rating_generator.features_out].reset_index(drop=True))

@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_rating_generator_prefix_suffix(df):

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        league='league'
    )
    rating_generator = UpdateRatingGenerator(prefix="prefix_", suffix="_suffix", unknown_features_out=[
        RatingUnknownFeatures.PLAYER_RATING_CHANGE,
        RatingUnknownFeatures.PERFORMANCE,
        RatingUnknownFeatures.RATING_DIFFERENCE,
        RatingUnknownFeatures.OPPONENT_RATING,
        RatingUnknownFeatures.RATING_MEAN
    ], non_estimator_known_features_out=[
        RatingKnownFeatures.TEAM_RATING_PROJECTED,
        RatingKnownFeatures.PLAYER_RATING,
        RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED,
        RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED,
        RatingKnownFeatures.TEAM_LEAGUE,
        RatingKnownFeatures.PLAYER_LEAGUE,
        RatingKnownFeatures.OPPONENT_LEAGUE,
        RatingKnownFeatures.OPPONENT_RATING_PROJECTED
    ])
    historical_df1 = df(
        {
            column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
            column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2],
            column_names.player_id: [1, 2, 3, 4, 1, 2, 3, 4],
            column_names.league: ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b'],
            column_names.start_date: [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
            rating_generator.performance_column: [1.0, 1.0, 0, 0, 1.0, 1.0, 0, 0],
        }
    )

    historical_df1_with_ratings = rating_generator.generate_historical(historical_df1, column_names=column_names)
    for non_estimator_known_features_out in rating_generator._non_estimator_known_features_out:
        expected_feature_name_out = rating_generator.prefix + non_estimator_known_features_out + rating_generator.suffix
        assert expected_feature_name_out in historical_df1_with_ratings.columns

    for unknown_features_out in rating_generator._unknown_features_out:
        expected_feature_name_out = rating_generator.prefix + unknown_features_out + rating_generator.suffix
        assert expected_feature_name_out in historical_df1_with_ratings.columns

    for f in rating_generator._features_out:
        expected_feature_out  = rating_generator.prefix + f + rating_generator.suffix
        assert expected_feature_out in historical_df1_with_ratings.columns


