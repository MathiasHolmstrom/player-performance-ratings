import numpy as np
import pandas as pd
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
    RatingFutureFeatures,
    RatingHistoricalFeatures,
)
from player_performance_ratings.ratings.rating_calculators import (
    MatchRatingGenerator,
    StartRatingGenerator,
)

from player_performance_ratings.ratings.rating_calculators.performance_predictor import (
    MATCH_CONTRIBUTION_TO_SUM_VALUE,
)


def test_rating_generator_update_id_different_from_match_id():
    """
    When 2 matches with the same update_id but different match_ids, then rating_update should only take place after both matches are finished.
    This means that the rating_change (expected performance) is based on the start rating of the players which defaults to 1000.
    Thus expected_performance for all matches will be 0.5.
    The rating_change for each match per player should then be (performance-0.5)*k*performance_weight for each match.
     (since confidence_weight is set 0 which means only rating_change_multiplier determines magnitude of rating change)
     The total rating_change across the update_id should be the sum of all the rating_changes for each player.

    """

    matches = [
        Match(
            id="1",
            update_id="1",
            day_number=2,
            teams=[
                MatchTeam(
                    id="1",
                    players=[
                        MatchPlayer(
                            id="1",
                            performance=MatchPerformance(
                                performance_value=0.7,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            ),
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=1,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            ),
                        ),
                    ],
                ),
                MatchTeam(
                    id="2",
                    players=[
                        MatchPlayer(
                            id="3",
                            performance=MatchPerformance(
                                performance_value=0.3,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            ),
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            ),
                        ),
                    ],
                ),
            ],
        ),
        Match(
            id="2",
            update_id="1",
            day_number=2,
            teams=[
                MatchTeam(
                    id="1",
                    players=[
                        MatchPlayer(
                            id="1",
                            performance=MatchPerformance(
                                performance_value=0,
                                participation_weight=0.2,
                                projected_participation_weight=0.2,
                            ),
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=0.3,
                                participation_weight=0.2,
                                projected_participation_weight=0.2,
                            ),
                        ),
                    ],
                ),
                MatchTeam(
                    id="2",
                    players=[
                        MatchPlayer(
                            id="3",
                            performance=MatchPerformance(
                                performance_value=1,
                                participation_weight=0.2,
                                projected_participation_weight=0.2,
                            ),
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0.7,
                                participation_weight=0.2,
                                projected_participation_weight=0.2,
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ]

    rating_change_multiplier = 10
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )

    rating_generator = UpdateRatingGenerator(
        match_rating_generator=MatchRatingGenerator(
            rating_change_multiplier=rating_change_multiplier, confidence_weight=0
        )
    )

    ratings = rating_generator.generate_historical_by_matches(
        matches=matches, column_names=column_names
    )

    expected_player_game_1_player1 = (0.7 - 0.5) * rating_change_multiplier * 0.1
    expected_player_game_1_player2 = (1 - 0.5) * rating_change_multiplier * 0.1
    expected_player_game_1_player3 = (0.3 - 0.5) * rating_change_multiplier * 0.1
    expected_player_game_1_player4 = (0 - 0.5) * rating_change_multiplier * 0.1
    expected_player_game_2_player1 = (0 - 0.5) * rating_change_multiplier * 0.2
    expected_player_game_2_player2 = (0.3 - 0.5) * rating_change_multiplier * 0.2
    expected_player_game_2_player3 = (1 - 0.5) * rating_change_multiplier * 0.2
    expected_player_game_2_player4 = (0.7 - 0.5) * rating_change_multiplier * 0.2

    new_confidence_sum = MATCH_CONTRIBUTION_TO_SUM_VALUE * 0.3
    expected_player_ratings = {
        "1": PlayerRating(
            id="1",
            rating_value=1000
            + expected_player_game_1_player1
            + expected_player_game_2_player1,
            games_played=0.3,
            confidence_sum=new_confidence_sum,
        ),
        "2": PlayerRating(
            id="2",
            rating_value=1000
            + expected_player_game_1_player2
            + expected_player_game_2_player2,
            games_played=0.3,
            confidence_sum=new_confidence_sum,
        ),
        "3": PlayerRating(
            id="3",
            rating_value=1000
            + expected_player_game_1_player3
            + expected_player_game_2_player3,
            games_played=0.3,
            confidence_sum=new_confidence_sum,
        ),
        "4": PlayerRating(
            id="4",
            rating_value=1000
            + expected_player_game_1_player4
            + expected_player_game_2_player4,
            games_played=0.3,
            confidence_sum=new_confidence_sum,
        ),
    }

    def is_close(a, b, tolerance=1e-5):
        return abs(a - b) <= tolerance

    # Then, in your test:
    for player_id, rating in rating_generator.player_ratings.items():
        expected_rating = expected_player_ratings[player_id]
        assert rating.id == expected_rating.id
        assert rating.rating_value == expected_rating.rating_value
        assert is_close(rating.games_played, expected_rating.games_played)
        assert is_close(rating.confidence_sum, expected_rating.confidence_sum)


def get_single_matches() -> list[Match]:
    return [
        Match(
            id="1",
            update_id="1",
            day_number=2,
            teams=[
                MatchTeam(
                    id="1",
                    players=[
                        MatchPlayer(
                            id="1",
                            performance=MatchPerformance(
                                performance_value=0.7,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            ),
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=1,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            ),
                        ),
                    ],
                ),
                MatchTeam(
                    id="2",
                    players=[
                        MatchPlayer(
                            id="3",
                            performance=MatchPerformance(
                                performance_value=0.3,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            ),
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ]


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )


def test_update_rating_generator_adds_correct_columns(column_names):
    matches = get_single_matches()
    rating_generator = UpdateRatingGenerator()
    ratings = rating_generator.generate_historical_by_matches(
        matches=matches,
        column_names=column_names,
        historical_features_out=[RatingHistoricalFeatures.PLAYER_RATING_CHANGE],
        future_features_out=[RatingFutureFeatures.PLAYER_RATING],
    )
    assert RatingHistoricalFeatures.PLAYER_RATING_CHANGE in ratings
    assert RatingFutureFeatures.PLAYER_RATING in ratings


def test_rating_generator_1_match(column_names):
    """
    When 1 match where the weighted performance is equal to the prior example with 2 matches per update_id and the sum of particiaption_weight is the same
     --> should return same values as previous test

    """
    matches = get_single_matches()

    rating_change_multiplier = 10  # k

    rating_generator = UpdateRatingGenerator(
        match_rating_generator=MatchRatingGenerator(
            rating_change_multiplier=rating_change_multiplier, confidence_weight=0
        )
    )

    _ = rating_generator.generate_historical_by_matches(
        matches=matches, column_names=column_names
    )

    expected_rating_change_game_1_player1 = (0.7 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player2 = (1 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player3 = (0.3 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player4 = (0 - 0.5) * rating_change_multiplier * 0.1

    new_confidence_sum = MATCH_CONTRIBUTION_TO_SUM_VALUE * 0.1
    expected_player_ratings = {
        "1": PlayerRating(
            id="1",
            rating_value=1000 + expected_rating_change_game_1_player1,
            games_played=0.1,
            confidence_sum=new_confidence_sum,
        ),
        "2": PlayerRating(
            id="2",
            rating_value=1000 + expected_rating_change_game_1_player2,
            games_played=0.1,
            confidence_sum=new_confidence_sum,
        ),
        "3": PlayerRating(
            id="3",
            rating_value=1000 + expected_rating_change_game_1_player3,
            games_played=0.1,
            confidence_sum=new_confidence_sum,
        ),
        "4": PlayerRating(
            id="4",
            rating_value=1000 + expected_rating_change_game_1_player4,
            games_played=0.1,
            confidence_sum=new_confidence_sum,
        ),
    }

    def is_close(a, b, tolerance=1e-5):
        return abs(a - b) <= tolerance

    # Then, in your test:
    for player_id, rating in rating_generator.player_ratings.items():
        expected_rating = expected_player_ratings[player_id]
        assert rating.id == expected_rating.id
        assert rating.rating_value == expected_rating.rating_value
        assert is_close(rating.games_played, expected_rating.games_played)
        assert is_close(rating.confidence_sum, expected_rating.confidence_sum)


def test_opponent_adjusted_rating_generator_with_projected_performance():
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )

    rating_generator = UpdateRatingGenerator(
        future_features_out=[RatingFutureFeatures.TEAM_RATING_PROJECTED],
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(harcoded_start_rating=1000),
        ),
    )

    df = pd.DataFrame(
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
                1,
                0.5,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            column_names.projected_participation_weight: [1, 1, 1, 1, 0.2, 1, 0.6, 0.6],
            column_names.participation_weight: [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    df_with_ratings = rating_generator.generate_historical(
        df=df,
        column_names=column_names,
        historical_features_out=[RatingHistoricalFeatures.TEAM_RATING],
    )

    assert (
        df_with_ratings[RatingFutureFeatures.TEAM_RATING_PROJECTED].iloc[4]
        == df_with_ratings[RatingFutureFeatures.TEAM_RATING_PROJECTED].iloc[5]
    )
    assert (
        df_with_ratings[RatingFutureFeatures.TEAM_RATING_PROJECTED].iloc[6]
        == df_with_ratings[RatingFutureFeatures.TEAM_RATING_PROJECTED][7]
    )
    assert (
        df_with_ratings[RatingFutureFeatures.TEAM_RATING_PROJECTED].iloc[4]
        < df_with_ratings[RatingHistoricalFeatures.TEAM_RATING][4]
    )


def test_update_rating_generator_generate_historical():
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )
    rating_generator = UpdateRatingGenerator(
        future_features_out=[
            RatingFutureFeatures.TEAM_RATING_PROJECTED,
            RatingFutureFeatures.PLAYER_RATING,
            RatingFutureFeatures.RATING_DIFFERENCE_PROJECTED,
            RatingFutureFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED,
            RatingFutureFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED,
        ],
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(harcoded_start_rating=1000),
        ),
        historical_features_out=[RatingHistoricalFeatures.PLAYER_RATING_CHANGE],
    )
    df = pd.DataFrame(
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
                1,
                0.5,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            column_names.projected_participation_weight: [1, 1, 1, 1, 0.2, 1, 0.6, 0.6],
            column_names.participation_weight: [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    ratings_df = rating_generator.generate_historical(df=df, column_names=column_names)
    cols = [
        *df.columns.tolist(),
        *rating_generator._historical_features_out,
        *rating_generator.future_features_out,
    ]
    for col in cols:
        assert col in ratings_df.columns

    assert len(ratings_df.columns) == len(cols)


def test_update_rating_generator_historical_and_future():
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )
    rating_generator = UpdateRatingGenerator(
        future_features_out=[
            RatingFutureFeatures.TEAM_RATING_PROJECTED,
            RatingFutureFeatures.PLAYER_RATING,
            RatingFutureFeatures.RATING_DIFFERENCE_PROJECTED,
            RatingFutureFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED,
            RatingFutureFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED,
        ],
        historical_features_out=[RatingHistoricalFeatures.PLAYER_RATING_CHANGE],
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(harcoded_start_rating=1000),
        ),
    )
    historical_df = pd.DataFrame(
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
                1,
                0.5,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            column_names.projected_participation_weight: [1, 1, 1, 1, 0.2, 1, 0.6, 0.6],
            column_names.participation_weight: [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    future_df = pd.DataFrame(
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
            column_names.projected_participation_weight: [1, 1, 1, 1],
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
        RatingFutureFeatures.TEAM_RATING_PROJECTED: [
            team_rating1,
            team_rating1,
            team_rating2,
            team_rating2,
        ],
        RatingFutureFeatures.PLAYER_RATING: [
            player_rating_1,
            player_rating_2,
            player_rating_3,
            player_rating_4,
        ],
        RatingFutureFeatures.RATING_DIFFERENCE_PROJECTED: [
            team_rating1 - team_rating2,
            team_rating1 - team_rating2,
            team_rating2 - team_rating1,
            team_rating2 - team_rating1,
        ],
        RatingFutureFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED: [
            player_rating_1 - team_rating2,
            player_rating_2 - team_rating2,
            player_rating_3 - team_rating1,
            player_rating_4 - team_rating1,
        ],
        RatingFutureFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED: [
            player_rating_1 - team_rating1,
            player_rating_2 - team_rating1,
            player_rating_3 - team_rating2,
            player_rating_4 - team_rating2,
        ],
        RatingHistoricalFeatures.PLAYER_RATING_CHANGE: [np.nan, np.nan, np.nan, np.nan],
    }
    expected_future_df = future_df.assign(**expected_future_ratings)
    pd.testing.assert_frame_equal(
        future_df_with_ratings, expected_future_df, check_dtype=False, check_like=True
    )
