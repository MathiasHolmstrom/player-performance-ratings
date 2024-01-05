import pandas as pd

from player_performance_ratings.data_structures import Match, MatchPlayer, MatchPerformance, MatchTeam, \
    PlayerRating, ColumnNames
from player_performance_ratings.ratings.enums import RatingColumnNames, HistoricalRatingColumnNames
from player_performance_ratings.ratings.opponent_adjusted_rating import TeamRatingGenerator, StartRatingGenerator, \
    OpponentAdjustedRatingGenerator
from player_performance_ratings.ratings.opponent_adjusted_rating.performance_predictor import \
    MATCH_CONTRIBUTION_TO_SUM_VALUE



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
                            )
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=1,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            )
                        )
                    ]
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
                            )
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            )
                        )
                    ]
                )
            ]

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
                            )
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=0.3,
                                participation_weight=0.2,
                                projected_participation_weight=0.2,
                            )
                        )
                    ]
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
                            )
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0.7,
                                participation_weight=0.2,
                                projected_participation_weight=0.2,
                            )
                        )
                    ]
                )
            ]

        )
    ]

    rating_change_multiplier = 10
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="won",
    )

    rating_generator = OpponentAdjustedRatingGenerator(
        column_names=column_names,
        team_rating_generator=TeamRatingGenerator(
            rating_change_multiplier=rating_change_multiplier,
            confidence_weight=0

        )
    )

    ratings = rating_generator.generate_historical(matches=matches)

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
        "1": PlayerRating(id="1", rating_value=1000 + expected_player_game_1_player1 + expected_player_game_2_player1,
                          games_played=0.3, confidence_sum=new_confidence_sum),
        "2": PlayerRating(id="2", rating_value=1000 + expected_player_game_1_player2 + expected_player_game_2_player2,
                          games_played=0.3, confidence_sum=new_confidence_sum),
        "3": PlayerRating(id="3", rating_value=1000 + expected_player_game_1_player3 + expected_player_game_2_player3,
                          games_played=0.3, confidence_sum=new_confidence_sum),
        "4": PlayerRating(id="4", rating_value=1000 + expected_player_game_1_player4 + expected_player_game_2_player4,
                          games_played=0.3, confidence_sum=new_confidence_sum),

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


def test_rating_generator_1_match():
    """
    When 1 match where the weighted performance is equal to the prior example with 2 matches per update_id and the sum of particiaption_weight is the same
     --> should return same values as previous test

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
                            )
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=1,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            )
                        )
                    ]
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
                            )
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0,
                                participation_weight=0.1,
                                projected_participation_weight=0.1,
                            )
                        )
                    ]
                )
            ]

        ),
    ]

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="won",
    )

    rating_change_multiplier = 10  # k

    rating_generator = OpponentAdjustedRatingGenerator(
        column_names=column_names,
        team_rating_generator=TeamRatingGenerator(
            rating_change_multiplier=rating_change_multiplier,
            confidence_weight=0

        )
    )

    _ = rating_generator.generate_historical(matches=matches)

    expected_rating_change_game_1_player1 = (0.7 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player2 = (1 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player3 = (0.3 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player4 = (0 - 0.5) * rating_change_multiplier * 0.1

    new_confidence_sum = MATCH_CONTRIBUTION_TO_SUM_VALUE * 0.1
    expected_player_ratings = {
        "1": PlayerRating(id="1", rating_value=1000 + expected_rating_change_game_1_player1,
                          games_played=0.1, confidence_sum=new_confidence_sum),
        "2": PlayerRating(id="2", rating_value=1000 + expected_rating_change_game_1_player2,
                          games_played=0.1, confidence_sum=new_confidence_sum),
        "3": PlayerRating(id="3", rating_value=1000 + expected_rating_change_game_1_player3,
                          games_played=0.1, confidence_sum=new_confidence_sum),
        "4": PlayerRating(id="4", rating_value=1000 + expected_rating_change_game_1_player4,
                          games_played=0.1, confidence_sum=new_confidence_sum),

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
        performance="won",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )

    df = pd.DataFrame({
        column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
        column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2],
        column_names.player_id: [1, 2, 3, 4, 1, 2, 3, 4],
        column_names.start_date: [pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01"),
                                  pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01"),
                                  pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-02"),
                                  pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-02")
                                  ],
        column_names.performance: [1, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5],
        column_names.projected_participation_weight: [1, 1, 1, 1, 0.2, 1, 0.6, 0.6],
        column_names.participation_weight: [1, 1, 1, 1, 1, 1, 1, 1],
    })

    rating_generator = OpponentAdjustedRatingGenerator(
        column_names=column_names,

        features_out=[RatingColumnNames.TEAM_RATING_PROJECTED],
        team_rating_generator=TeamRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(
                harcoded_start_rating=1000
            )
        )
    )
    _ = rating_generator.generate_historical(df=df)

    assert rating_generator.ratings_df[RatingColumnNames.TEAM_RATING_PROJECTED].iloc[4] == \
           rating_generator.ratings_df[RatingColumnNames.TEAM_RATING_PROJECTED].iloc[5]
    assert rating_generator.ratings_df[RatingColumnNames.TEAM_RATING_PROJECTED].iloc[6] == \
           rating_generator.ratings_df[RatingColumnNames.TEAM_RATING_PROJECTED][7]
    assert rating_generator.ratings_df[RatingColumnNames.TEAM_RATING_PROJECTED].iloc[4] < \
           rating_generator.ratings_df[HistoricalRatingColumnNames.TEAM_RATING][4]


def test_test_opponent_adjusted_rating_generator_with_projected_performance_features_out():
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="won",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )

    df = pd.DataFrame({
        column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
        column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2],
        column_names.player_id: [1, 2, 3, 4, 1, 2, 3, 4],
        column_names.start_date: [pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01"),
                                  pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01"),
                                  pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-02"),
                                  pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-02")
                                  ],
        column_names.performance: [1, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5],
        column_names.projected_participation_weight: [1, 1, 1, 1, 0.2, 1, 0.6, 0.6],
        column_names.participation_weight: [1, 1, 1, 1, 1, 1, 1, 1],
    })

    rating_generator = OpponentAdjustedRatingGenerator(
        column_names=column_names,
        features_out=[RatingColumnNames.TEAM_RATING_PROJECTED, RatingColumnNames.PLAYER_RATING,
                      RatingColumnNames.RATING_DIFFERENCE_PROJECTED,
                      RatingColumnNames.PLAYER_RATING_DIFFERENCE_PROJECTED,
                      RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED],
        team_rating_generator=TeamRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(
                harcoded_start_rating=1000
            )
        )
    )
    ratings = rating_generator.generate_historical(df=df)
    assert len(rating_generator.features_out) == len(ratings)


def test_opponent_adjusted_rating_generator_historical_and_future():
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="won",
        projected_participation_weight="projected_participation_weight",
        participation_weight="participation_weight",
    )

    historical_df = pd.DataFrame({
        column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
        column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2],
        column_names.player_id: [1, 2, 3, 4, 1, 2, 3, 4],
        column_names.start_date: [pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01"),
                                  pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01"),
                                  pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-02"),
                                  pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-02")
                                  ],
        column_names.performance: [1, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5],
        column_names.projected_participation_weight: [1, 1, 1, 1, 0.2, 1, 0.6, 0.6],
        column_names.participation_weight: [1, 1, 1, 1, 1, 1, 1, 1],
    })

    future_df = pd.DataFrame(
        {
            column_names.match_id: [3, 3, 3, 3],
            column_names.team_id: [1, 1, 2, 2],
            column_names.player_id: [1, 2, 3, 4],
            column_names.start_date: [pd.to_datetime("2022-01-05"), pd.to_datetime("2022-01-05"),
                                      pd.to_datetime("2022-01-05"), pd.to_datetime("2022-01-05"),
                                      ],
            column_names.projected_participation_weight: [1, 1, 1, 1],
        }
    )

    rating_generator = OpponentAdjustedRatingGenerator(
        column_names=column_names,
        features_out=[RatingColumnNames.TEAM_RATING_PROJECTED,
                      RatingColumnNames.PLAYER_RATING,
                      RatingColumnNames.RATING_DIFFERENCE_PROJECTED,
                      RatingColumnNames.PLAYER_RATING_DIFFERENCE_PROJECTED,
                      RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED],
        team_rating_generator=TeamRatingGenerator(
            confidence_weight=0,
            start_rating_generator=StartRatingGenerator(
                harcoded_start_rating=1000
            )
        )
    )
    _ = rating_generator.generate_historical(df=historical_df)
    player_ratings = rating_generator.player_ratings
    future_ratings = rating_generator.generate_future(df=future_df)

    player_rating_1 = player_ratings[1].rating_value
    player_rating_2 = player_ratings[2].rating_value
    player_rating_3 = player_ratings[3].rating_value
    player_rating_4 = player_ratings[4].rating_value

    team_rating1 = player_rating_1 * 0.5 + player_rating_2 * 0.5
    team_rating2 = player_rating_3 * 0.5 + player_rating_4 * 0.5

    expected_future_ratings = {
        RatingColumnNames.TEAM_RATING_PROJECTED: [team_rating1, team_rating1, team_rating2, team_rating2],
        RatingColumnNames.PLAYER_RATING: [player_rating_1, player_rating_2, player_rating_3, player_rating_4],
        RatingColumnNames.RATING_DIFFERENCE_PROJECTED: [team_rating1 - team_rating2, team_rating1 - team_rating2,
                                                        team_rating2 - team_rating1, team_rating2 - team_rating1],
        RatingColumnNames.PLAYER_RATING_DIFFERENCE_PROJECTED: [player_rating_1 - team_rating2,
                                                               player_rating_2 - team_rating2,
                                                               player_rating_3 - team_rating1,
                                                               player_rating_4 - team_rating1],
        RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED: [player_rating_1 - team_rating1,
                                                                         player_rating_2 - team_rating1,
                                                                         player_rating_3 - team_rating2,
                                                                         player_rating_4 - team_rating2],
    }

    assert future_ratings == expected_future_ratings
