from player_performance_ratings.data_structures import Match, MatchPlayer, MatchPerformance, MatchTeam, \
    PlayerRating
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings import TeamRatingGenerator
from player_performance_ratings.ratings.opponent_adjusted_rating.performance_predictor import MATCH_CONTRIBUTION_TO_SUM_VALUE
from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import OpponentAdjustedRatingGenerator


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
                            )
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=1,
                                participation_weight=0.1,
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
                            )
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0,
                                participation_weight=0.1,
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
                            )
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=0.3,
                                participation_weight=0.2,
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
                            )
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0.7,
                                participation_weight=0.2,
                            )
                        )
                    ]
                )
            ]

        )
    ]

    rating_change_multiplier = 10  # k

    rating_generator = OpponentAdjustedRatingGenerator(
        team_rating_generator=TeamRatingGenerator(
            rating_change_multiplier=rating_change_multiplier,
            confidence_weight=0

        )
    )

    ratings = rating_generator.generate(matches=matches)

    expected_player_game_1_player1 = (0.7 - 0.5) * rating_change_multiplier * 0.1
    expected_player_game_1_player2 = (1 - 0.5) * rating_change_multiplier * 0.1
    expected_player_game_1_player3 = (0.3 - 0.5) * rating_change_multiplier * 0.1
    expected_player_game_1_player4 = (0 - 0.5) * rating_change_multiplier * 0.1
    expected_player_game_2_player1 = (0 - 0.5) * rating_change_multiplier * 0.2
    expected_player_game_2_player2 = (0.3 - 0.5) * rating_change_multiplier * 0.2
    expected_player_game_2_player3 = (1 - 0.5) * rating_change_multiplier * 0.2
    expected_player_game_2_player4 = (0.7 - 0.5) * rating_change_multiplier * 0.2

    assert ratings[RatingColumnNames.PLAYER_RATING_CHANGE] == [
        expected_player_game_1_player1,
        expected_player_game_1_player2,
        expected_player_game_1_player3,
        expected_player_game_1_player4,
        expected_player_game_2_player1,
        expected_player_game_2_player2,
        expected_player_game_2_player3,
        expected_player_game_2_player4
    ]
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
                            )
                        ),
                        MatchPlayer(
                            id="2",
                            performance=MatchPerformance(
                                performance_value=1,
                                participation_weight=0.1,
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
                            )
                        ),
                        MatchPlayer(
                            id="4",
                            performance=MatchPerformance(
                                performance_value=0,
                                participation_weight=0.1,
                            )
                        )
                    ]
                )
            ]

        ),
    ]

    rating_change_multiplier = 10  # k

    rating_generator = OpponentAdjustedRatingGenerator(
        team_rating_generator=TeamRatingGenerator(
            rating_change_multiplier=rating_change_multiplier,
            confidence_weight=0

        )
    )

    ratings = rating_generator.generate(matches=matches)

    expected_rating_change_game_1_player1 = (0.7 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player2 = (1 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player3 = (0.3 - 0.5) * rating_change_multiplier * 0.1
    expected_rating_change_game_1_player4 = (0 - 0.5) * rating_change_multiplier * 0.1


    assert ratings[RatingColumnNames.PLAYER_RATING_CHANGE] == [
        expected_rating_change_game_1_player1,
        expected_rating_change_game_1_player2,
        expected_rating_change_game_1_player3,
        expected_rating_change_game_1_player4,

    ]
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

