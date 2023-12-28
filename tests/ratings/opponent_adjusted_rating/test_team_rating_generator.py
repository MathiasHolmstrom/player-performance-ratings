import copy

import mock
import pytest

from player_performance_ratings.data_structures import MatchTeam, PreMatchPlayerRating, MatchPerformance, \
    MatchPlayer, PreMatchTeamRating, PlayerRating, TeamRatingChange, PlayerRatingChange
from player_performance_ratings.ratings.opponent_adjusted_rating.team_rating_generator import TeamRatingGenerator
from player_performance_ratings.ratings.opponent_adjusted_rating.performance_predictor import RatingDifferencePerformancePredictor, \
    MATCH_CONTRIBUTION_TO_SUM_VALUE
from player_performance_ratings.ratings.opponent_adjusted_rating.team_rating_generator import EXPECTED_MEAN_CONFIDENCE_SUM


def test_generate_pre_match_team_rating():
    """
    When multiple players play for a team with different participant_weight, rating_value should be calculated according to weights.

    """

    day_number = 1
    match_team = MatchTeam(
        id="1",
        players=[
            MatchPlayer(
                id="1",
                performance=MatchPerformance(
                    performance_value=1,
                    participation_weight=0.5,
                    projected_participation_weight=0.5,
                ),
            ),
            MatchPlayer(
                id="2",
                performance=MatchPerformance(
                    performance_value=1,
                    participation_weight=0.3,
                    projected_participation_weight=0.3,
                ),
            ),
        ],
    )

    team_rating_generator = TeamRatingGenerator()

    team_rating_generator.player_ratings = {
        "1": PlayerRating(
            id="1",
            rating_value=1100,
            games_played=0,
            confidence_sum=0,
            prev_rating_changes=[],
        ),
        "2": PlayerRating(
            id="2",
            rating_value=900,
            games_played=0,
            confidence_sum=0,
            prev_rating_changes=[],
        )
    }

    pre_match_team_rating = team_rating_generator.generate_pre_match_team_rating(
        day_number=day_number,
        match_team=match_team,
    )

    expected_pre_match_team_rating = PreMatchTeamRating(
        id="1",
        league=None,
        players=[
            PreMatchPlayerRating(
                id="1",
                rating_value=1100,
                games_played=0,
                league=None,
                match_performance=MatchPerformance(
                    performance_value=1,
                    participation_weight=0.5,
                    projected_participation_weight=0.5,
                ),
                position=None
            ),
            PreMatchPlayerRating(
                id="2",
                rating_value=900,
                games_played=0,
                league=None,
                match_performance=MatchPerformance(
                    performance_value=1,
                    participation_weight=0.3,
                    projected_participation_weight=0.3,
                ),
                position=None
            ),
        ],
        rating_value=(1100 * 0.5 + 900 * 0.3) / (0.5 + 0.3),
    )
    assert pre_match_team_rating == expected_pre_match_team_rating


@pytest.mark.parametrize("confidence_sum", [0, 25, 50, 75, 100])
def test_generate_rating_change(confidence_sum):
    """
    When players have confidence_sum over 0 and confidence_weight == 0
    --> rating_change_multiplier should be equal to confidence_change_multiplier
    --> rating_change_value for each player should be calculated according to adjusted rating_change_multiplier, performance and predicted_performance
    --> rating_change_value for the team should be weighted across players according to participation_weight

    """

    performance_predictor = RatingDifferencePerformancePredictor(rating_diff_coef=0.005757,
                                                                 rating_diff_team_from_entity_coef=0.0,
                                                                 team_rating_diff_coef=0.0)

    team_rating_generator = TeamRatingGenerator(
        confidence_weight=1,
        confidence_value_denom=50,
        confidence_max_days=100,
        rating_change_multiplier=10,
        confidence_days_ago_multiplier=0.06,
        performance_predictor=performance_predictor
    )

    team_rating_generator.player_ratings = {
        "1": PlayerRating(
            id="1",
            rating_value=1100,
            games_played=0,
            confidence_sum=confidence_sum,
            prev_rating_changes=[],
        ),
        "2": PlayerRating(
            id="2",
            rating_value=900,
            games_played=0,
            confidence_sum=confidence_sum,
            prev_rating_changes=[],
        ),
        "3": PlayerRating(
            id="3",
            rating_value=1100,
            games_played=0,
            confidence_sum=confidence_sum,
            prev_rating_changes=[],
        ),
        "4": PlayerRating(
            id="4",
            rating_value=1100,
            games_played=0,
            confidence_sum=confidence_sum,
            prev_rating_changes=[],
        )
    }

    pre_match_team_ratings = [
        PreMatchTeamRating(
            id="1",
            league=None,
            players=[
                PreMatchPlayerRating(
                    id="1",
                    rating_value=1100,
                    games_played=0,
                    league=None,
                    match_performance=MatchPerformance(
                        performance_value=1,
                        participation_weight=0.5,
                        projected_participation_weight=0.5,
                    ),
                    position=None
                ),
                PreMatchPlayerRating(
                    id="2",
                    rating_value=900,
                    games_played=0,
                    league=None,
                    match_performance=MatchPerformance(
                        performance_value=0.8,
                        participation_weight=0.3,
                        projected_participation_weight=0.3,
                    ),
                    position=None
                ),
            ],
            rating_value=(1100 * 0.5 + 900 * 0.3) / (0.5 + 0.3),
        ),
        PreMatchTeamRating(
            id="2",
            league=None,
            rating_value=1100,
            players=[]
        ),
    ]

    rating_change = team_rating_generator.generate_rating_change(day_number=1,
                                                                 pre_match_team_rating=pre_match_team_ratings[0],
                                                                 pre_match_opponent_team_rating=pre_match_team_ratings[
                                                                     1])

    expected_player1_predicted_performance = performance_predictor.predict_performance(
        player_rating=pre_match_team_ratings[0].players[0],
        opponent_team_rating=pre_match_team_ratings[1],
        team_rating=pre_match_team_ratings[0]
    )

    expected_player2_predicted_performance = performance_predictor.predict_performance(
        player_rating=pre_match_team_ratings[0].players[1],
        opponent_team_rating=pre_match_team_ratings[1],
        team_rating=pre_match_team_ratings[0]
    )

    expected_rating_change_multiplier = max(team_rating_generator.rating_change_multiplier * (
            (EXPECTED_MEAN_CONFIDENCE_SUM - confidence_sum) / team_rating_generator.confidence_value_denom + 1),
                                            team_rating_generator.rating_change_multiplier * team_rating_generator.min_rating_change_multiplier_ratio)

    expected_player1_rating_change_value = (pre_match_team_ratings[0].players[
                                                0].match_performance.performance_value - expected_player1_predicted_performance) * \
                                           expected_rating_change_multiplier * pre_match_team_ratings[0].players[
                                               0].match_performance.participation_weight

    expected_player2_rating_change_value = (pre_match_team_ratings[0].players[
                                                1].match_performance.performance_value - expected_player2_predicted_performance) * \
                                           expected_rating_change_multiplier * pre_match_team_ratings[0].players[
                                               1].match_performance.participation_weight

    expected_rating_change = TeamRatingChange(
        id="1",
        league=None,
        performance=(1 * 0.5 + 0.8 * 0.3) / (0.5 + 0.3),
        pre_match_rating_value=(1100 * 0.5 + 900 * 0.3) / (0.5 + 0.3),
        predicted_performance=(
                                      expected_player1_predicted_performance * 0.5 + expected_player2_predicted_performance * 0.3) / (
                                      0.5 + 0.3),
        rating_change_value=expected_player1_rating_change_value * 0.5 + expected_player2_rating_change_value * 0.3,
        players=[
            PlayerRatingChange(
                id="1",
                day_number=1,
                league=None,
                participation_weight=0.5,
                predicted_performance=expected_player1_predicted_performance,
                performance=1,
                pre_match_rating_value=1100,
                rating_change_value=expected_player1_rating_change_value,
            ),
            PlayerRatingChange(
                id="2",
                day_number=1,
                league=None,
                participation_weight=0.3,
                predicted_performance=expected_player2_predicted_performance,
                performance=0.8,
                pre_match_rating_value=900,
                rating_change_value=expected_player2_rating_change_value,
            ),
        ]
    )
    assert rating_change == expected_rating_change


def test_update_by_team_rating_change():
    """
    Player Ratings should be updated by the passed in rating_change_value.
    confidence_sum and games_played should also be updated accordingly to performance_weight

    """

    team_rating_change = TeamRatingChange(
        id="1",
        league=None,
        performance=1,
        pre_match_rating_value=1100,
        predicted_performance=1,
        rating_change_value=10,
        players=[
            PlayerRatingChange(
                id="1",
                day_number=1,
                league=None,
                participation_weight=0.5,
                predicted_performance=1,
                performance=1,
                pre_match_rating_value=1100,
                rating_change_value=5,
            ),
            PlayerRatingChange(
                id="2",
                day_number=1,
                league=None,
                participation_weight=0.3,
                predicted_performance=1,
                performance=1,
                pre_match_rating_value=900,
                rating_change_value=3,
            ),
        ]
    )

    team_rating_generator = TeamRatingGenerator()
    original_player_ratings = {
        "1": PlayerRating(
            id="1",
            last_match_day_number=0,
            rating_value=1100,
            games_played=1,
            confidence_sum=1,
            prev_rating_changes=[],
        ),
        "2": PlayerRating(
            id="2",
            rating_value=900,
            last_match_day_number=0,
            games_played=1,
            confidence_sum=1,
            prev_rating_changes=[],
        )
    }
    team_rating_generator.player_ratings = copy.deepcopy(original_player_ratings)
    team_rating_generator.update_rating_by_team_rating_change(team_rating_change=team_rating_change)

    expected_player1_confidence_sum = original_player_ratings[
                                          "1"].confidence_sum - 1 * team_rating_generator.confidence_days_ago_multiplier + \
                                      MATCH_CONTRIBUTION_TO_SUM_VALUE * team_rating_change.players[
                                          0].participation_weight
    expected_player2_confidence_sum = original_player_ratings[
                                          "2"].confidence_sum - 1 * team_rating_generator.confidence_days_ago_multiplier + \
                                      MATCH_CONTRIBUTION_TO_SUM_VALUE * team_rating_change.players[
                                          1].participation_weight

    expected_player_ratings = {
        "1": PlayerRating(
            id="1",
            rating_value=team_rating_change.players[0].rating_change_value + original_player_ratings["1"].rating_value,
            games_played=team_rating_change.players[0].participation_weight + original_player_ratings["1"].games_played,
            last_match_day_number=1,
            confidence_sum=expected_player1_confidence_sum,
            prev_rating_changes=[],
        ),
        "2": PlayerRating(
            id="2",
            rating_value=team_rating_change.players[1].rating_change_value + original_player_ratings["2"].rating_value,
            games_played=team_rating_change.players[1].participation_weight + original_player_ratings["2"].games_played,
            last_match_day_number=1,
            confidence_sum=expected_player2_confidence_sum,
            prev_rating_changes=[],
        )
    }

    assert team_rating_generator.player_ratings == expected_player_ratings


def test_league_ratings_are_updated_when_player_ratings_are_updated():
    """
    When player ratings are updated, league_ratings should also be updated accordingly

    """

    team_rating_change = TeamRatingChange(
        id="1",
        league="league1",
        performance=1,
        pre_match_rating_value=1100,
        predicted_performance=1,
        rating_change_value=10,
        players=[
            PlayerRatingChange(
                id="1",
                day_number=1,
                league="league1",
                participation_weight=0.5,
                predicted_performance=1,
                performance=1,
                pre_match_rating_value=1100,
                rating_change_value=5,
            ),
            PlayerRatingChange(
                id="2",
                day_number=1,
                league="league1",
                participation_weight=0.3,
                predicted_performance=1,
                performance=1,
                pre_match_rating_value=900,
                rating_change_value=3,
            ),
        ]
    )

    start_rating_mock = mock.Mock()

    team_rating_generator = TeamRatingGenerator(
        start_rating_generator=start_rating_mock
    )
    original_player_ratings = {
        "1": PlayerRating(
            id="1",
            last_match_day_number=0,
            rating_value=1100,
            games_played=1,
            confidence_sum=1,
            prev_rating_changes=[],
        ),
        "2": PlayerRating(
            id="2",
            rating_value=900,
            last_match_day_number=0,
            games_played=1,
            confidence_sum=1,
            prev_rating_changes=[],
        )
    }
    team_rating_generator.player_ratings = copy.deepcopy(original_player_ratings)
    team_rating_generator.update_rating_by_team_rating_change(team_rating_change=team_rating_change)

    assert start_rating_mock.update_league_ratings.call_count == 2

    assert team_rating_generator._league_rating_changes["league1"] == team_rating_change.players[
        0].rating_change_value + team_rating_change.players[1].rating_change_value


def test_player_ratings_are_updated_when_league_ratings_reaches_threshold():
    """


    """

    team_rating_change = TeamRatingChange(
        id="1",
        league="league1",
        performance=1,
        pre_match_rating_value=1100,
        predicted_performance=1,
        rating_change_value=10,
        players=[
            PlayerRatingChange(
                id="1",
                day_number=1,
                league="league1",
                participation_weight=0.5,
                predicted_performance=1,
                performance=1,
                pre_match_rating_value=1100,
                rating_change_value=5,
            ),
        ]
    )

    team_rating_generator = TeamRatingGenerator(
    )
    original_player_ratings = {
        "1": PlayerRating(
            id="1",
            last_match_day_number=0,
            rating_value=1100,
            games_played=1,
            confidence_sum=1,
            prev_rating_changes=[],
        ),
    }
    team_rating_generator._league_rating_changes = {
        "league1": team_rating_generator.league_rating_change_update_threshold - 4}
    team_rating_generator._league_rating_changes_count = {"league1": 1}

    team_rating_generator.player_ratings = copy.deepcopy(original_player_ratings)

    expected_new_player_rating = original_player_ratings["1"].rating_value + team_rating_change.players[
        0].rating_change_value + (team_rating_generator.league_rating_change_update_threshold - 4 +
                                  team_rating_change.players[
                                      0].rating_change_value) / 2 * team_rating_generator.league_rating_adjustor_multiplier

    team_rating_generator.update_rating_by_team_rating_change(team_rating_change=team_rating_change)

    assert team_rating_generator._league_rating_changes["league1"] == 0
    assert team_rating_generator.player_ratings["1"].rating_value == expected_new_player_rating
