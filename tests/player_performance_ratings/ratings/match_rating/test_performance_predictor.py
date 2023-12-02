import pytest

from player_performance_ratings.data_structures import PreMatchPlayerRating, MatchPerformance, PreMatchTeamRating
from player_performance_ratings.ratings.match_rating.performance_predictor import RatingDifferencePerformancePredictor


@pytest.mark.parametrize("team_rating_value", [0, 1000])
@pytest.mark.parametrize("player_rating_value", [0,  500, 1000])
@pytest.mark.parametrize("opponent_team_rating_value", [0, 1000])
def test_rating_difference_performance_predictor(team_rating_value: float, player_rating_value: float,
                                                 opponent_team_rating_value: float):
    """
    Property based testing based on following parameters:
    * rating_diff_coef is positive and >rating_diff_team_from_entity_coef and team_rating_diff_coef
    * team_rating_diff_coef is positive and > rating_diff_team_from_entity_coef
    * rating_diff_team_from_entity_coef is positive and
    * team_rating_diff_coef + rating_diff_team_from_entity_coef = rating_diff_coef
    """

    performance_predictor = RatingDifferencePerformancePredictor(
        rating_diff_coef=0.003,
        rating_diff_team_from_entity_coef=0.001,
        team_rating_diff_coef=0.002
    )

    player_rating = PreMatchPlayerRating(
        rating_value=player_rating_value,
        match_performance=MatchPerformance(
            participation_weight=0.5,
            performance_value=0.5
        ),
        games_played=1,
        league='league',
        id="1"
    )

    team_rating = PreMatchTeamRating(
        rating_value=team_rating_value,
        league='league',
        id="1",
        players=[]
    )

    opponent_team_rating = PreMatchTeamRating(
        rating_value=opponent_team_rating_value,
        league='league',
        id="2",
        players=[]
    )

    predicted_performance = performance_predictor.predict_performance(
        player_rating=player_rating,
        opponent_team_rating=opponent_team_rating,
        team_rating=team_rating
    )

    if player_rating_value > opponent_team_rating_value and player_rating_value >= team_rating_value:
        assert predicted_performance > 0.5

    elif player_rating_value < opponent_team_rating_value and player_rating_value <= team_rating_value:
        assert predicted_performance < 0.5

    if player_rating_value > opponent_team_rating_value and player_rating_value < team_rating_value:
        assert predicted_performance > 0.5

    elif player_rating_value < opponent_team_rating_value and player_rating_value > team_rating_value:
        assert predicted_performance < 0.5

    if player_rating_value == opponent_team_rating_value and team_rating_value > player_rating_value:
        assert predicted_performance > 0.5

    if player_rating_value == opponent_team_rating_value and team_rating_value < player_rating_value:
        assert predicted_performance < 0.5

    if player_rating_value == opponent_team_rating_value and team_rating_value == player_rating_value:
        assert predicted_performance == 0.5


