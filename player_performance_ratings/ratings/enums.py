from dataclasses import dataclass
from enum import Enum

@dataclass
class InputColumnNames:
    hour = 'hour'
    performance = 'performance'


@dataclass
class RatingColumnNames:
    player_rating_difference_from_team = "player_rating_difference_from_team"
    player_rating_difference_from_team_projected = "player_rating_difference_from_team_projected"
    player_rating = 'player_rating'
    player_rating_change = "player_rating_change"
    opponent_rating = "opponent_rating"
    opponent_rating_projected = "opponent_rating_projected"
    team_rating = 'team_rating'
    team_rating_change = "team_rating_change"
    team_rating_projected = 'team_rating_projected'
    player_rating_difference = 'player_rating_difference'
    player_rating_difference_projected = 'player_rating_difference_projected'
    rating_difference = 'rating_difference'
    rating_difference_projected = 'rating_difference_projected'
    rating_mean = 'rating_mean'
    rating_mean_projected = 'rating_mean_projected'
    player_league = "player_league"
    opponent_league = "opponent_league"
    match_id = "match_id"

class PredictedRatingMethod(Enum):
    DEFAULT = 'default'
    MEAN_RATING = "mean_rating"