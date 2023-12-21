from dataclasses import dataclass
from enum import Enum


@dataclass
class InputColumnNames:
    hour = 'hour'
    performance = 'performance'


@dataclass
class RatingColumnNames:
    PLAYER_RATING_DIFFERENCE_FROM_TEAM = "player_rating_difference_from_team"
    PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED = "player_rating_difference_from_team_projected"
    PLAYER_RATING = 'player_rating'
    PLAYER_RATING_CHANGE = "player_rating_change"
    OPPONENT_RATING = "opponent_rating"
    OPPONENT_RATING_PROJECTED = "opponent_rating_projected"
    TEAM_RATING = 'team_rating'
    TEAM_RATING_CHANGE = "team_rating_change"
    TEAM_RATING_PROJECTED = 'team_rating_projected'
    PLAYER_RATING_DIFFERENCE = 'player_rating_difference'
    PLAYER_RATING_DIFFERENCE_PROJECTED = 'player_rating_difference_projected'
    RATING_DIFFERENCE = 'rating_difference'
    RATING_DIFFERENCE_PROJECTED = 'rating_difference_projected'
    RATING_MEAN = 'rating_mean'
    RATING_MEAN_PROJECTED = 'rating_mean_projected'
    PLAYER_LEAGUE = "player_league"
    OPPONENT_LEAGUE = "opponent_league"
    MATCH_ID = "match_id"
    PLAYER_PREDICTED_PERFORMANCE = "player_predicted_performance"
    TIME_WEIGHTED_RATING = "time_weighted_rating"
    TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO = "time_weighted_rating_likelihood_ratio"
    TIME_WEIGHTED_RATING_EVIDENCE = "time_weighted_rating_evidence"

    PERFORMANCE = "performance"




class PredictedRatingMethod(Enum):
    DEFAULT = 'default'
    MEAN_RATING = "mean_rating"
