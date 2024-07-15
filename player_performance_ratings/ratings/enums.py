from dataclasses import dataclass
from enum import Enum


@dataclass
class InputColumnNames:
    hour = "hour"
    performance = "performance"


@dataclass
class RatingKnownFeatures:
    PLAYER_RATING = "player_rating"
    PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED = (
        "player_rating_difference_from_team_projected"
    )
    OPPONENT_RATING_PROJECTED = "opponent_rating_projected"
    TEAM_RATING_PROJECTED = "team_rating_projected"
    PLAYER_RATING_DIFFERENCE_PROJECTED = "player_rating_difference_projected"
    RATING_DIFFERENCE_PROJECTED = "rating_difference_projected"
    RATING_MEAN_PROJECTED = "rating_mean_projected"
    TEAM_LEAGUE = "team_league"
    PLAYER_LEAGUE = "player_league"
    OPPONENT_LEAGUE = "opponent_league"
    MATCH_ID = "match_id"
    RATING_DIFFERENCE_POSITION = "rating_difference_position"


@dataclass
class RatingHistoricalFeatures:
    PERFORMANCE = "performance"
    RATING_DIFFERENCE = "rating_difference"
    PLAYER_RATING_DIFFERENCE = "player_rating_difference"
    TEAM_RATING = "team_rating"
    OPPONENT_RATING = "opponent_rating"
    PLAYER_RATING_DIFFERENCE_FROM_TEAM = "player_rating_difference_from_team"
    PLAYER_RATING_CHANGE = "player_rating_change"
    RATING_MEAN = "rating_mean"
    PLAYER_PREDICTED_PERFORMANCE = "player_predicted_performance"


class PredictedRatingMethod(Enum):
    DEFAULT = "default"
    MEAN_RATING = "mean_rating"
