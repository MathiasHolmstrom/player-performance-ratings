from dataclasses import dataclass
from enum import Enum, StrEnum


@dataclass
class RatingKnownFeatures(StrEnum):
    PLAYER_RATING = "player_rating"
    PLAYER_OFF_RATING = "player_off_rating"
    PLAYER_DEF_RATING = "player_def_rating"
    PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED = "player_rating_difference_from_team_projected"
    TEAM_OFF_RATING_PROJECTED = "team_off_rating_projected"
    TEAM_DEF_RATING_PROJECTED = "team_def_rating_projected"
    OPPONENT_OFF_RATING_PROJECTED = "opponent_off_rating_projected"
    OPPONENT_DEF_RATING_PROJECTED = "opponent_def_rating_projected"
    OPPONENT_RATING_PROJECTED = "opponent_rating_projected"
    TEAM_RATING_PROJECTED = "team_rating_projected"
    PLAYER_RATING_DIFFERENCE_PROJECTED = "player_rating_difference_projected"
    TEAM_RATING_DIFFERENCE_PROJECTED = "team_rating_difference_projected"
    RATING_MEAN_PROJECTED = "rating_mean_projected"
    TEAM_LEAGUE = "team_league"
    PLAYER_LEAGUE = "player_league"
    OPPONENT_LEAGUE = "opponent_league"


class RatingUnknownFeatures(StrEnum):
    PERFORMANCE = "performance"
    TEAM_RATING_DIFFERENCE = "team_rating_difference"
    PLAYER_RATING_DIFFERENCE = "player_rating_difference"
    TEAM_RATING = "team_rating"
    OPPONENT_RATING = "opponent_rating"
    PLAYER_RATING_DIFFERENCE_FROM_TEAM = "player_rating_difference_from_team"
    PLAYER_RATING_CHANGE = "player_rating_change"
    RATING_MEAN = "rating_mean"
    PLAYER_PREDICTED_PERFORMANCE = "player_predicted_performance"
    PLAYER_PREDICTED_OFF_PERFORMANCE = "player_predicted_off_performance"
    PLAYER_PREDICTED_DEF_PERFORMANCE = "player_predicted_def_performance"


class PredictedRatingMethod(Enum):
    DEFAULT = "default"
    MEAN_RATING = "mean_rating"
