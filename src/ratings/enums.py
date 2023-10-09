from dataclasses import dataclass
from enum import Enum

@dataclass
class InputColumnNames:
    hour = 'hour'
    performance = 'performance'


@dataclass
class RatingColumnNames:
    entity_rating_difference_from_team = "entity_rating_difference_from_team"
    entity_rating_difference_from_team_projected = "entity_rating_difference_from_team_projected"
    entity_rating = 'entity_rating'
    opponent_rating = "opponent_rating"
    opponent_rating_projected = "opponent_rating_projected"
    team_rating = 'team_rating'
    team_rating_change = "team_rating_change"
    team_rating_projected = 'team_rating_projected'
    entity_rating_difference = 'entity_rating_difference'
    entity_rating_difference_projected = 'entity_rating_difference_projected'
    rating_difference = 'rating_difference'
    rating_difference_projected = 'rating_difference_projected'
    rating_mean = 'rating_mean'
    rating_mean_projected = 'rating_mean_projected'

class PredictedRatingMethod(Enum):
    DEFAULT = 'default'
    MEAN_RATING = "mean_rating"