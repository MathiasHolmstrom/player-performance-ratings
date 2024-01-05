import math
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from lightgbm import LGBMRegressor


from player_performance_ratings.data_structures import PreMatchPlayerRating, PreMatchTeamRating

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class PerformancePredictor(ABC):

    @abstractmethod
    def predict_performance(self,
                            player_rating: PreMatchPlayerRating,
                            opponent_team_rating: PreMatchTeamRating,
                            team_rating: PreMatchTeamRating
                            ) -> float:
        pass



class RatingDifferencePerformancePredictor(PerformancePredictor):

    # TODO: Performance prediction based on team-players sharing time with.
    def __init__(self,
                 rating_diff_coef: float = 0.005757,
                 rating_diff_team_from_entity_coef: float = 0.0,
                 team_rating_diff_coef: float = 0.0,
                 max_predict_value: float = 1,
                 participation_weight_coef: Optional[float] = None,
                 mean_participation_weight: Optional[float] = None,
                 ):
        self.rating_diff_coef = rating_diff_coef
        self.rating_diff_team_from_entity_coef = rating_diff_team_from_entity_coef
        self.team_rating_diff_coef = team_rating_diff_coef
        self.max_predict_value = max_predict_value
        self.mean_participation_weight = mean_participation_weight
        self.participation_weight_coef = participation_weight_coef
        if self.participation_weight_coef and mean_participation_weight is None:
            raise ValueError("mean_duration must be passed if duration_column_name is passed")

    def predict_performance(self,
                            player_rating: PreMatchPlayerRating,
                            opponent_team_rating: PreMatchTeamRating,
                            team_rating: PreMatchTeamRating
                            ) -> float:
        rating_difference = player_rating.rating_value - opponent_team_rating.rating_value

        rating_diff_team_from_entity = team_rating.rating_value - player_rating.rating_value
        team_rating_diff = team_rating.rating_value - opponent_team_rating.rating_value

        value = self.rating_diff_coef * rating_difference + \
                self.rating_diff_team_from_entity_coef * rating_diff_team_from_entity + team_rating_diff * self.team_rating_diff_coef
        if self.participation_weight_coef:
            value += value * self.participation_weight_coef * (
                    player_rating.match_performance.participation_weight - self.mean_participation_weight)

        prediction = (math.exp(value)) / (1 + math.exp(value))
        if prediction > self.max_predict_value:
            return self.max_predict_value
        elif prediction < (1 - self.max_predict_value):
            return (1 - self.max_predict_value)
        return prediction


class RatingMeanPerformancePredictor(PerformancePredictor):

    def __init__(self,
                 coef: float = 0.005757,
                 max_predict_value: float = 1,
                 last_sample_count: int = 1500
                 ):
        self.coef = coef
        self.max_predict_value = max_predict_value
        self.last_sample_count = last_sample_count
        self.sum_ratings = []
        self.sum_rating = 0
        self.rating_count = 0

    def predict_performance(self,
                            player_rating: PreMatchPlayerRating,
                            opponent_team_rating: PreMatchTeamRating,
                            team_rating: PreMatchTeamRating) -> float:

        self.sum_ratings.append(player_rating.rating_value)
        self.rating_count += 1
        self.sum_rating += player_rating.rating_value
        start_index = max(0, len(self.sum_ratings) - self.last_sample_count)
        self.sum_ratings = self.sum_ratings[start_index:]
        #  average_rating = sum(self.sum_ratings) / len(self.sum_ratings)
        historical_average_rating = self.sum_rating / self.rating_count
        net_mean_rating_over_historical_average = player_rating.rating_value * 0.5 + opponent_team_rating.rating_value * 0.5 - historical_average_rating

        value = self.coef * net_mean_rating_over_historical_average
        prediction = (math.exp(value)) / (1 + math.exp(value))
        if prediction > self.max_predict_value:
            return self.max_predict_value
        elif prediction < (1 - self.max_predict_value):
            return (1 - self.max_predict_value)
        return prediction
