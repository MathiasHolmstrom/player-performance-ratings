import copy
import logging
import math
import time
from typing import Dict, List, Union
import math

from src.ratings.data_structures import PlayerRating, Match, MatchPlayer, PerformancePredictorParameters, \
    PreMatchPlayerRating, PreMatchTeamRating

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class PerformancePredictor:

    # TODO: Performance prediction based on team-players sharing time with.
    def __init__(self,
                 params: PerformancePredictorParameters,
                 ):
        self.params = params

    def predict_performance(self,
                            player_rating: PreMatchPlayerRating,
                            opponent_team_rating: PreMatchTeamRating,
                            team_rating: PreMatchTeamRating
                            ) -> float:
        rating_difference = player_rating.rating_value - opponent_team_rating.raitng_value
        if team_rating is not None:
            rating_diff_team_from_entity = team_rating.raitng_value - player_rating.rating_value
            team_rating_diff = team_rating.raitng_value - opponent_team_rating.raitng_value
        else:
            rating_diff_team_from_entity = 0
            team_rating_diff = 0

        value = self.params.rating_diff_coef * rating_difference + \
                self.params.rating_diff_team_from_entity_coef * rating_diff_team_from_entity + team_rating_diff * self.params.team_rating_diff_coef
        prediction = (math.exp(value)) / (1 + math.exp(value))
        if prediction > self.params.max_predict_value:
            return self.params.max_predict_value
        elif prediction < (1 - self.params.max_predict_value):
            return (1 - self.params.max_predict_value)
        return prediction


class RatingMeanPerformancePredictor:

    def __init__(self,
                 rating_diff_coef,
                 rating_diff_team_from_entity_coef,
                 team_rating_diff_coef: float,
                 max_predict_value: float = 1,
                 last_sample_count: int = 1500
                 ):
        self.rating_diff_coef = rating_diff_coef
        self.team_rating_diff_coef = team_rating_diff_coef
        self.rating_diff_team_from_entity_coef = rating_diff_team_from_entity_coef
        self.max_predict_value = max_predict_value
        self.last_sample_count = last_sample_count
        self.sum_ratings = []
        self.sum_rating = 0
        self.rating_count = 0

    def predict_performance(self, rating: float, opponent_rating: float, team_rating: float = 0) -> float:

        self.sum_ratings.append(rating)
        self.rating_count += 1
        self.sum_rating += rating
        start_index = max(0, len(self.sum_ratings) - self.last_sample_count)
        self.sum_ratings = self.sum_ratings[start_index:]
        #  average_rating = sum(self.sum_ratings) / len(self.sum_ratings)
        average_rating = self.sum_rating / self.rating_count
        mean_rating = rating * 0.5 + opponent_rating * 0.5 - average_rating

        value = self.rating_diff_coef * mean_rating
        prediction = (math.exp(value)) / (1 + math.exp(value))
        if prediction > self.max_predict_value:
            return self.max_predict_value
        elif prediction < (1 - self.max_predict_value):
            return (1 - self.max_predict_value)
        return prediction


class MatchGenerator():

    def __init__(self,

                 league_identifier: LeagueIdentifier,
                 match_rating_calculator: DefaultMatchRatingCalculator,
                 ):
        self.league_identifier = league_identifier
        self.match_rating_calculator = match_rating_calculator

    def generate(self, match: Match, calculate_participation_weight: bool):
        try:
            self._validate_match(match)
        except ValueError:
            raise

        self._update_entity_leagues(match)

        match = self.match_rating_calculator.generate_pre_match_ratings(
            match=match,
            calculate_participation_weight=calculate_participation_weight
        )
        self._set_match_league(match)
        return match

    def update_ratings_for_matches(self, matches: List[Match]):
        self.match_rating_calculator.update_entity_ratings_for_matches(matches)

        for match in matches:
            if match.league is not None:
                self._update_entity_leagues(match)

            for match_entity in match.entities:
                self.match_rating_calculator.update_league_ratings(match.day_number, match_entity)

            if match.league is not None:
                self.match_rating_calculator.update_entity_ratings_by_league_result(match=match)

