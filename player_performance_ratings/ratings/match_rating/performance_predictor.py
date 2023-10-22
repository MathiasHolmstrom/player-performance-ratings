import math

from player_performance_ratings.data_structures import PreMatchPlayerRating, PreMatchTeamRating

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class PerformancePredictor:

    # TODO: Performance prediction based on team-players sharing time with.
    def __init__(self,
                 rating_diff_coef: float = 0.005757,
                 rating_diff_team_from_entity_coef: float= 0.0,
                 team_rating_diff_coef: float= 0.0,
                 max_predict_value: float = 1,
                 ):
        self.rating_diff_coef = rating_diff_coef
        self.rating_diff_team_from_entity_coef = rating_diff_team_from_entity_coef
        self.team_rating_diff_coef = team_rating_diff_coef
        self.max_predict_value = max_predict_value

    def predict_performance(self,
                            player_rating: PreMatchPlayerRating,
                            opponent_team_rating: PreMatchTeamRating,
                            team_rating: PreMatchTeamRating
                            ) -> float:
        rating_difference = player_rating.rating_value - opponent_team_rating.rating_value
        if team_rating is not None:
            rating_diff_team_from_entity = team_rating.rating_value - player_rating.rating_value
            team_rating_diff = team_rating.rating_value - opponent_team_rating.rating_value
        else:
            rating_diff_team_from_entity = 0
            team_rating_diff = 0

        value = self.rating_diff_coef * rating_difference + \
                self.rating_diff_team_from_entity_coef * rating_diff_team_from_entity + team_rating_diff * self.team_rating_diff_coef
        prediction = (math.exp(value)) / (1 + math.exp(value))
        if prediction > self.max_predict_value:
            return self.max_predict_value
        elif prediction < (1 - self.max_predict_value):
            return (1 - self.max_predict_value)
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