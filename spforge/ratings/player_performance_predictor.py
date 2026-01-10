import math
from abc import ABC, abstractmethod

from spforge.data_structures import (
    PreMatchPlayerRating,
    PreMatchTeamRating,
)

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = "certain_sum"


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class PlayerPerformancePredictor(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def predict_performance(
        self,
        player_rating: PreMatchPlayerRating,
        opponent_team_rating: PreMatchTeamRating,
        team_rating: PreMatchTeamRating,
    ) -> float:
        pass


class PlayerRatingNonOpponentPerformancePredictor(PlayerPerformancePredictor):

    def __init__(
        self,
        coef: float = 0.0015,
        last_sample_count: int = 1500,
        min_count_for_historical_average: int = 200,
        historical_average_value_default: float = 1000,
    ):
        self.coef = coef
        self.last_sample_count = last_sample_count
        self.min_count_for_historical_average = min_count_for_historical_average
        self.historical_average_value_default = historical_average_value_default
        if self.min_count_for_historical_average < 1:
            raise ValueError("min_count_for_historical_average must be positive")
        self._prev_entries_ratings = []

    def reset(self):
        self._prev_entries_ratings = []

    def predict_performance(
        self,
        player_rating: PreMatchPlayerRating,
        opponent_team_rating: PreMatchTeamRating,
        team_rating: PreMatchTeamRating,
    ) -> float:
        start_index = max(0, len(self._prev_entries_ratings) - self.last_sample_count)
        recent_prev_entries_ratings = self._prev_entries_ratings[start_index:]
        if len(recent_prev_entries_ratings) > self.min_count_for_historical_average:
            historical_average_rating = sum(recent_prev_entries_ratings) / len(
                recent_prev_entries_ratings
            )
        else:
            historical_average_rating = self.historical_average_value_default
        net_mean_rating_over_historical_average = (
            player_rating.rating_value - historical_average_rating
        )

        value = self.coef * net_mean_rating_over_historical_average
        prediction = (math.exp(value)) / (1 + math.exp(value))
        self._prev_entries_ratings.append(player_rating.rating_value)

        return prediction


class RatingPlayerDifferencePerformancePredictor(PlayerPerformancePredictor):

    # TODO: Performance prediction based on team-players sharing time with.
    def __init__(
        self,
        rating_diff_coef: float = 0.005757,
        rating_diff_team_from_entity_coef: float = 0.0,
        team_rating_diff_coef: float = 0.0,
        max_predict_value: float = 1,
        participation_weight_coef: float | None = None,
    ):
        self.rating_diff_coef = rating_diff_coef
        self.rating_diff_team_from_entity_coef = rating_diff_team_from_entity_coef
        self.team_rating_diff_coef = team_rating_diff_coef
        self.max_predict_value = max_predict_value
        self.participation_weight_coef = participation_weight_coef

    def reset(self):
        pass

    def predict_performance(
        self,
        player_rating: PreMatchPlayerRating,
        opponent_team_rating: PreMatchTeamRating,
        team_rating: PreMatchTeamRating,
    ) -> float:

        if player_rating.match_performance.team_players_playing_time and isinstance(
            player_rating.match_performance.team_players_playing_time, dict
        ):
            weight_team_rating = 0
            sum_playing_time = 0
            for team_player in team_rating.players:
                if (
                    str(team_player.id) in player_rating.match_performance.team_players_playing_time
                    or str(team_player.id)
                    in player_rating.match_performance.team_players_playing_time
                ):
                    weight_team_rating += (
                        team_player.rating_value
                        * player_rating.match_performance.team_players_playing_time[
                            str(team_player.id)
                        ]
                    )
                    sum_playing_time += player_rating.match_performance.team_players_playing_time[
                        str(team_player.id)
                    ]

            if sum_playing_time > 0:
                team_rating_value = weight_team_rating / sum_playing_time
            else:
                team_rating_value = team_rating.rating_value

        else:
            team_rating_value = team_rating.rating_value

        if player_rating.match_performance.opponent_players_playing_time and isinstance(
            player_rating.match_performance.team_players_playing_time, dict
        ):
            weight_opp_rating = 0
            sum_playing_time = 0
            for opp_player in opponent_team_rating.players:
                if (
                    str(opp_player.id)
                    in player_rating.match_performance.opponent_players_playing_time
                ):
                    weight_opp_rating += (
                        opp_player.rating_value
                        * player_rating.match_performance.opponent_players_playing_time[
                            str(opp_player.id)
                        ]
                    )
                    sum_playing_time += (
                        player_rating.match_performance.opponent_players_playing_time[
                            str(opp_player.id)
                        ]
                    )
            if sum_playing_time > 0:
                opp_rating_value = weight_opp_rating / sum_playing_time
            else:
                opp_rating_value = opponent_team_rating.rating_value

        else:
            opp_rating_value = opponent_team_rating.rating_value

        rating_difference = player_rating.rating_value - opp_rating_value

        rating_diff_team_from_entity = team_rating_value - player_rating.rating_value
        team_rating_diff = team_rating_value - opp_rating_value

        value = (
            self.rating_diff_coef * rating_difference
            + self.rating_diff_team_from_entity_coef * rating_diff_team_from_entity
            + team_rating_diff * self.team_rating_diff_coef
        )

        prediction = (math.exp(value)) / (1 + math.exp(value))

        if prediction > self.max_predict_value:
            return self.max_predict_value
        elif prediction < (1 - self.max_predict_value):
            return 1 - self.max_predict_value

        return prediction


class RatingMeanPerformancePredictor(PlayerPerformancePredictor):

    def __init__(
        self,
        coef: float = 0.005757,
        max_predict_value: float = 1,
        last_sample_count: int = 1500,
    ):
        self.coef = coef
        self.max_predict_value = max_predict_value
        self.last_sample_count = last_sample_count
        self.sum_ratings = []
        self.sum_rating = 0
        self.rating_count = 0

    def reset(self):
        self.sum_ratings = []
        self.sum_rating = 0
        self.rating_count = 0

    def predict_performance(
        self,
        player_rating: PreMatchPlayerRating,
        opponent_team_rating: PreMatchTeamRating,
        team_rating: PreMatchTeamRating,
    ) -> float:

        self.sum_ratings.append(player_rating.rating_value)
        self.rating_count += 1
        self.sum_rating += player_rating.rating_value
        start_index = max(0, len(self.sum_ratings) - self.last_sample_count)
        self.sum_ratings = self.sum_ratings[start_index:]
        historical_average_rating = self.sum_rating / self.rating_count
        net_mean_rating_over_historical_average = (
            player_rating.rating_value * 0.5
            + opponent_team_rating.rating_value * 0.5
            - historical_average_rating
        )

        value = self.coef * net_mean_rating_over_historical_average
        prediction = (math.exp(value)) / (1 + math.exp(value))
        if prediction > self.max_predict_value:
            return self.max_predict_value
        elif prediction < (1 - self.max_predict_value):
            return 1 - self.max_predict_value
        return prediction
