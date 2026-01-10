import math
from abc import ABC, abstractmethod
from collections.abc import Mapping

from spforge.data_structures import PreMatchTeamRating


def _logistic(value: float) -> float:
    # numerically safe enough for typical rating ranges; swap to expit if you want
    return math.exp(value) / (1 + math.exp(value))


def _clamp_probability(p: float, max_predict_value: float) -> float:
    """
    If max_predict_value == 1 -> no clamping.
    If max_predict_value < 1 -> clamp to [1-max_predict_value, max_predict_value].
    """
    if max_predict_value >= 1:
        return p
    lo = 1 - max_predict_value
    hi = max_predict_value
    return min(hi, max(lo, p))


def weighted_team_rating_from_players(
    team_rating: PreMatchTeamRating,
    players_playing_time: Mapping[str, float] | None = None,
) -> float:
    """
    Optional helper for team predictors if you *do* have a team-level playing-time map.

    players_playing_time: mapping of player_id(str) -> playing_time weight
    If missing/empty -> falls back to team_rating.rating_value.
    """
    if not players_playing_time or not isinstance(players_playing_time, dict):
        return team_rating.rating_value

    weight_sum = 0.0
    time_sum = 0.0
    for p in getattr(team_rating, "players", []) or []:
        key = str(p.id)
        if key in players_playing_time:
            t = players_playing_time[key]
            weight_sum += p.rating_value * t
            time_sum += t

    return (weight_sum / time_sum) if time_sum > 0 else team_rating.rating_value


class TeamPerformancePredictor(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def predict_performance(
        self,
        rating_value: float,
        opponent_team_rating_value: float,
    ) -> float:
        """
        Return a probability-like prediction for team vs opponent (e.g., win prob proxy).
        """
        pass


class TeamRatingNonOpponentPerformancePredictor(TeamPerformancePredictor):
    """
    Team-only analogue of PlayerRatingNonOpponentPerformancePredictor:
    compares team rating to a rolling historical average team rating.
    """

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
        self._prev_entries_ratings: list[float] = []

    def reset(self) -> None:
        self._prev_entries_ratings = []

    def predict_performance(
        self,
        rating_value: float,
        opponent_team_rating_value: float,
    ) -> float:
        start_index = max(0, len(self._prev_entries_ratings) - self.last_sample_count)
        recent = self._prev_entries_ratings[start_index:]

        if len(recent) > self.min_count_for_historical_average:
            historical_average = sum(recent) / len(recent)
        else:
            historical_average = self.historical_average_value_default

        net_over_hist = rating_value - historical_average
        value = self.coef * net_over_hist
        prediction = _logistic(value)

        self._prev_entries_ratings.append(rating_value)
        return prediction


class TeamRatingDifferencePerformancePredictor(TeamPerformancePredictor):
    """
    Team-only analogue of RatingPlayerDifferencePerformancePredictor:
    Uses team rating vs opponent team rating (and optional derived terms).

    NOTE: the original player predictor had player-specific "playing time vs specific opponents".
    That concept does not exist at team level unless you have an expected lineup/playing-time map.
    If you DO have that, use `weighted_team_rating_from_players(...)` externally and pass it in
    by constructing a synthetic PreMatchTeamRating.rating_value or by extending this class.
    """

    def __init__(
        self,
        rating_diff_coef: float = 0.005757,
        # kept for parity, but these are less meaningful in pure team vs team
        rating_diff_team_from_entity_coef: float = 0.0,
        team_rating_diff_coef: float = 0.0,
        max_predict_value: float = 1.0,
    ):
        self.rating_diff_coef = rating_diff_coef
        self.rating_diff_team_from_entity_coef = rating_diff_team_from_entity_coef
        self.team_rating_diff_coef = team_rating_diff_coef
        self.max_predict_value = max_predict_value

    def reset(self) -> None:
        pass

    def predict_performance(
        self,
        rating_value: float,
        opponent_team_rating_value: float,
    ) -> float:
        # base term: team vs opponent
        rating_difference = rating_value - opponent_team_rating_value

        # In the player version these were:
        #   rating_diff_team_from_entity = team_rating_value - player_rating.rating_value
        #   team_rating_diff = team_rating_value - opp_rating_value
        # For team-only, there is no "entity" separate from team, so:
        rating_diff_team_from_entity = 0.0
        team_rating_diff = rating_difference

        value = (
            self.rating_diff_coef * rating_difference
            + self.rating_diff_team_from_entity_coef * rating_diff_team_from_entity
            + self.team_rating_diff_coef * team_rating_diff
        )

        prediction = _logistic(value)
        return _clamp_probability(prediction, self.max_predict_value)


class TeamRatingMeanPerformancePredictor(TeamPerformancePredictor):
    """
    Team-only analogue of RatingMeanPerformancePredictor:
    Uses the mean of team and opponent, compared to a historical average.
    """

    def __init__(
        self,
        coef: float = 0.005757,
        max_predict_value: float = 1.0,
        last_sample_count: int = 1500,
    ):
        self.coef = coef
        self.max_predict_value = max_predict_value
        self.last_sample_count = last_sample_count
        self._entries: list[float] = []
        self._sum = 0.0
        self._count = 0

    def reset(self) -> None:
        self._entries = []
        self._sum = 0.0
        self._count = 0

    def predict_performance(
        self,
        rating_value: float,
        opponent_team_rating_value: float,
    ) -> float:
        # update history with TEAM rating
        self._entries.append(rating_value)
        self._count += 1
        self._sum += rating_value

        # keep a rolling window list (note: sum/count still track full history like your original)
        start_index = max(0, len(self._entries) - self.last_sample_count)
        self._entries = self._entries[start_index:]

        historical_average = self._sum / self._count

        net_over_hist = 0.5 * rating_value + 0.5 * opponent_team_rating_value - historical_average

        value = self.coef * net_over_hist
        prediction = _logistic(value)
        return _clamp_probability(prediction, self.max_predict_value)
