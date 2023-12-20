import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from player_performance_ratings.data_structures import Match, ColumnNames, TeamRating, PlayerRating
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.rating_generator import RatingGenerator


class BayesianMovingAverage(RatingGenerator):

    def __init__(self,
                 evidence_exponential_weight: float = 0.96,
                 likelihood_exponential_weight: float = 0.98,
                 likelihood_denom: float = 50,
                 prior_granularity_count_max: int = 200,
                 prior_by_league: bool = True,
                 prior_by_position: bool = True,
                 ):

        self._player_ratings: dict[str, PlayerRating] = {}
        self.evidence_exponential_weight = evidence_exponential_weight
        self.likelihood_exponential_weight = likelihood_exponential_weight
        self.likelihood_denom = likelihood_denom
        self.prior_granularity_count_max = prior_granularity_count_max
        self.by_league = prior_by_league
        self.by_position = prior_by_position
        self.player_performances: dict[str, list[float]] = {}
        self.player_days: dict[str, list[int]] = {}
        self._player_prior_ratings: dict[str, float] = {}
        self._granularity_ratings: dict[str, list[float]] = {}
        self._granularity_players: dict[str, list[str]] = {}

    def generate(self, matches: list[Match], df: Optional[pd.DataFrame] = None,
                 column_names: ColumnNames = None) -> dict[RatingColumnNames, list[float]]:

        if df is not None and column_names is not None:
            mean_performance_value = df[column_names.performance].mean()
        else:
            count = 0
            sum_mean_performance_value = 0
            for match in matches:
                for team in match.teams:
                    for team_player in team.players:
                        count += 1
                        sum_mean_performance_value += team_player.performance.performance_value

            mean_performance_value = sum_mean_performance_value / count

        ratings = {
            RatingColumnNames.TIME_WEIGHTED_RATING: [],
            RatingColumnNames.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO: [],
            RatingColumnNames.TIME_WEIGHTED_RATING_EVIDENCE: [],
        }

        for match in matches:
            for team in match.teams:
                for team_player in team.players:
                    self.player_performances.setdefault(team_player.id, [])
                    self.player_days.setdefault(team_player.id, [])

                    days_agos = match.day_number - np.array(self.player_days[team_player.id])
                    weights = np.power(self.evidence_exponential_weight, days_agos)
                    evidence_performances = np.sum(
                        np.array(self.player_performances[team_player.id]) * weights) / np.sum(
                        weights)
                    if math.isnan(evidence_performances):
                        evidence_performances = None
                    likelihood_exponential_weights = np.power(self.likelihood_exponential_weight, days_agos)

                    prior_rating = self._generate_base_prior(league=team_player.league, position=team_player.position,
                                                             base_prior_value=mean_performance_value,
                                                             player_id=team_player.id)

                    likelihood_ratio = min(sum(likelihood_exponential_weights) / self.likelihood_denom, 1)
                    posterior_rating = (
                                               1 - likelihood_ratio) * prior_rating + likelihood_ratio * evidence_performances if evidence_performances is not None else prior_rating

                    self._update_granularity_ratings(league=team_player.league, player_rating=posterior_rating,
                                                     position=team_player.position, player_id=team_player.id)

                    self.player_days[team_player.id].append(match.day_number)
                    self.player_performances[team_player.id].append(team_player.performance.performance_value)

                    ratings[RatingColumnNames.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO].append(likelihood_ratio)
                    ratings[RatingColumnNames.TIME_WEIGHTED_RATING].append(posterior_rating)
                    ratings[RatingColumnNames.TIME_WEIGHTED_RATING_EVIDENCE].append(evidence_performances)

        return ratings

    def _generate_base_prior(self, player_id: str, base_prior_value: float, league: Optional[str],
                             position: Optional[str]) -> float:

        if player_id in self._player_prior_ratings:
            return self._player_prior_ratings[player_id]

        granularity_id = ""
        if self.by_league:
            league = league if league is not None else ""
            granularity_id += league
        if self.by_position:
            position = position if position is not None else ""
            granularity_id += position

        self._granularity_ratings.setdefault(granularity_id, [])
        self._granularity_players.setdefault(granularity_id, [])
        granularity_weight = len(self._granularity_ratings[granularity_id]) / self.prior_granularity_count_max

        if len(self._granularity_ratings[granularity_id]) == 0:
            self._player_prior_ratings[player_id] = base_prior_value
            return base_prior_value

        prior_rating = granularity_weight * np.mean(self._granularity_ratings[granularity_id]) + (
                1 - granularity_weight) * base_prior_value
        self._player_prior_ratings[player_id] = prior_rating
        return prior_rating

    def _update_granularity_ratings(self, player_id: str, player_rating: float, league: Optional[str],
                                    position: Optional[str]):
        granularity_id = ""
        if self.by_league:
            league = league if league is not None else ""
            granularity_id += league
        if self.by_position:
            position = position if position is not None else ""
            granularity_id += position

        if player_id in self._granularity_players[granularity_id]:
            index = self._granularity_players[granularity_id].index(player_id)
            self._granularity_ratings[granularity_id][index] = player_rating
            return

        self._granularity_players[granularity_id].append(player_id)
        self._granularity_ratings[granularity_id].append(player_rating)

    @property
    def player_ratings(self) -> dict[str, PlayerRating]:
        return self._player_ratings

    @property
    def team_ratings(self) -> list[TeamRating]:
        return self.team_ratings
