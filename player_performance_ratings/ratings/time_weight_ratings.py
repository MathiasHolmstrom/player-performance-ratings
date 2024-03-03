import math
from typing import Optional, Union

import numpy as np
import pandas as pd

from player_performance_ratings.data_structures import Match, ColumnNames, TeamRating, PlayerRating
from player_performance_ratings.ratings import convert_df_to_matches
from player_performance_ratings.ratings.enums import RatingEstimatorFeatures, RatingHistoricalFeatures
from player_performance_ratings.ratings.rating_generator import RatingGenerator


# NOT WORKING AT THE MOMENT

class BayesianTimeWeightedRating(RatingGenerator):
    """
    Generates ratings for players and teams by watching past performances higher using a combination of prior, evidence and likelihood.
    """

    def __init__(self,
                 column_names: ColumnNames,
                 evidence_exponential_weight: float = 0.96,
                 likelihood_exponential_weight: float = 0.98,
                 likelihood_denom: float = 50,
                 prior_granularity_count_max: int = 200,
                 prior_by_league: bool = True,
                 prior_by_position: bool = True,
                 features_created: Optional[list[str]] = None,
                 ):

        """

        :param evidence_exponential_weight:
            The weight of the evidence. Value should be between 0 and 1.
            A value closer to 0 makes performances further in the past less important.

        :param likelihood_exponential_weight:
            Determines how the likelihood_ratio is calculated. The likelihood ratio determines how much the prior and evidence are weighted.
            A likelihood_exponential_weight closer to 0 makes the prior have a larger weight relative to the evidence.

        :param likelihood_denom:
            The denominator of the likelihood_ratio.
            An increase in the likelihood_denom makes the player require more data (and with higher recency) for the likelihood_ratio to approach 1.
            Thus a higher likelihood_denom makes the prior have a larger weight relative to the evidence.

        :param prior_granularity_count_max:
            In calculating the prior-rating, past ratings of players in similar groups/granularity are used.
            If no players exist in the group at the time the prior of a player is detemrined, the player will receive the mean of all performances as his rating
            The more players exist in the group, the more the prior-rating will be based on the past performances of players in the group.
            A higher prior_granularity_count_max results in a lower weight of the within-group-players-rating relative to the mean of all performances across all players in the dataset.


        :param prior_by_league:
            If True, the prior-rating will be based on ratings of players within the same league
        :param prior_by_position:
            If True, the prior-rating will be based on ratings of players within the same position
        """

        super().__init__(column_names=column_names)
        if evidence_exponential_weight < 0 or evidence_exponential_weight > 1:
            raise ValueError("evidence_exponential_weight must be between 0 and 1")
        if likelihood_exponential_weight < 0 or likelihood_exponential_weight > 1:
            raise ValueError("likelihood_exponential_weight must be between 0 and 1")

        self._player_ratings: dict[str, PlayerRating] = {}
        self.evidence_exponential_weight = evidence_exponential_weight
        self.likelihood_exponential_weight = likelihood_exponential_weight
        self.likelihood_denom = likelihood_denom
        self.prior_granularity_count_max = prior_granularity_count_max
        self.by_league = prior_by_league
        self.by_position = prior_by_position
        self._features_out = features_created or [RatingEstimatorFeatures.TIME_WEIGHTED_RATING,
                                                  RatingEstimatorFeatures.TIME_WEIGHTED_RATING_EVIDENCE,
                                                  RatingEstimatorFeatures.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO]

        self.player_performances: dict[str, list[float]] = {}
        self.player_days: dict[str, list[int]] = {}
        self._player_prior_ratings: dict[str, float] = {}
        self._granularity_ratings: dict[str, list[float]] = {}
        self._granularity_players: dict[str, list[str]] = {}

    def generate_historical(self, df: Optional[pd.DataFrame] = None, matches: Optional[list[Match]] = None) -> dict[RatingEstimatorFeatures, list[float]]:

        if matches is None and df is None:
            raise ValueError("If matches is not passed, df and column names must be massed")

        if matches is None:
            matches = convert_df_to_matches(df=df, column_names=self.column_names)

        if df is not None:
            mean_performance_value = df[self.column_names.performance].mean()
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
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING: [],
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO: [],
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING_EVIDENCE: [],
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

                    ratings[RatingEstimatorFeatures.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO].append(likelihood_ratio)
                    ratings[RatingEstimatorFeatures.TIME_WEIGHTED_RATING].append(posterior_rating)
                    ratings[RatingEstimatorFeatures.TIME_WEIGHTED_RATING_EVIDENCE].append(evidence_performances)

        return ratings

    def generate_future(self, matches: Optional[list[Match]] = None, df: Optional[pd.DataFrame] = None) -> dict[
        RatingEstimatorFeatures, list[float]]:
        raise NotImplementedError("This method is not implemented")

    def _generate_base_prior(self, player_id: str, base_prior_value: float, league: Optional[str],
                             position: Optional[str]) -> float:
        granularity_id = ""
        if self.by_league:
            league = league if league is not None else ""
            granularity_id += league
        if self.by_position:
            position = position if position is not None else ""
            granularity_id += position

        if player_id in self._player_prior_ratings and granularity_id in self._granularity_players and player_id in \
                self._granularity_players[granularity_id]:
            return self._player_prior_ratings[player_id]

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

    @property
    def estimator_features_out(self) -> list[RatingEstimatorFeatures]:
        return self._features_out

    @property
    def features_out(self) -> list[Union[RatingEstimatorFeatures, RatingHistoricalFeatures]]:
        return self._features_out
