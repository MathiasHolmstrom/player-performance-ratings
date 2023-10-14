import inspect
from dataclasses import dataclass

from typing import Any, Optional
import networkx as nx
import scipy
from sklearn.cluster import SpectralClustering

from src.scorer.base_score import BaseScorer, LogLossScorer

print(scipy.__version__)

import pandas as pd

from src.predictor.match_predictor import MatchPredictor
from src.ratings.data_prepararer import MatchGenerator
from src.ratings.data_structures import ColumnNames, Match
from src.ratings.enums import RatingColumnNames
from src.ratings.factory.match_generator_factory import RatingGeneratorFactory
from src.ratings.league_identifier import LeagueIdentifier
from src.ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator


@dataclass
class LeagueH2H:
    league: str
    opp_league: str
    weight: float
    mean_rating_change: float


RC = RatingColumnNames


class StartRatingTuner():

    def __init__(self,
                 column_names: ColumnNames,
                 iterations: int = 4,
                 learning_step: float = 20,
                 scorer: Optional[BaseScorer] = None,
                 match_predictor: Optional[MatchPredictor] = None,
                 start_rating_generator: Optional[StartRatingGenerator] = None,
                 player_rating_generator: Optional[PlayerRatingGenerator] = None,
                 performance_predictor: Optional[PerformancePredictor] = None,
                 team_rating_generator: Optional[TeamRatingGenerator] = None,
                 league_identifier: Optional[LeagueIdentifier] = None,

                 ):

        rating_column_names = [RC.rating_difference, RC.player_rating_change, RC.player_league, RC.opponent_league]
        self.iterations = iterations
        self.multiplier = learning_step
        self.match_predictor = match_predictor or MatchPredictor(column_names=column_names,
                                                                 rating_features=rating_column_names)
        self.scorer = scorer or LogLossScorer(target=self.match_predictor.predictor.target, weight_cross_league=3,
                                              prob_column_name=self.match_predictor.predictor.prob_column_name)

        self.start_rating_generator = start_rating_generator or StartRatingGenerator()
        self.player_rating_generator = player_rating_generator or PlayerRatingGenerator()
        self.performance_predictor = performance_predictor or PerformancePredictor()
        self.team_rating_generator = team_rating_generator or TeamRatingGenerator()
        self.league_identifier = league_identifier or LeagueIdentifier()
        self.column_names = column_names
        self._scores = []
        self._start_ratings: list[dict[str, float]] = []

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> dict[str, float]:
        if matches is None:
            match_generator = MatchGenerator(column_names=self.column_names)
            matches = match_generator.generate(df=df)

        start_rating_init_params = list(
            inspect.signature(self.start_rating_generator.__class__.__init__).parameters.keys())[1:]
        new_start_ratings = self.start_rating_generator.league_ratings.copy()
        for iteration in range(self.iterations + 1):
            start_rating_instance_variables = {attr: getattr(self.start_rating_generator, attr) for attr in
                                               dir(self.start_rating_generator) if
                                               attr in start_rating_init_params and attr != 'league_ratings'}

            start_rating_generator = StartRatingGenerator(league_ratings=new_start_ratings,
                                                          **start_rating_instance_variables)
            rating_generator_factory = RatingGeneratorFactory(
                start_rating_generator=start_rating_generator,
                team_rating_generator=self.team_rating_generator,
                player_rating_generator=self.player_rating_generator,
                performance_predictor=self.performance_predictor,
            )
            rating_generator = rating_generator_factory.create()
            self.match_predictor.rating_generator = rating_generator
            df = self.match_predictor.generate(df=df, matches=matches)

            score = self.scorer.score(df)
            self._scores.append(score)
            self._start_ratings.append(new_start_ratings.copy())
            if iteration == self.iterations:
                min_idx = self._scores.index(min(self._scores))
                return self._start_ratings[min_idx]
            new_start_ratings = self._optimize_start_ratings(df)

    def _optimize_start_ratings(self, df: pd.DataFrame) -> dict[str, float]:
        new_start_ratings = self.match_predictor.rating_generator.team_rating_generator.player_rating_generator.start_rating_generator.league_ratings

        league_rating_changes = (df
        .groupby([RatingColumnNames.player_league, RatingColumnNames.opponent_league])
        .agg(
            {
                RatingColumnNames.player_rating_change: 'mean',
                self.column_names.player_id: 'count'
            }
        )
        .reset_index()
        .rename(
            columns={
                RatingColumnNames.player_rating_change: 'mean_rating_change',
                self.column_names.player_id: 'count'
            })
        )
        leagues = league_rating_changes[RatingColumnNames.player_league].unique().tolist()

        league_opp_league_h2hs: dict[str, dict[str, LeagueH2H]] = {}
        league_to_played_against_leagues = {}

        for league in leagues:
            league_to_played_against_leagues[league] = \
                league_rating_changes[league_rating_changes[RatingColumnNames.player_league] == league][
                    RatingColumnNames.opponent_league].unique().tolist()

            league_opp_league_h2hs[league] = {}
            for opp_league in leagues:
                if league == opp_league:
                    continue

                rows = league_rating_changes[(league_rating_changes[RatingColumnNames.player_league] == league) &
                                             (league_rating_changes[
                                                  RatingColumnNames.opponent_league] == opp_league)]
                if len(rows) == 0:
                    weight = 0
                    mean_rating_change = 0
                else:
                    count = rows['count'].iloc[0]
                    weight = min(1, count / 100)
                    mean_rating_change = rows['mean_rating_change'].iloc[0]

                league_h2h = LeagueH2H(
                    weight=weight,
                    mean_rating_change=mean_rating_change,
                    league=league,
                    opp_league=opp_league,
                )
                league_opp_league_h2hs[league][opp_league] = league_h2h

        final_league_opp_league_h2hs: dict[str, dict[str, LeagueH2H]] = {}

        league_sum_final_weights = {}
        for league, opp_league_h2h in league_opp_league_h2hs.items():
            league_sum_final_weights[league] = 0
            final_league_opp_league_h2hs[league] = {}
            for opp_league, h2h in opp_league_h2h.items():
                if h2h.weight < 1:
                    league_against = league_to_played_against_leagues[league]
                    opp_played_against = league_to_played_against_leagues[opp_league]
                    shared_leagues = [l for l in league_against if
                                      l in opp_played_against and l not in (league, opp_league)]

                    relative_shared_weighted_mean = 0
                    shared_weight = 0
                    for shared_league in shared_leagues:
                        opp_vs_shared_h2h = league_opp_league_h2hs[opp_league][shared_league]
                        league_vs_shared_h2h = league_opp_league_h2hs[league][shared_league]

                        sum_rating_change = league_vs_shared_h2h.mean_rating_change - opp_vs_shared_h2h.mean_rating_change

                        relative_shared_weighted_mean += min(
                            opp_vs_shared_h2h.weight, league_vs_shared_h2h.weight) * sum_rating_change
                        shared_weight += min(
                            opp_vs_shared_h2h.weight, league_vs_shared_h2h.weight)

                weighted_mean = h2h.mean_rating_change * h2h.weight + min(1,
                                                                          shared_weight) * relative_shared_weighted_mean
                final_weight = min(1, h2h.weight + shared_weight)
                final_league_h2h = LeagueH2H(
                    weight=final_weight,
                    mean_rating_change=weighted_mean,
                    league=league,
                    opp_league=opp_league,
                )
                final_league_opp_league_h2hs[league][opp_league] = final_league_h2h
                league_sum_final_weights[league] += final_weight

        for league, final_opp_league_h2h in final_league_opp_league_h2hs.items():

            if league not in new_start_ratings:
                continue
            start_rating_rating_change = 0

            for opp_league, final_h2h in final_opp_league_h2h.items():
                start_rating_rating_change += final_h2h.weight / league_sum_final_weights[
                    league] * final_h2h.mean_rating_change * self.multiplier

            new_start_ratings[league] += start_rating_rating_change

        return new_start_ratings

    def _get_cross_league_matches(self, matches: list[Match]) -> list[Match]:
        cross_league_matches = []
        for match in matches:
            if len(match.teams) == 2:
                if match.teams[0].league != match.teams[1].league:
                    cross_league_matches.append(match)

        return cross_league_matches

    def _improve_start_rating(self, original_league_ratings: dict[str, float]) -> dict[str, float]:
        return {'lfl': 2}


class ParameterTuner():

    def __init__(self,
                 performance_params: Optional[dict[str, list[Any]]] = None,
                 rating_params: Optional[dict[str, list[Any]]] = None,
                 start_rating_params: Optional[dict[str, list[Any]]] = None
                 ):
        self.performance_params = performance_params or {}
        self.rating_params = rating_params or {}
        self.start_rating_params = start_rating_params or {}

    def tune(self, df: pd.DataFrame):
        rating_generator_factory = RatingGeneratorFactory()
        match_predictor = MatchPredictor()
        parameter_tuning = ParameterTuner()

        df = match_predictor.generate(df=df)

    @property
    def best_model(self) -> MatchPredictor:
        return


class AutoPredictor():

    def __init__(self,
                 column_names: ColumnNames,
                 hyperparameter_tuning: bool = True

                 ):
        self.column_names = column_names
        self.hyperparameter_tuning = hyperparameter_tuning

    def predict(self, df: pd.DataFrame):
        if self.hyperparameter_tuning:
            tuner = ParameterTuner()

        parameter_tuning.tune(df=df)
        parameter_tuning.best_params
