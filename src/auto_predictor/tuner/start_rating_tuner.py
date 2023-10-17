import copy
from typing import Optional

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from src.auto_predictor.tuner.base_tuner import ParameterSearchRange, add_custom_hyperparams
from src.predictor.match_predictor import MatchPredictor
from src.ratings.data_structures import ColumnNames, Match
from src.ratings.enums import RatingColumnNames
from src.ratings.factory.match_generator_factory import RatingGeneratorFactory
from src.ratings.league_identifier import LeagueIdentifier
from src.ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator
from src.ratings.data_prepararer import MatchGenerator
from src.scorer.base_score import BaseScorer, LogLossScorer
import inspect
from dataclasses import dataclass
import random
from itertools import combinations


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
                 match_predictor: MatchPredictor,
                 search_ranges: Optional[list[ParameterSearchRange]] = None,
                 iterations: int = 4,
                 learning_step: float = 10,
                 final_adj_iterations: int = 6,
                 scorer: Optional[BaseScorer] = None,
                 n_trials: int = 8,
                 ):

        self.final_adj_iterations = final_adj_iterations
        rating_column_names = [RC.rating_difference, RC.player_rating_change, RC.player_league, RC.opponent_league]
        self.iterations = iterations
        self.learning_step = learning_step
        self.n_trials = n_trials
        self.match_predictor = match_predictor
        self.search_ranges = search_ranges
        self.scorer = scorer or LogLossScorer(target=self.match_predictor.predictor.target, weight_cross_league=2,
                                              pred_column=self.match_predictor.predictor.pred_column)

        self.column_names = column_names
        self._scores = []

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> StartRatingGenerator:

        def objective(trial: BaseTrial, df: pd.DataFrame, league_start_ratings: dict[str, float]) -> float:
            params = {'league_ratings': league_start_ratings}
            params = add_custom_hyperparams(params=params,
                                            trial=trial,
                                            parameter_search_range=self.search_ranges,
                                            )
            start_rating_generator = StartRatingGenerator(**params)
            match_predictor = copy.deepcopy(self.match_predictor)
            match_predictor.rating_generator.team_rating_generator.player_rating_generator.start_rating_generator = start_rating_generator
            df = match_predictor.generate(df=df)
            return self.scorer.score(df)

        if matches is None:
            match_generator = MatchGenerator(column_names=self.column_names)
            matches = match_generator.generate(df=df)

        new_league_ratings = {}
        best_models = []
        for iteration in range(self.iterations + 1):

            if self.search_ranges is None:
                start_rating_generator = StartRatingGenerator(**new_league_ratings)
                match_predictor = copy.deepcopy(self.match_predictor)
                match_predictor.rating_generator.team_rating_generator.player_rating_generator.start_rating_generator = start_rating_generator
                df = match_predictor.generate(df=df)
                score = self.scorer.score(df)
                self._scores.append(score)
                best_models.append(StartRatingGenerator(**new_league_ratings))

            else:
                direction = "minimize"
                study_name = "optuna_study"
                optuna_seed = 12
                sampler = TPESampler(seed=optuna_seed)
                study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
                callbacks = []
                study.optimize(lambda trial: objective(trial, df, league_start_ratings=new_league_ratings),
                               n_trials=self.n_trials, callbacks=callbacks)
                best_params= study.best_params
                best_params['league_ratings'] = new_league_ratings
                best_models.append(StartRatingGenerator(**best_params))
                self._scores.append(study.best_value)

            if iteration == self.iterations:
                min_idx = self._scores.index(min(self._scores))
                return best_models[min_idx]


            start_rating_generator = StartRatingGenerator(**study.best_params)
            match_predictor = copy.deepcopy(self.match_predictor)
            match_predictor.rating_generator.team_rating_generator.player_rating_generator.start_rating_generator = start_rating_generator
            df = match_predictor.generate(df=df, matches=matches)

            new_league_ratings = self._optimize_start_ratings(df, match_predictor=match_predictor)

    def _optimize_start_ratings(self, df: pd.DataFrame, match_predictor: MatchPredictor) -> dict[str, float]:
        player_rating_generator = match_predictor.rating_generator.team_rating_generator.player_rating_generator
        new_start_ratings = match_predictor.rating_generator.team_rating_generator.player_rating_generator.start_rating_generator.league_ratings

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
                    league] * final_h2h.mean_rating_change * self.learning_step

            new_start_ratings[league] += start_rating_rating_change

        for iter in range(self.final_adj_iterations):
            random.seed(iter)
            league_combinations = list(combinations(leagues, 2))
            random.shuffle(league_combinations)
            # Iterate over the combinations
            for league_pair in league_combinations:
                league = league_pair[0]
                opp_league = league_pair[1]
                if league == opp_league:
                    continue
                opp_league_h2h = league_opp_league_h2hs[league]
                if league not in new_start_ratings:
                    continue

                h2h = opp_league_h2h[opp_league]

                if opp_league not in new_start_ratings:
                    continue

                start_rating_diff = new_start_ratings[league] - new_start_ratings[opp_league]
                expected_start_rating_diff = h2h.mean_rating_change * player_rating_generator.rating_change_multiplier * 0.4
                new_start_ratings[league] += (expected_start_rating_diff - start_rating_diff) * h2h.weight * 0.5

        return new_start_ratings

    def _get_cross_league_matches(self, matches: list[Match]) -> list[Match]:
        cross_league_matches = []
        for match in matches:
            if len(match.teams) == 2:
                if match.teams[0].league != match.teams[1].league:
                    cross_league_matches.append(match)

        return cross_league_matches
