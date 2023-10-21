import copy
import logging
from typing import Optional, Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.tuner.optimizer.start_rating_optimizer import StartLeagueRatingOptimizer
from player_performance_ratings.tuner.base_tuner import ParameterSearchRange, add_custom_hyperparams
from src import MatchPredictor
from src import ColumnNames, Match
from src import RatingColumnNames
from src import StartRatingGenerator
from src import MatchGenerator
from src import BaseScorer, LogLossScorer

logging.basicConfig(level=logging.INFO)

RC = RatingColumnNames


class StartRatingTuner():

    def __init__(self,
                 column_names: ColumnNames,
                 match_predictor: MatchPredictor,
                 search_ranges: Optional[list[ParameterSearchRange]] = None,
                 start_rating_parameters: Optional[dict[str, Any]] = None,
                 start_rating_optimizer: Optional[StartLeagueRatingOptimizer] = None,
                 scorer: Optional[BaseScorer] = None,
                 n_trials: int = 0,
                 ):

        self.start_rating_parameters = start_rating_parameters
        rating_column_names = [RC.rating_difference, RC.player_rating_change, RC.player_league, RC.opponent_league]
        self.start_rating_optimizer = start_rating_optimizer or StartLeagueRatingOptimizer(column_names=column_names,
                                                                                           match_predictor=match_predictor)
        self.n_trials = n_trials
        self.match_predictor = match_predictor
        self.search_ranges = search_ranges
        self.scorer = scorer or LogLossScorer(target=self.match_predictor.predictor.target, weight_cross_league=5,
                                              pred_column=self.match_predictor.predictor.pred_column)

        self.column_names = column_names

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> StartRatingGenerator:

        def objective(trial: BaseTrial, df: pd.DataFrame, league_start_ratings: dict[str, float]) -> float:
            params = self.start_rating_parameters or {}
            if league_start_ratings != {}:
                params['league_ratings'] = league_start_ratings

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

        if self.start_rating_optimizer:
            optimized_league_ratings = self.start_rating_optimizer.optimize(df,
                                                                            matches=matches)
        else:
            optimized_league_ratings = self.match_predictor.rating_generator.team_rating_generator.player_rating_generator.start_rating_generator.league_ratings

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        logging.info(f"start league ratings {optimized_league_ratings}")
        study.optimize(lambda trial: objective(trial, df, league_start_ratings=optimized_league_ratings),
                       n_trials=self.n_trials, callbacks=callbacks)

        return StartRatingGenerator(league_ratings=optimized_league_ratings, **study.best_params)


def _get_cross_league_matches(self, matches: list[Match]) -> list[Match]:
    cross_league_matches = []
    for match in matches:
        if len(match.teams) == 2:
            if match.teams[0].league != match.teams[1].league:
                cross_league_matches.append(match)

    return cross_league_matches
