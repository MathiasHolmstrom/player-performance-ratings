import copy
import inspect
import logging
from typing import Optional, Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings import BaseTransformer
from player_performance_ratings.data_structures import Match
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.match_rating.start_rating.start_rating_generator import \
    StartRatingGenerator
from player_performance_ratings.ratings.rating_generator import OpponentAdjustedRatingGenerator
from player_performance_ratings.scorer.score import BaseScorer
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.rating_generator_tuner.start_rating_optimizer import StartLeagueRatingOptimizer
from player_performance_ratings.tuner.base_tuner import ParameterSearchRange, add_params_from_search_range

logging.basicConfig(level=logging.INFO)

RC = RatingColumnNames

DEFAULT_START_RATING_SEARCH_RANGE = [
    ParameterSearchRange(
        name='league_quantile',
        type='uniform',
        low=0.12,
        high=.4,
    ),
    ParameterSearchRange(
        name='min_count_for_percentiles',
        type='int',
        low=50,
        high=200,
    ),
    ParameterSearchRange(
        name='team_rating_subtract',
        type='int',
        low=20,
        high=300
    ),
    ParameterSearchRange(
        name='team_weight',
        type='uniform',
        low=0,
        high=0.7
    )
]


class StartRatingTuner():

    def __init__(self,
                 search_ranges: Optional[list[ParameterSearchRange]] = None,
                 start_rating_parameters: Optional[dict[str, Any]] = None,
                 start_rating_optimizer: Optional[StartLeagueRatingOptimizer] = None,
                 n_trials: int = 8,
                 ):

        self.start_rating_parameters = start_rating_parameters
        self.start_rating_optimizer = start_rating_optimizer
        self.n_trials = n_trials
        self.search_ranges = search_ranges or DEFAULT_START_RATING_SEARCH_RANGE

    def tune(self,
             df: pd.DataFrame,
             rating_generator: OpponentAdjustedRatingGenerator,
             rating_index: int,
             scorer: BaseScorer,
             match_predictor_factory: MatchPredictorFactory,
             matches: list[Match]
             ) -> StartRatingGenerator:

        def objective(trial: BaseTrial, df: pd.DataFrame, league_start_ratings: dict[str, float]) -> float:

            params = self.start_rating_parameters or {}
            if league_start_ratings != {}:
                params['league_ratings'] = league_start_ratings

            params = add_params_from_search_range(params=params,
                                                  trial=trial,
                                                  parameter_search_range=self.search_ranges,
                                                  )
            params = self._add_start_rating_hyperparams(params=params, trial=trial,
                                                        league_start_ratings=league_start_ratings)
            start_rating_generator = StartRatingGenerator(**params)
            rating_g = copy.deepcopy(rating_generator)
            rating_g.team_rating_generator.start_rating_generator = start_rating_generator

            match_predictor = match_predictor_factory.create(
                idx_rating_generator=(rating_index, rating_g),
            )
            df_with_prediction = match_predictor.generate_historical(df=df, store_ratings=False)
            return scorer.score(df_with_prediction, classes_=match_predictor.predictor.classes_)

        if self.start_rating_optimizer:
            optimized_league_ratings = self.start_rating_optimizer.optimize(df,
                                                                            matches=matches)
        else:
            optimized_league_ratings = rating_generator.team_rating_generator.start_rating_generator.league_ratings

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        # logging.info(f"start league ratings {optimized_league_ratings}")
        study.optimize(lambda trial: objective(trial, df, league_start_ratings=optimized_league_ratings),
                       n_trials=self.n_trials, callbacks=callbacks)
        start_rating_generator_params = list(
            inspect.signature(StartRatingGenerator().__class__.__init__).parameters.keys())[1:]
        params = {attr: getattr(StartRatingGenerator(), attr) for attr in
                  dir(StartRatingGenerator()) if
                  attr in start_rating_generator_params}
        best_params = {k: v for k, v in study.best_params.items() if k in params}
        best_league_ratings = {k: v for k, v in study.best_params.items() if k not in best_params}
        return StartRatingGenerator(league_ratings=best_league_ratings, **best_params)

    def _add_start_rating_hyperparams(self, params, trial, league_start_ratings):
        params['league_ratings'] = {}
        for league, rating in league_start_ratings.items():
            low_start_rating = league_start_ratings[league] - 120
            high_start_rating = league_start_ratings[league] + 120

            params['league_ratings'][league] = trial.suggest_int(league, low=low_start_rating, high=high_start_rating)

        return params


def _get_cross_league_matches(self, matches: list[Match]) -> list[Match]:
    cross_league_matches = []
    for match in matches:
        if len(match.teams) == 2:
            if match.teams[0].league != match.teams[1].league:
                cross_league_matches.append(match)

    return cross_league_matches
