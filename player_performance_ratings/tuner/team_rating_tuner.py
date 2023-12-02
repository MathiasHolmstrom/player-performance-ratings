import copy
import inspect
from typing import Optional

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings.data_structures import Match
from player_performance_ratings.ratings.match_rating import TeamRatingGenerator
from player_performance_ratings.scorer.score import BaseScorer, LogLossScorer
from player_performance_ratings.tuner.base_tuner import ParameterSearchRange, add_params_from_search_range

DEFAULT_TEAM_SEARCH_RANGES = [
    ParameterSearchRange(
        name='certain_weight',
        type='uniform',
        low=0.7,
        high=0.95
    ),
    ParameterSearchRange(
        name='certain_days_ago_multiplier',
        type='uniform',
        low=0.02,
        high=.12,
    ),
    ParameterSearchRange(
        name='max_days_ago',
        type='uniform',
        low=40,
        high=200,
    ),
    ParameterSearchRange(
        name='max_certain_sum',
        type='uniform',
        low=20,
        high=70,
    ),
    ParameterSearchRange(
        name='certain_value_denom',
        type='uniform',
        low=10,
        high=50
    ),
    ParameterSearchRange(
        name='reference_certain_sum_value',
        type='uniform',
        low=0.4,
        high=5
    ),
    ParameterSearchRange(
        name='rating_change_multiplier',
        type='uniform',
        low=25,
        high=140
    ),
]


class TeamRatingTuner():

    def __init__(self,
                 match_predictor: MatchPredictor,
                 search_ranges: Optional[list[ParameterSearchRange]] = None,
                 scorer: Optional[BaseScorer] = None,
                 n_trials: int = 30,
                 ):
        self.search_ranges = search_ranges or DEFAULT_TEAM_SEARCH_RANGES
        self.match_predictor = match_predictor
        self.scorer = scorer or LogLossScorer(target=self.match_predictor.predictor.target,
                                              pred_column=self.match_predictor.predictor.pred_column
                                              )
        self.n_trials = n_trials

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> TeamRatingGenerator:
        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:
            params = {}
            params = add_params_from_search_range(params=params,
                                                  trial=trial,
                                                  parameter_search_range=self.search_ranges)

            performance_predictor = self.match_predictor.rating_generator.team_rating_generator.performance_predictor
            performance_predictor_params =list(
            inspect.signature(performance_predictor.__class__.__init__).parameters.keys())[1:]

            for param in params.copy():
                if param in performance_predictor_params:
                    performance_predictor.__setattr__(param, params[param])
                    params.pop(param)

            team_rating_generator = TeamRatingGenerator(**params,
                                   performance_predictor=performance_predictor)
            match_predictor = copy.deepcopy(self.match_predictor)
            match_predictor.rating_generator.team_rating_generator = team_rating_generator
            df_with_prediction = match_predictor.generate(df=df, matches=matches)
            return self.scorer.score(df_with_prediction, classes_=match_predictor.predictor.classes_)

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(lambda trial: objective(trial, df), n_trials=self.n_trials, callbacks=callbacks)

        best_params = study.best_params

        performance_predictor = self.match_predictor.rating_generator.team_rating_generator.performance_predictor
        performance_predictor_params = list(
            inspect.signature(performance_predictor.__class__.__init__).parameters.keys())[1:]

        for param in best_params.copy():
            if param in performance_predictor_params:
                performance_predictor.__setattr__(param, best_params[param])
                best_params.pop(param)

        return TeamRatingGenerator(**best_params,
                                   performance_predictor=performance_predictor)
