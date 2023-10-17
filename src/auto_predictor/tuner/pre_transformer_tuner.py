import copy
import inspect
import logging
from dataclasses import dataclass
from typing import Optional, Match, Tuple, Literal, Union, Any

import optuna
import pandas as pd
import pendulum
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from src.auto_predictor.tuner.base_tuner import BaseTuner, ParameterSearchRange, add_custom_hyperparams, \
    TransformerTuner
from src.predictor.match_predictor import MatchPredictor
from src.predictor.ml_wrappers.base_wrapper import BaseMLWrapper
from src.ratings.data_prepararer import MatchGenerator
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.factory.match_generator_factory import RatingGeneratorFactory
from src.ratings.league_identifier import LeagueIdentifier
from src.ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator
from src.scorer.base_score import BaseScorer, LogLossScorer
from src.transformers import BaseTransformer
from src.transformers.common import ColumnWeight

RC = RatingColumnNames


def insert_params_to_common_transformers(object: object, params, parameter_search_range):
    if object.__class__.__name__ == "ColumnsWeighter":
        sum_weights = sum([params[p.name] for p in parameter_search_range])
        for p in parameter_search_range:
            params[p.name] = params[p.name] / sum_weights
        column_weights = [ColumnWeight(name=p.name, weight=params[p.name]) for p in
                          parameter_search_range]
        for p in parameter_search_range:
            del params[p.name]
        params['column_weights'] = column_weights

    return params


def add_hyperparams_to_common_transformers(object: object, params: dict[str, Union[float, None, bool, int, str]],
                                           trial: BaseTrial,
                                           parameter_search_range: list[ParameterSearchRange]) -> dict:
    params = add_custom_hyperparams(params=params, trial=trial, parameter_search_range=parameter_search_range)

    return insert_params_to_common_transformers(object=object, params=params,
                                                parameter_search_range=parameter_search_range)


def pre_transformer_objective(trial: BaseTrial,
                              df: pd.DataFrame,
                              pre_transformer_search_ranges: list[
                                  Tuple[BaseTransformer, list[ParameterSearchRange]]],
                              match_predictor: MatchPredictor,
                              scorer: BaseScorer

                              ) -> float:
    pre_rating_transformers = []
    for transformer, parameter_search_range in pre_transformer_search_ranges:
        transformer_params = list(
            inspect.signature(transformer.__class__.__init__).parameters.keys())[1:]
        params = {attr: getattr(transformer, attr) for attr in
                  dir(transformer) if
                  attr in transformer_params}
        params = add_hyperparams_to_common_transformers(object=transformer, params=params, trial=trial,
                                                        parameter_search_range=parameter_search_range)
        class_transformer = type(transformer)
        pre_rating_transformers.append(class_transformer(**params))

    match_predictor = copy.deepcopy(match_predictor)
    match_predictor.pre_rating_transformers = pre_rating_transformers

    df = match_predictor.generate(df=df)
    return scorer.score(df)


class PreTransformerTuner(TransformerTuner):

    def __init__(self,
                 pre_transformer_search_ranges: list[Tuple[BaseTransformer, list[ParameterSearchRange]]],
                 match_predictor: MatchPredictor,
                 n_trials: int = 30,
                 scorer: Optional[BaseScorer] = None,
                 ):
        self.pre_transformer_search_ranges = pre_transformer_search_ranges
        self.match_predictor = match_predictor

        self.n_trials = n_trials

        self.scorer = scorer or LogLossScorer(target=self.match_predictor.predictor.target,
                                              weight_cross_league=3,
                                              pred_column=self.match_predictor.predictor.pred_column
                                              )

        self.column_names = match_predictor.column_names
        self._scores = []

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> list[BaseTransformer]:

        best_transformers = []
        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(
            lambda trial: pre_transformer_objective(trial, df, self.pre_transformer_search_ranges, self.match_predictor,
                                                    self.scorer), n_trials=self.n_trials, callbacks=callbacks)

        best_params = study.best_params
        for transformer, search_range in self.pre_transformer_search_ranges:
            param_values = {p.name: best_params[p.name] for p in search_range if p.name in best_params}
            class_transformer = type(transformer)
            transformer_params = list(
                inspect.signature(transformer.__class__.__init__).parameters.keys())[1:]
            params = {attr: getattr(transformer, attr) for attr in
                      dir(transformer) if
                      attr in transformer_params}

            for k, v in param_values.items():
                params[k] = v

            params = insert_params_to_common_transformers(object=transformer, params=params,
                                                          parameter_search_range=search_range)
            best_transformers.append(class_transformer(**params))

        return best_transformers
