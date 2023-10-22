import copy
import inspect
from typing import Optional, Match, Tuple, Union

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.scorer.score import LogLossScorer, BaseScorer
from player_performance_ratings.transformers.base_transformer import BaseTransformer
from player_performance_ratings.tuner.base_tuner import ParameterSearchRange, add_custom_hyperparams, \
    TransformerTuner

from player_performance_ratings.transformers.common import ColumnWeight

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

        def objective(trial: BaseTrial,
                      df: pd.DataFrame,
                      pre_transformer_search_ranges: list[
                          Tuple[BaseTransformer, list[ParameterSearchRange]]],
                      match_predictor: MatchPredictor,
                      scorer: BaseScorer,
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

        best_transformers = []
        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(
            lambda trial: objective(trial, df, self.pre_transformer_search_ranges, self.match_predictor,
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
