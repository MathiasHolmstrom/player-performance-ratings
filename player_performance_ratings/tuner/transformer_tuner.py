import copy
import inspect
import logging
from typing import Optional, Tuple, Union, Literal

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.data_structures import Match
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import RatingGenerator
from player_performance_ratings.scorer.score import BaseScorer
from player_performance_ratings.transformation import ColumnWeight
from player_performance_ratings.transformation.base_transformer import BaseTransformer

from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory

from player_performance_ratings.tuner.utils import add_params_from_search_range, ParameterSearchRange, \
    create_pre_rating_search_range_for_auto

RC = RatingColumnNames


def insert_params_to_common_transformers(object: object, params, parameter_search_range: list[ParameterSearchRange]):
    if object.__class__.__name__ == "ColumnsWeighter":
        sum_weights = sum([params[p.name] for p in parameter_search_range])
        for p in parameter_search_range:
            params[p.name] = params[p.name] / sum_weights

        column_weights = [ColumnWeight(name=p.name, weight=params[p.name], **p.custom_params) for p in
                          parameter_search_range]

        for p in parameter_search_range:
            del params[p.name]
        params['column_weights'] = column_weights

    return params


def add_hyperparams_to_common_transformers(object: object, params: dict[str, Union[float, None, bool, int, str]],
                                           trial: BaseTrial,
                                           parameter_search_range: list[ParameterSearchRange]) -> dict:
    params = add_params_from_search_range(params=params, trial=trial, parameter_search_range=parameter_search_range)

    return insert_params_to_common_transformers(object=object, params=params,
                                                parameter_search_range=parameter_search_range)


class TransformerTuner:

    def __init__(self,
                 pre_or_post: Literal["pre_rating", "post_rating"],
                 transformer_search_ranges: Optional[list[Tuple[BaseTransformer, list[ParameterSearchRange]]]] = None,
                 feature_names: Optional[Union[list[str], list[list[str]]]] = None,
                 lower_is_better_features: Optional[list[str]] = None,
                 n_trials: int = 30,
                 ):

        if transformer_search_ranges is None and feature_names is None:
            raise ValueError("Either transformer_search_ranges or feature_names must be provided")

        self.transformer_search_ranges = transformer_search_ranges
        self.pre_or_post = pre_or_post
        self.feature_names = feature_names
        self.lower_is_better_features = lower_is_better_features

        self.n_trials = n_trials
        self._scores = []

    def tune(self, df: pd.DataFrame,
             scorer: BaseScorer,
             match_predictor_factory: MatchPredictorFactory,
             pre_rating_transformers: Optional[list[BaseTransformer]] = None,
             matches: Optional[list[list[Match]]] = None,
             rating_generators: Optional[list[RatingGenerator]] = None,
             ) -> list[BaseTransformer]:

        if not matches:
            matches = None

        if self.transformer_search_ranges is None:
            logging.info("Creating transformer search ranges")
            self.transformer_search_ranges = create_pre_rating_search_range_for_auto(
                feature_names=self.feature_names,
                column_names=match_predictor_factory.column_names,
                lower_is_better_features=self.lower_is_better_features
            )

        match_predictor_factory = copy.deepcopy(match_predictor_factory)

        def objective(trial: BaseTrial,
                      df: pd.DataFrame,
                      transformer_search_ranges: list[
                          Tuple[BaseTransformer, list[ParameterSearchRange]]],
                      match_predictor_factory: MatchPredictorFactory,
                      scorer: BaseScorer,
                      ) -> float:
            transformers = []
            for transformer, parameter_search_range in transformer_search_ranges:
                transformer_params = list(
                    inspect.signature(transformer.__class__.__init__).parameters.keys())[1:]
                params = {attr: getattr(transformer, attr) for attr in
                          dir(transformer) if
                          attr in transformer_params}
                params = add_hyperparams_to_common_transformers(object=transformer, params=params, trial=trial,
                                                                parameter_search_range=parameter_search_range)
                class_transformer = type(transformer)
                transformers.append(class_transformer(**params))

            # TODO: Fix it properly so it works with pre and post transformers after tuned
            if self.pre_or_post == "pre_rating":
                match_predictor = match_predictor_factory.create(pre_rating_transformers=transformers)
            else:
                match_predictor = match_predictor_factory.create(pre_rating_transformers=pre_rating_transformers,
                                                                 rating_generators=rating_generators,
                                                                 post_rating_transformers=transformers)

            df_with_prediction = match_predictor.generate_historical(df=df, matches=matches, store_ratings=False)
            return scorer.score(df_with_prediction, classes_=match_predictor.classes_)

        best_transformers = []
        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(
            lambda trial: objective(trial, df=df, transformer_search_ranges=self.transformer_search_ranges,
                                    match_predictor_factory=match_predictor_factory, scorer=scorer),
            n_trials=self.n_trials, callbacks=callbacks)

        best_params = study.best_params
        for transformer, search_range in self.transformer_search_ranges:
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
