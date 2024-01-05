import copy

from typing import Optional, Union

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.data_structures import Match
from player_performance_ratings.ratings import ColumnWeight, PerformancesGenerator
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.scorer.score import BaseScorer
from player_performance_ratings.transformation.base_transformer import BaseTransformer

from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory

from player_performance_ratings.tuner.utils import add_params_from_search_range, ParameterSearchRange

RC = RatingColumnNames


class PerformancesGeneratorTuner:

    def __init__(self,
                 performances_weight_search_ranges: dict[str, list[ParameterSearchRange]],
                 pre_transformations: Optional[list[BaseTransformer]] = None,
                 feature_names: Optional[Union[list[str], list[list[str]]]] = None,
                 lower_is_better_features: Optional[list[str]] = None,
                 n_trials: int = 30,
                 ):

        self.performances_weight_search_ranges = performances_weight_search_ranges
        self.pre_transformations = pre_transformations
        self.feature_names = feature_names
        self.lower_is_better_features = lower_is_better_features

        self.n_trials = n_trials
        self._scores = []

    def tune(self, df: pd.DataFrame,
             scorer: BaseScorer,
             match_predictor_factory: MatchPredictorFactory,
             matches: Optional[list[list[Match]]] = None,
             ) -> PerformancesGenerator:

        df = df.copy()

        if not matches:
            matches = None

        column_names = [r.column_names for r in match_predictor_factory.rating_generators]


        match_predictor_factory = copy.deepcopy(match_predictor_factory)

        def objective(trial: BaseTrial,
                      df: pd.DataFrame,
                      match_predictor_factory: MatchPredictorFactory,
                      scorer: BaseScorer,
                      ) -> float:


            column_weights = []
            for performance_name, search_range in self.performances_weight_search_ranges.items():
                raw_params = {}
                renamed_search = copy.deepcopy(search_range)
                for sr in renamed_search:
                    sr.name = f"{performance_name}__{sr.name}"
                add_params_from_search_range(trial=trial, params=raw_params, parameter_search_range=renamed_search)
                column_weights.append(
                    self._create_column_weights(params=raw_params, remove_string=f"{performance_name}__"))

            col_names = [r.column_names for r in match_predictor_factory.rating_generators]
            performances_generator = PerformancesGenerator(
                column_names=col_names,
                column_weights=column_weights,
                pre_transformations=self.pre_transformations
            )
            match_predictor = match_predictor_factory.create(performances_generator=performances_generator)

            df_with_prediction = match_predictor.generate_historical(df=df, matches=matches, store_ratings=False)
            test_df = df_with_prediction[
                df_with_prediction[match_predictor.date_column_name] > match_predictor.train_split_date]
            return scorer.score(test_df, classes_=match_predictor.predictor.classes_)

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(
            lambda trial: objective(trial, df=df,
                                    match_predictor_factory=match_predictor_factory, scorer=scorer),
            n_trials=self.n_trials, callbacks=callbacks)

        best_params = study.best_params
        best_column_weights = self._select_best_column_weights(all_params=best_params)


        return PerformancesGenerator(column_weights=best_column_weights, column_names=column_names,
                                     pre_transformations=self.pre_transformations)

    def _create_column_weights(self, params: dict, remove_string: str) -> list[ColumnWeight]:

        sum_weights = sum([v for _, v in params.items()])
        column_weights = []
        for name, weight in params.items():
            column_weights.append(ColumnWeight(name=name.replace(remove_string, ""), weight=weight / sum_weights))

        return column_weights

    def _select_best_column_weights(self, all_params: dict) -> list[list[ColumnWeight]]:
        best_column_weights = []
        for performance_name in self.performances_weight_search_ranges:
            column_weights = []
            sum_weights = 0
            for param in all_params:

                if f"{performance_name}__" in param:
                    sum_weights += all_params[param]

            for param in all_params:
                if f"{performance_name}__" in param:
                    column_weights.append(ColumnWeight(name=param.replace(f"{performance_name}__", ""), weight=all_params[param] / sum_weights))

            best_column_weights.append(column_weights)

        return best_column_weights
