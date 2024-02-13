import copy

from typing import Optional, Union

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.cross_validator.cross_validator import CrossValidator
from player_performance_ratings.ratings import ColumnWeight, PerformancesGenerator
from player_performance_ratings.ratings.enums import RatingEstimatorFeatures
from player_performance_ratings.transformation.base_transformer import BaseTransformer

from player_performance_ratings import PipelineFactory

from player_performance_ratings.tuner.utils import add_params_from_search_range, ParameterSearchRange

RC = RatingEstimatorFeatures


class PerformancesGeneratorTuner:

    def __init__(self,
                 performances_weight_search_ranges: dict[str, list[ParameterSearchRange]],
                 feature_names: Optional[Union[list[str], list[list[str]]]] = None,
                 lower_is_better_features: Optional[list[str]] = None,
                 n_trials: int = 30,
                 ):

        self.performances_weight_search_ranges = performances_weight_search_ranges
        self.feature_names = feature_names
        self.lower_is_better_features = lower_is_better_features

        self.n_trials = n_trials
        self._scores = []

    def tune(self,
             df: pd.DataFrame,
             rating_idx: int,
             cross_validator: CrossValidator,
             pipeline_factory: PipelineFactory,
             ) -> PerformancesGenerator:

        if pipeline_factory.performances_generator is None:
            raise ValueError("pipeline_factory.performances_generator is None. Please provide a performances_generator.")

        df = df.copy()

        column_names = [r.column_names for r in pipeline_factory.rating_generators]

        pipeline_factory = copy.deepcopy(pipeline_factory)

        def objective(trial: BaseTrial,
                      df: pd.DataFrame,
                      pipeline_factory: PipelineFactory,
                      ) -> float:

            best_pre_transformers = copy.deepcopy(pipeline_factory.performances_generator.pre_transformations)
            column_weights = [cw for cw in pipeline_factory.performances_generator.column_weights]
            for performance_name, search_range in self.performances_weight_search_ranges.items():
                column_weights[rating_idx] = []
                raw_params = {}
                renamed_search = copy.deepcopy(search_range)
                for sr in renamed_search:
                    sr.name = f"{performance_name}__{sr.name}"
                add_params_from_search_range(trial=trial, params=raw_params, parameter_search_range=renamed_search)
                new_col_weights = self._create_column_weights(params=raw_params, remove_string=f"{performance_name}__")
                for new_col_weight in new_col_weights:
                    if new_col_weight.name in [cw.name for cw in column_weights[rating_idx]]:
                        column_weights[rating_idx][[cw.name for cw in column_weights[rating_idx]].index(new_col_weight.name)] = new_col_weight
                    else:
                        column_weights[rating_idx].append(new_col_weight)

                col_names = [r.column_names for r in pipeline_factory.rating_generators]

                performances_generator = PerformancesGenerator(
                column_names = col_names,
                column_weights = column_weights,
                pre_transformations = best_pre_transformers,
            )
            pipeline = pipeline_factory.create(performances_generator=performances_generator)
            return pipeline.cross_validate_score(df=df, cross_validator=cross_validator,
                                                 create_performance=True, create_rating_features=True)

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(
            lambda trial: objective(trial, df=df,
                                    pipeline_factory=pipeline_factory),
            n_trials=self.n_trials, callbacks=callbacks)

        best_params = study.best_params
        best_column_weights = self._select_best_column_weights(all_params=best_params)

        return PerformancesGenerator(column_weights=best_column_weights,
                                     column_names=column_names,
                                     pre_transformations=pipeline_factory.performances_generator.pre_transformations,
                                     )

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
                    column_weights.append(ColumnWeight(name=param.replace(f"{performance_name}__", ""),
                                                       weight=all_params[param] / sum_weights))

            best_column_weights.append(column_weights)

        return best_column_weights
