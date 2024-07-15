import copy
from dataclasses import dataclass

from typing import Optional, Union

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.cross_validator.cross_validator import CrossValidator
from player_performance_ratings.ratings.performance_generator import (
    ColumnWeight,
    PerformancesGenerator,
)
from player_performance_ratings.ratings.enums import RatingKnownFeatures

from player_performance_ratings import PipelineFactory

from player_performance_ratings.tuner.utils import (
    add_params_from_search_range,
    ParameterSearchRange,
)

RC = RatingKnownFeatures


@dataclass
class PerformancesSearchRange:
    search_ranges: list[ParameterSearchRange]
    name: str = "performance"


class PerformancesGeneratorTuner:

    def __init__(
        self,
        performances_search_range: Union[
            list[PerformancesSearchRange], PerformancesSearchRange
        ],
        feature_names: Optional[Union[list[str], list[list[str]]]] = None,
        n_trials: int = 30,
    ):

        self.performances_search_ranges = (
            performances_search_range
            if isinstance(performances_search_range, list)
            else [performances_search_range]
        )
        self.feature_names = feature_names

        self.n_trials = n_trials
        self._scores = []

    def tune(
        self,
        df: pd.DataFrame,
        rating_idx: int,
        cross_validator: CrossValidator,
        pipeline_factory: PipelineFactory,
    ) -> PerformancesGenerator:

        if pipeline_factory.performances_generator is None:
            raise ValueError(
                "pipeline_factory.performances_generator is None. Please provide a performances_generator."
            )

        df = df.copy()

        pipeline_factory = copy.deepcopy(pipeline_factory)

        def objective(
            trial: BaseTrial,
            df: pd.DataFrame,
            pipeline_factory: PipelineFactory,
        ) -> float:

            best_pre_transformers = [
                copy.deepcopy(p)
                for p in pipeline_factory.performances_generator.original_transformers
            ]
            performances = [
                cw for cw in pipeline_factory.performances_generator.performances
            ]
            for performance_search_range in self.performances_search_ranges:
                performance_name = performance_search_range.name
                search_range = performance_search_range.search_ranges
                performances[rating_idx].weights = []
                raw_params = {}
                renamed_search = copy.deepcopy(search_range)
                for sr in renamed_search:
                    sr.name = f"{performance_name}__{sr.name}"
                add_params_from_search_range(
                    trial=trial,
                    params=raw_params,
                    parameter_search_range=renamed_search,
                )
                new_col_weights = self._create_column_weights(
                    params=raw_params,
                    remove_string=f"{performance_name}__",
                    search_range=search_range,
                )
                for new_col_weight in new_col_weights:
                    if new_col_weight.name in [
                        cw.name for cw in performances[rating_idx].weights
                    ]:
                        performances[rating_idx].weights[
                            [cw.name for cw in performances[rating_idx].weights].index(
                                new_col_weight.name
                            )
                        ] = new_col_weight
                    else:
                        performances[rating_idx].weights.append(new_col_weight)

            performances_generator = PerformancesGenerator(
                performances=performances,
                transformers=best_pre_transformers,
                auto_transform_performance=pipeline_factory.performances_generator.auto_transform_performance,
            )
            pipeline = pipeline_factory.create(
                performances_generator=performances_generator
            )
            return pipeline.cross_validate_score(
                df=df,
                cross_validator=cross_validator,
                create_performance=True,
                create_rating_features=True,
            )

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(
            direction=direction, study_name=study_name, sampler=sampler
        )
        callbacks = []
        study.optimize(
            lambda trial: objective(trial, df=df, pipeline_factory=pipeline_factory),
            n_trials=self.n_trials,
            callbacks=callbacks,
        )

        best_params = study.best_params
        best_column_weights = self._select_best_performances(all_params=best_params)
        performances = [
            cw for cw in pipeline_factory.performances_generator.performances
        ]
        performances[rating_idx].weights = best_column_weights[rating_idx]
        return PerformancesGenerator(
            performances=performances,
            transformers=pipeline_factory.performances_generator.original_transformers,
        )

    def _create_column_weights(
        self, params: dict, remove_string: str, search_range: list[ParameterSearchRange]
    ) -> list[ColumnWeight]:

        sum_weights = sum([v for _, v in params.items()])
        column_weights = []
        lower_is_better = False
        for name, weight in params.items():
            for search in search_range:
                if search.name == name.split("__")[1]:
                    lower_is_better = search.lower_is_better
                    break
            column_weights.append(
                ColumnWeight(
                    name=name.replace(remove_string, ""),
                    weight=weight / sum_weights,
                    lower_is_better=lower_is_better,
                )
            )

        return column_weights

    def _select_best_performances(self, all_params: dict) -> list[list[ColumnWeight]]:
        best_column_weights = []
        for performance_search_range in self.performances_search_ranges:
            performance_name = performance_search_range.name
            search_range = performance_search_range.search_ranges
            column_weights = []
            sum_weights = 0
            for param in all_params:

                if f"{performance_name}__" in param:
                    sum_weights += all_params[param]

            for param in all_params:

                if f"{performance_name}__" in param:
                    lower_is_better = False
                    for s in search_range:
                        if s.name == param.split("__")[1]:
                            lower_is_better = s.lower_is_better
                            break

                    column_weights.append(
                        ColumnWeight(
                            name=param.replace(f"{performance_name}__", ""),
                            weight=all_params[param] / sum_weights,
                            lower_is_better=lower_is_better,
                        )
                    )

            best_column_weights.append(column_weights)

        return best_column_weights
