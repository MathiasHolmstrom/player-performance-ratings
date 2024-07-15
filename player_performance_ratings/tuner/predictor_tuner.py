import copy
import inspect
from typing import Optional

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial
from player_performance_ratings.transformers.base_transformer import BaseTransformer

from player_performance_ratings.cross_validator.cross_validator import CrossValidator
from player_performance_ratings import PipelineFactory, ColumnNames

from player_performance_ratings.predictor import BasePredictor


from player_performance_ratings.tuner.utils import (
    ParameterSearchRange,
    add_params_from_search_range,
    get_default_lgbm_regressor_search_range,
)


class PredictorTuner:

    def __init__(
        self,
        search_ranges: Optional[list[ParameterSearchRange]] = None,
        default_params: Optional[dict] = None,
        n_trials: int = 30,
    ):
        self.search_ranges = search_ranges
        self.default_params = default_params or {}
        self.n_trials = n_trials

    def tune(
        self,
        df: pd.DataFrame,
        pipeline_factory: PipelineFactory,
        cross_validator: CrossValidator,
    ) -> BasePredictor:

        deepest_estimator = pipeline_factory.predictor.estimator

        estimator_subclass_level = 0
        while hasattr(deepest_estimator, "estimator"):
            deepest_estimator = deepest_estimator.estimator
            estimator_subclass_level += 1

        if self.search_ranges is None and deepest_estimator.__class__.__name__ in (
            "LGBMRegressor",
            "LGBMClassifier",
        ):
            search_ranges = get_default_lgbm_regressor_search_range()
        elif self.search_ranges is None:
            raise ValueError(
                "search_ranges can't be None if estimator is not LGBMRegressor or LGBMClassifier"
            )
        else:
            search_ranges = self.search_ranges

        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:

            predictor = pipeline_factory.predictor

            if estimator_subclass_level == 0:
                param_names = list(
                    inspect.signature(
                        predictor.estimator.__class__.__init__
                    ).parameters.keys()
                )[1:]
                params = {
                    attr: getattr(predictor.estimator, attr)
                    for attr in param_names
                    if attr != "kwargs"
                }
                if "_other_params" in predictor.estimator.__dict__:
                    params.update(predictor.estimator._other_params)
            elif estimator_subclass_level == 1:
                param_names = list(
                    inspect.signature(
                        predictor.estimator.estimator.__class__.__init__
                    ).parameters.keys()
                )[1:]
                params = {
                    attr: getattr(predictor.estimator.estimator, attr)
                    for attr in param_names
                    if attr != "kwargs"
                }
                if "_other_params" in predictor.estimator.estimator.__dict__:
                    params.update(predictor.estimator.estimator._other_params)
            elif estimator_subclass_level == 2:
                param_names = list(
                    inspect.signature(
                        predictor.estimator.estimator.estimator.__class__.__init__
                    ).parameters.keys()
                )[1:]
                params = {
                    attr: getattr(predictor.estimator.estimator.estimator, attr)
                    for attr in param_names
                    if attr != "kwargs"
                }
                if "_other_params" in predictor.estimator.estimator.estimator.__dict__:
                    params.update(predictor.estimator.estimator.estimator._other_params)

            else:
                raise ValueError(
                    f"estimator_subclass_level can't be higher than 2, got {estimator_subclass_level}"
                )

            params = add_params_from_search_range(
                params=params, trial=trial, parameter_search_range=search_ranges
            )
            for param, value in self.default_params.items():
                params[param] = value

            predictor = copy.deepcopy(pipeline_factory.predictor)
            for param in params:
                if estimator_subclass_level == 1:
                    setattr(predictor.estimator.estimator, param, params[param])
                elif estimator_subclass_level == 2:
                    setattr(
                        predictor.estimator.estimator.estimator, param, params[param]
                    )
                elif estimator_subclass_level > 2:
                    raise ValueError(
                        f"estimator_subclass_level can't be higher than 2, got {estimator_subclass_level}"
                    )
                else:
                    setattr(predictor.estimator, param, params[param])

            pipeline = pipeline_factory.create(predictor=predictor)
            create_rating_features = False
            for rating_generator in pipeline.rating_generators:
                create_rating_features = any(
                    feature not in df.columns
                    for feature in rating_generator.known_features_return
                )
                if create_rating_features:
                    break
            return pipeline.cross_validate_score(
                df=df,
                create_performance=False,
                create_rating_features=create_rating_features,
                cross_validator=cross_validator,
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
            lambda trial: objective(trial, df),
            n_trials=self.n_trials,
            callbacks=callbacks,
        )
        best_estimator_params = study.best_params
        other_predictor_params = list(
            inspect.signature(
                pipeline_factory.predictor.__class__.__init__
            ).parameters.keys()
        )[1:]

        if estimator_subclass_level > 0:
            if estimator_subclass_level == 1:
                best_estimator_params.update(
                    pipeline_factory.predictor.estimator.estimator._other_params
                )
            elif estimator_subclass_level == 2:
                best_estimator_params.update(
                    pipeline_factory.predictor.estimator.estimator.estimator._other_params
                )

        else:
            if "_other_params" in pipeline_factory.predictor.estimator.__dict__:
                best_estimator_params.update(
                    pipeline_factory.predictor.estimator._other_params
                )

        other_predictor_params = {
            attr: getattr(pipeline_factory.predictor, attr)
            for attr in other_predictor_params
            if attr not in ("estimator")
        }

        predictor_class = pipeline_factory.predictor.__class__
        if estimator_subclass_level == 1:

            potential_parent_names = list(
                inspect.signature(
                    pipeline_factory.predictor.estimator.__class__.__init__
                ).parameters.keys()
            )[1:]
            other_parent_params = {
                attr: getattr(pipeline_factory.predictor.estimator, attr)
                for attr in potential_parent_names
                if attr != "estimator"
            }

            estimator_class = pipeline_factory.predictor.estimator.estimator.__class__
            parent_estimator_class = pipeline_factory.predictor.estimator.__class__
            parent_estimator = parent_estimator_class(
                estimator=estimator_class(**best_estimator_params),
                **other_parent_params,
            )
            return predictor_class(estimator=parent_estimator, **other_predictor_params)
        elif estimator_subclass_level == 2:
            potential_parent_names = list(
                inspect.signature(
                    pipeline_factory.predictor.estimator.estimator.__class__.__init__
                ).parameters.keys()
            )[1:]
            other_parent_params = {
                attr: getattr(pipeline_factory.predictor.estimator.estimator, attr)
                for attr in potential_parent_names
                if attr != "estimator"
            }
            estimator_class = (
                pipeline_factory.predictor.estimator.estimator.estimator.__class__
            )
            parent_estimator_class = (
                pipeline_factory.predictor.estimator.estimator.__class__
            )
            parent_estimator = parent_estimator_class(
                estimator=estimator_class(**best_estimator_params),
                **other_parent_params,
            )

            parent_parent_estimator_class = (
                pipeline_factory.predictor.estimator.__class__
            )
            parent_parent_estimator = parent_parent_estimator_class(
                estimator=parent_estimator
            )
            return predictor_class(
                estimator=parent_parent_estimator, **other_predictor_params
            )

        elif estimator_subclass_level == 0:

            estimator_class = pipeline_factory.predictor.estimator.__class__
            return predictor_class(
                estimator=estimator_class(**best_estimator_params),
                **other_predictor_params,
            )

        else:
            raise ValueError(
                f"estimator_subclass_level can't be higher than 2, got {estimator_subclass_level}"
            )
