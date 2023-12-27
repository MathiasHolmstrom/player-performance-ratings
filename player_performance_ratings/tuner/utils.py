import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, Any, Tuple

from optuna.trial import BaseTrial
from sklearn.preprocessing import StandardScaler

from player_performance_ratings.transformation.pre_transformers import SymmetricDistributionTransformer, \
    NetOverPredictedTransformer, SkLearnTransformerWrapper, MinMaxTransformer, ColumnWeight, ColumnsWeighter

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformation.base_transformer import BaseTransformer


@dataclass
class ParameterSearchRange:
    name: str
    type: Literal["uniform", "loguniform", "int", "categorical", "discrete_uniform"]
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[list[Any]] = None
    custom_params: dict[str, Any] = field(default_factory=dict)


def add_params_from_search_range(trial: BaseTrial, parameter_search_range: list[ParameterSearchRange],
                                 params: dict) -> dict:
    for config in parameter_search_range:
        if config.type == "uniform":
            params[config.name] = trial.suggest_uniform(config.name, low=config.low, high=config.high)
        elif config.type == "loguniform":
            params[config.name] = trial.suggest_loguniform(config.name, low=config.low, high=config.high)
        elif config.type == "int":
            params[config.name] = trial.suggest_int(config.name, low=config.low, high=config.high)
        elif config.type == "categorical":
            params[config.name] = trial.suggest_categorical(config.name, config.choices)
        else:
            logging.warning(f"Unknown type {config.type} for parameter {config.name}")

    return params


def create_pre_rating_search_range_for_auto(feature_names: Union[list[str], list[list[str]]],
                                            column_names: list[ColumnNames],
                                            lower_is_better_features: Optional[list[str]] = None) -> list[
    Tuple[BaseTransformer, list[ParameterSearchRange]]]:
    lower_is_better_features = lower_is_better_features or []

    pre_transformer_search_ranges = []

    if isinstance(feature_names[0], str):
        feature_names = [feature_names for _ in range(len(column_names))]

    for idx, column_name in enumerate(column_names):
        if column_name.position is None:
            granularity = []
        else:
            granularity = [column_name.position]

        if idx == 0 or column_names[idx].position != column_names[idx - 1].position:
            distribution_transformer = SymmetricDistributionTransformer(features=feature_names[idx],
                                                                        granularity=granularity)
            pre_transformer_search_ranges.append((distribution_transformer, []))

        if column_name.position is not None:
            position_predicted_transformer = NetOverPredictedTransformer(features=feature_names[idx],
                                                                         granularity=[column_names[idx].position])
            pre_transformer_search_ranges.append((position_predicted_transformer, []))

        pre_transformer_search_ranges.append(
            (SkLearnTransformerWrapper(transformer=StandardScaler(), features=feature_names[idx]), []))
        pre_transformer_search_ranges.append(
            (MinMaxTransformer(features=feature_names[idx]), []))

        column_weights = []
        column_weighter_search_range = []
        for feature in feature_names[idx]:
            if feature in lower_is_better_features:
                lower_is_better = True
            else:
                lower_is_better = False
            column_weight = ColumnWeight(
                name=feature,
                weight=1 / len(feature_names[idx]),
                lower_is_better=lower_is_better
            )
            column_weights.append(column_weight)
            column_weighter_search_range.append(
                ParameterSearchRange(
                    name=feature,
                    type='uniform',
                    low=0,
                    high=1,
                )
            )

        pre_transformer_search_ranges.append((ColumnsWeighter(column_weights=None,
                                                              weighted_column_name=column_name.performance),
                                              column_weighter_search_range))

    return pre_transformer_search_ranges


def get_default_team_rating_search_range() -> list[ParameterSearchRange]:
    return [
        ParameterSearchRange(
            name='confidence_weight',
            type='uniform',
            low=0.7,
            high=0.95
        ),
        ParameterSearchRange(
            name='confidence_days_ago_multiplier',
            type='uniform',
            low=0.02,
            high=.12,
        ),
        ParameterSearchRange(
            name='confidence_max_days',
            type='uniform',
            low=40,
            high=150,
        ),
        ParameterSearchRange(
            name='confidence_max_sum',
            type='uniform',
            low=60,
            high=300,
        ),
        ParameterSearchRange(
            name='confidence_value_denom',
            type='uniform',
            low=50,
            high=350
        ),
        ParameterSearchRange(
            name='rating_change_multiplier',
            type='uniform',
            low=30,
            high=100
        ),
        ParameterSearchRange(
            name='min_rating_change_multiplier_ratio',
            type='uniform',
            low=0.02,
            high=0.2,
        )
    ]

