import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, Any

from optuna.trial import BaseTrial


@dataclass
class ParameterSearchRange:
    name: str
    type: Literal["uniform", "loguniform", "int", "categorical", "discrete_uniform"]
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[list[Any]] = None
    lower_is_better: bool = False
    custom_params: Optional[dict[str, Any]] = (None,)

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


def add_params_from_search_range(
    trial: BaseTrial, parameter_search_range: list[ParameterSearchRange], params: dict
) -> dict:
    for config in parameter_search_range:
        if config.type == "uniform":
            params[config.name] = trial.suggest_uniform(
                config.name, low=config.low, high=config.high
            )
        elif config.type == "loguniform":
            params[config.name] = trial.suggest_loguniform(
                config.name, low=config.low, high=config.high
            )
        elif config.type == "int":
            params[config.name] = trial.suggest_int(
                config.name, low=config.low, high=config.high
            )
        elif config.type == "categorical":
            params[config.name] = trial.suggest_categorical(config.name, config.choices)
        else:
            logging.warning(f"Unknown type {config.type} for parameter {config.name}")

    return params


def get_default_lgbm_classifier_search_range() -> list[ParameterSearchRange]:
    return [
        ParameterSearchRange(
            name="learning_rate",
            type="uniform",
            low=0.02,
            high=0.1,
        ),
        ParameterSearchRange(
            name="n_estimators",
            type="int",
            low=40,
            high=800,
        ),
        ParameterSearchRange(
            name="num_leaves",
            type="int",
            low=10,
            high=100,
        ),
        ParameterSearchRange(
            name="max_depth",
            type="int",
            low=2,
            high=10,
        ),
        ParameterSearchRange(
            name="min_child_samples",
            type="int",
            low=2,
            high=200,
        ),
        ParameterSearchRange(
            name="reg_alpha",
            type="uniform",
            low=0,
            high=5,
        ),
    ]


def get_default_lgbm_regressor_search_range() -> list[ParameterSearchRange]:

    return [
        ParameterSearchRange(
            name="learning_rate",
            type="uniform",
            low=0.02,
            high=0.1,
        ),
        ParameterSearchRange(
            name="n_estimators",
            type="int",
            low=50,
            high=1000,
        ),
        ParameterSearchRange(
            name="num_leaves",
            type="int",
            low=10,
            high=100,
        ),
        ParameterSearchRange(
            name="max_depth",
            type="int",
            low=2,
            high=14,
        ),
        ParameterSearchRange(
            name="min_child_samples",
            type="int",
            low=2,
            high=200,
        ),
        ParameterSearchRange(
            name="reg_alpha",
            type="uniform",
            low=0,
            high=5,
        ),
    ]


def get_default_team_rating_search_range() -> list[ParameterSearchRange]:
    return [
        ParameterSearchRange(
            name="confidence_weight", type="uniform", low=0.7, high=0.95
        ),
        ParameterSearchRange(
            name="confidence_days_ago_multiplier",
            type="uniform",
            low=0.02,
            high=0.12,
        ),
        ParameterSearchRange(
            name="confidence_max_days",
            type="uniform",
            low=60,
            high=220,
        ),
        ParameterSearchRange(
            name="confidence_max_sum",
            type="uniform",
            low=60,
            high=300,
        ),
        ParameterSearchRange(
            name="confidence_value_denom", type="uniform", low=25, high=120
        ),
        ParameterSearchRange(
            name="rating_change_multiplier", type="uniform", low=10, high=80
        ),
        ParameterSearchRange(
            name="min_rating_change_multiplier_ratio",
            type="uniform",
            low=0.02,
            high=0.2,
        ),
        ParameterSearchRange(
            name="team_id_change_confidence_sum_decrease",
            type="uniform",
            low=0,
            high=15,
        ),
    ]
