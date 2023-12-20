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
    custom_params: dict[str, Any] = field(default_factory=dict)

def add_params_from_search_range(trial: BaseTrial,parameter_search_range: list[ParameterSearchRange], params: dict) -> dict:
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
