from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Match, Literal, Union, Any

import pandas as pd
from optuna.trial import BaseTrial

from player_performance_ratings.transformers.base_transformer import BaseTransformer


class TransformerTuner(ABC):

    @abstractmethod
    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> list[BaseTransformer]:
        pass

class BaseTuner(ABC):

    @abstractmethod
    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> dict[str, float]:
        pass


@dataclass
class ParameterSearchRange:
    name: str
    type: Literal["uniform", "loguniform", "int", "categorical", "discrete_uniform"]
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[list[Any]] = None
    custom_params: dict[str, Any] = field(default_factory=dict)

def add_params_from_search_range(params: dict, trial: BaseTrial,
                                 parameter_search_range: list[ParameterSearchRange]) -> dict:
    for config in parameter_search_range:
        if config.type == "uniform":
            params[config.name] = trial.suggest_uniform(config.name, low=config.low, high=config.high)
        elif config.type == "loguniform":
            params[config.name] = trial.suggest_loguniform(config.name, low=config.low, high=config.high)
        elif config.type == "int":
            params[config.name] = trial.suggest_int(config.name, low=config.low, high=config.high)
        elif config.type == "categorical":
            params[config.name] = trial.suggest_categorical(config.name, config.choices)
        elif config.type == 'discrete_uniform:':
            params[config.name] = trial.suggest_discrete_uniform(config.name, low=config.low, high=config.high)

    return params


