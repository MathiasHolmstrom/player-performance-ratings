from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.ratings.enums import RatingColumnNames


class Operator(Enum):
    EQUALS = '=='
    NOT_EQUALS = '!='
    GREATER_THAN = '>'
    LESS_THAN = '<'
    GREATER_THAN_OR_EQUALS = '>='
    LESS_THAN_OR_EQUALS = '<='
    IN = 'in'
    NOT_IN = 'not in'


@dataclass
class Filter:
    column_name: str
    value: Union[Any, list[Any]]
    operator: Operator


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    for filter in filters:
        if filter.operator == Operator.EQUALS:
            df = df[df[filter.column_name] == filter.value]
        elif filter.operator == Operator.NOT_EQUALS:
            df = df[df[filter.column_name] != filter.value]
        elif filter.operator == Operator.GREATER_THAN:
            df = df[df[filter.column_name] > filter.value]
        elif filter.operator == Operator.LESS_THAN:
            df = df[df[filter.column_name] < filter.value]
        elif filter.operator == Operator.GREATER_THAN_OR_EQUALS:
            df = df[df[filter.column_name] >= filter.value]
        elif filter.operator == Operator.LESS_THAN_OR_EQUALS:
            df = df[df[filter.column_name] <= filter.value]
        elif filter.operator == Operator.IN:
            df = df[df[filter.column_name].isin(filter.value)]
        elif filter.operator == Operator.NOT_IN:
            df = df[~df[filter.column_name].isin(filter.value)]

    return df


class BaseScorer(ABC):

    def __init__(self, target: str, pred_column: str, filters: Optional[list[Filter]] = None,
                 granularity: Optional[list[str]] = None):
        self.target = target
        self.pred_column = pred_column
        self.filters = filters or []
        self.granularity = granularity

    @abstractmethod
    def score(self, df: pd.DataFrame) -> float:
        pass


class SklearnScorer(BaseScorer):

    def __init__(self,
                 pred_column: str,
                 scorer_function: Callable,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 granularity: Optional[list[str]] = None,
                 filters: Optional[list[Filter]] = None
                 ):
        self.pred_column_name = pred_column
        self.scorer_function = scorer_function
        super().__init__(target=target, pred_column=pred_column, granularity=granularity, filters=filters)

    def score(self, df: pd.DataFrame) -> float:
        df = df.copy()
        df = apply_filters(df, self.filters)
        if self.granularity:
            grouped = df.groupby(self.granularity)[self.pred_column_name, self.target].mean().reset_index()
        else:
            grouped = df
        if isinstance(df[self.pred_column_name].iloc[0], list):
            return self.scorer_function(grouped[self.target], np.asarray(grouped[self.pred_column_name]).tolist())
        return self.scorer_function(grouped[self.target], grouped[self.pred_column_name])


class OrdinalLossScorer(BaseScorer):

    def __init__(self,
                 pred_column: str,
                 target_range: list[int],
                 target: Optional[str] = PredictColumnNames.TARGET,
                 granularity: Optional[list[str]] = None,
                 filters: Optional[list[Filter]] = None
                 ):

        self.pred_column_name = pred_column
        self.target_range = target_range
        self.granularity = granularity
        super().__init__(target=target, pred_column=pred_column, filters=filters, granularity=granularity)

    def score(self, df: pd.DataFrame) -> float:

        df = df.copy()

        probs = df[self.pred_column_name]
        last_column_name = 'prob_under_0.5'
        df[last_column_name] = probs.apply(lambda x: x[0])

        df = apply_filters(df, self.filters)

        class_index = 0

        sum_lr = 0

        for class_ in self.target_range:
            class_index += 1
            p_c = 'prob_under_' + str(class_ + 0.5)
            df[p_c] = probs.apply(lambda x: x[class_index]) + df[last_column_name]

            count_exact = len(df[df['__target'] == class_])
            weight_class = count_exact / len(df)

            if self.granularity:
                grouped = df.groupby(self.granularity + ['__target'])[p_c].mean().reset_index()
            else:
                grouped = df

            grouped['min'] = 0.0001
            grouped['max'] = 0.9999
            grouped[p_c] = np.minimum(grouped['max'], grouped[p_c])
            grouped[p_c] = np.maximum(grouped['min'], grouped[p_c])
            grouped['log_loss'] = 0

            grouped.loc[grouped['__target'] <= class_, 'log_loss'] = np.log(grouped[p_c])
            grouped.loc[grouped['__target'] > class_, 'log_loss'] = np.log(1 - grouped[p_c])
            sum_lr -= grouped['log_loss'].mean() * weight_class

            last_column_name = p_c

        return sum_lr
