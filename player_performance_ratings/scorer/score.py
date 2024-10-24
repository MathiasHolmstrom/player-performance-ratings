import math
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable, Union, Any
from narwhals.typing import FrameT
import narwhals as nw
import narwhals
import numpy as np
import pandas as pd
from polars.polars import col

from player_performance_ratings.consts import PredictColumnNames

import polars as pl


class Operator(Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUALS = ">="
    LESS_THAN_OR_EQUALS = "<="
    IN = "in"
    NOT_IN = "not in"


@dataclass
class Filter:
    column_name: str
    value: Union[Any, list[Any]]
    operator: Operator


def _apply_filters_pandas(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
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


def _apply_filters_polars(df: pl.DataFrame, filters: list[Filter]) -> pl.DataFrame:
    for filter in filters:
        if filter.operator == Operator.EQUALS:
            df = df.filter(col(filter.column_name) == filter.value)
        elif filter.operator == Operator.NOT_EQUALS:
            df = df.filter(col(filter.column_name) != filter.value)
        elif filter.operator == Operator.GREATER_THAN:
            df = df.filter(col(filter.column_name) > filter.value)
        elif filter.operator == Operator.LESS_THAN:
            df = df.filter(col(filter.column_name) < filter.value)
        elif filter.operator == Operator.GREATER_THAN_OR_EQUALS:
            df = df.filter(col(filter.column_name) >= filter.value)
        elif filter.operator == Operator.LESS_THAN_OR_EQUALS:
            df = df.filter(col(filter.column_name) <= filter.value)
        elif filter.operator == Operator.IN:
            df = df.filter(col(filter.column_name).isin(filter.value))
        elif filter.operator == Operator.NOT_IN:
            df = df.filter(~col(filter.column_name).isin(filter.value))

    return df

@narwhals.narwhalify()
def apply_filters(df: FrameT, filters: list[Filter]) -> Union[pd.DataFrame, pl.DataFrame]:
    for filter in filters:
        if filter.operator == Operator.EQUALS:
            df = df.filter(nw.col(filter.column_name) == filter.value)
        elif filter.operator == Operator.NOT_EQUALS:
            df = df.filter(nw.col(filter.column_name) != filter.value)
        elif filter.operator == Operator.GREATER_THAN:
            df = df.filter(nw.col(filter.column_name) > filter.value)
        elif filter.operator == Operator.LESS_THAN:
            df = df.filter(nw.col(filter.column_name) < filter.value)
        elif filter.operator == Operator.GREATER_THAN_OR_EQUALS:
            df = df.filter(nw.col(filter.column_name) >= filter.value)
        elif filter.operator == Operator.LESS_THAN_OR_EQUALS:
            df = df.filter(nw.col(filter.column_name) <= filter.value)
        elif filter.operator == Operator.IN:
            df = df.filter(nw.col(filter.column_name).is_in(filter.value))
        elif filter.operator == Operator.NOT_IN:
            df = df.filter(~nw.col(filter.column_name).is_in(filter.value))

    return df.to_native()

class BaseScorer(ABC):

    def __init__(
            self,
            target: str,
            pred_column: str,
            validation_column: Optional[str],
            filters: Optional[list[Filter]] = None,
            granularity: Optional[list[str]] = None,
    ):
        self.target = target
        self.pred_column = pred_column
        self.validation_column = validation_column
        self.filters = []
        if validation_column:
            self.filters.append(Filter(column_name=self.validation_column, value=1, operator=Operator.EQUALS))
        self.granularity = granularity

    @abstractmethod
    def score(self, df: Union[pl.DataFrame, pd.DataFrame]) -> float:
        pass


class SklearnScorer(BaseScorer):

    def __init__(
            self,
            pred_column: str,
            scorer_function: Callable,
            target: Optional[str] = PredictColumnNames.TARGET,
            validation_column: Optional[str] = None,
            granularity: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
    ):
        """
        :param pred_column: The column name of the predictions
        :param scorer_function: SKlearn scorer function, e.g. los_loss
        :param target: The column name of the target
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param granularity: The columns to group by before calculating the score
        :param filters: The filters to apply before calculating
        """

        self.pred_column_name = pred_column
        self.scorer_function = scorer_function
        super().__init__(
            target=target,
            pred_column=pred_column,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
        )

    @narwhals.narwhalify
    def score(self, df: Union[pd.DataFrame, pl.DataFrame]) -> float:

        df = apply_filters(df=df, filters=self.filters)
        if self.granularity:
            grouped = (
                df.groupby(self.granularity)
                .agg([
                    pl.col(self.pred_column_name).mean().alias(self.pred_column_name),
                    pl.col(self.target).mean().alias(self.target)
                ])
            )
        else:
            grouped = df

        if isinstance(grouped[self.pred_column_name].to_list()[0], list):
            return self.scorer_function(
                grouped[self.target],
                [item for item in grouped[self.pred_column_name].to_list()],
            )

        return self.scorer_function(
            grouped[self.target].to_list(),
            grouped[self.pred_column_name].to_list()
        )


class ProbabilisticMeanBias(BaseScorer):

    def __init__(
            self,
            pred_column: str,
            class_column_name: str = "classes",
            target: Optional[str] = PredictColumnNames.TARGET,
            validation_column: Optional[str] = None,
            granularity: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
    ):

        self.pred_column_name = pred_column
        self.class_column_name = class_column_name
        super().__init__(
            target=target,
            pred_column=pred_column,
            filters=filters,
            granularity=granularity,
            validation_column=validation_column,
        )

    def score(self, df: pd.DataFrame) -> float:
        df = df.copy()
        df = apply_filters(df, self.filters)
        df.reset_index(drop=True, inplace=True)

        distinct_classes_variations = df.drop_duplicates(subset=[self.class_column_name])[
            self.class_column_name
        ].tolist()

        sum_lrs = [0 for _ in range(len(distinct_classes_variations))]
        sum_lr = 0
        for variation_idx, distinct_class_variation in enumerate(
                distinct_classes_variations
        ):

            if not isinstance(distinct_class_variation, list):
                if math.isnan(distinct_class_variation):
                    continue

            rows_target_group = df[
                df[self.class_column_name].apply(lambda x: x == distinct_class_variation)
            ]
            probs = rows_target_group[self.pred_column_name]
            last_column_name = f"prob_under_{distinct_class_variation[0] - 0.5}"
            rows_target_group[last_column_name] = probs.apply(lambda x: x[0])

            for idx, class_ in enumerate(distinct_class_variation[1:]):

                prob_under = "prob_under_" + str(class_ + 0.5)
                rows_target_group[prob_under] = (
                        probs.apply(lambda x: x[idx + 1])
                        + rows_target_group[last_column_name]
                )

                count_exact = len(
                    rows_target_group[rows_target_group["__target"] == class_]
                )
                weight_class = count_exact / len(rows_target_group)

                if self.granularity:
                    grouped = (
                        rows_target_group.groupby(self.granularity + ["__target"])[prob_under]
                        .mean()
                        .reset_index()
                    )
                else:
                    grouped = rows_target_group

                grouped["min"] = 0.0001
                grouped["max"] = 0.9999
                grouped[prob_under] = np.minimum(grouped["max"], grouped[prob_under])
                grouped[prob_under] = np.maximum(grouped["min"], grouped[prob_under])

                grouped.loc[grouped['__target'] <= class_, "__went_under"] = 1
                grouped.loc[grouped['__target'] > class_, "__went_under"] = 0

                under_prob_mean = grouped[prob_under].mean()
                under_actual_mean = grouped['__went_under'].mean()

                overbias = under_prob_mean - under_actual_mean
                sum_lrs[variation_idx] += overbias * weight_class

                last_column_name = prob_under
            sum_lr += sum_lrs[variation_idx] * len(rows_target_group) / len(df)
        return sum_lr


class OrdinalLossScorerPolars(BaseScorer):
    """TODO: Support different types of probability-class constructions"""

    def __init__(
            self,
            pred_column: str,
            target: Optional[str] = PredictColumnNames.TARGET,
            validation_column: Optional[str] = None,
            granularity: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
    ):
        self.pred_column_name = pred_column
        self.granularity = granularity
        super().__init__(
            target=target,
            pred_column=pred_column,
            filters=filters,
            granularity=granularity,
            validation_column=validation_column,
        )

    def score(self, df: pl.DataFrame) -> float:
        df = apply_filters(df, self.filters)
        field_names = [int(field) for field in df[self.pred_column].struct.fields]
        min_field = min(field_names)
        df = df.with_columns(
            [pl.col(self.pred_column).struct.field(str(field)).alias(f"prob_{field}") for field in field_names])
        prob_col_under = 'sum_prob_under'
        df = df.with_columns(pl.col(f'prob_{min_field}').alias(prob_col_under))
        sum_lr = 0
        count_higher_rows = len(df.filter(pl.col(self.target) > min_field))
        for class_ in field_names[1:]:

            count_exact = len(df.filter(pl.col(self.target) == class_-1))
            weight_class = count_exact / count_higher_rows

            df = df.with_columns([
                pl.lit(0.0001).alias("min"),
                pl.lit(0.9999).alias("max")
            ])

            df = df.with_columns(
                pl.max_horizontal(prob_col_under, "min").alias(prob_col_under),
                          )
            df = df.with_columns(
                pl.min_horizontal(prob_col_under, "max").alias(prob_col_under),
            )

            df = df.with_columns([
                pl.when(pl.col("__target") < class_)
                .then(pl.col(prob_col_under).log())
                .otherwise((1 - pl.col(prob_col_under)).log())
                .alias("log_loss")
            ])

            log_loss = df["log_loss"].mean()
            sum_lr -= log_loss * weight_class

            df = df.with_columns(
                (pl.col(f"prob_{class_}") + pl.col(prob_col_under)).alias(prob_col_under)
            )

        return sum_lr

class OrdinalLossScorer(BaseScorer):

    def __init__(
            self,
            pred_column: str,
            class_column_name: str = "classes",
            target: Optional[str] = PredictColumnNames.TARGET,
            validation_column: Optional[str] = None,
            granularity: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
    ):
        """
        :param pred_column: The column name of the predictions
        :param class_column_name: The column name that contains the integer values of the classes.
        :param target: The column name of the target
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param granularity: The columns to group by before calculating the score
        :param filters: The filters to apply before calculating
        """

        self.pred_column_name = pred_column
        self.class_column_name = class_column_name
        self.granularity = granularity
        super().__init__(
            target=target,
            pred_column=pred_column,
            filters=filters,
            granularity=granularity,
            validation_column=validation_column,
        )

    def score(self, df: pd.DataFrame) -> float:

        df = df.copy()
        df = apply_filters(df, self.filters)
        df.reset_index(drop=True, inplace=True)

        distinct_classes_variations = df.drop_duplicates(subset=[self.class_column_name])[
            self.class_column_name
        ].tolist()

        sum_lrs = [0 for _ in range(len(distinct_classes_variations))]
        sum_lr = 0
        for variation_idx, distinct_class_variation in enumerate(
                distinct_classes_variations
        ):

            if not isinstance(distinct_class_variation, list):
                if math.isnan(distinct_class_variation):
                    continue

            rows_target_group = df[
                df[self.class_column_name].apply(lambda x: x == distinct_class_variation)
            ]
            probs = rows_target_group[self.pred_column_name]
            last_column_name = f"prob_under_{distinct_class_variation[0] - 0.5}"
            rows_target_group[last_column_name] = probs.apply(lambda x: x[0])

            for idx, class_ in enumerate(distinct_class_variation[1:]):

                p_c = "prob_under_" + str(class_ + 0.5)
                rows_target_group[p_c] = (
                        probs.apply(lambda x: x[idx + 1])
                        + rows_target_group[last_column_name]
                )

                count_exact = len(
                    rows_target_group[rows_target_group["__target"] == class_]
                )
                weight_class = count_exact / len(rows_target_group)

                if self.granularity:
                    grouped = (
                        rows_target_group.groupby(self.granularity + ["__target"])[p_c]
                        .mean()
                        .reset_index()
                    )
                else:
                    grouped = rows_target_group

                grouped["min"] = 0.0001
                grouped["max"] = 0.9999
                grouped[p_c] = np.minimum(grouped["max"], grouped[p_c])
                grouped[p_c] = np.maximum(grouped["min"], grouped[p_c])
                grouped["log_loss"] = 0
                grouped.loc[grouped["__target"] <= class_, "log_loss"] = np.log(
                    grouped[p_c]
                )
                grouped.loc[grouped["__target"] > class_, "log_loss"] = np.log(
                    1 - grouped[p_c]
                )
                log_loss = grouped["log_loss"].mean()
                sum_lrs[variation_idx] -= log_loss * weight_class

                last_column_name = p_c
            sum_lr += sum_lrs[variation_idx] * len(rows_target_group) / len(df)
        return sum_lr
