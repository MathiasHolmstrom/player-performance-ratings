import math
import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable, Union, Any
from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw
import narwhals
import numpy as np
import pandas as pd
from polars.polars import col

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

        if df[filter.column_name].dtype in (pl.Datetime, pl.Date) and isinstance(
                filter.value, str
        ):
            filter_value = datetime.datetime.fromisoformat(filter.value).replace(
                tzinfo=datetime.timezone.utc
            )
        else:
            filter_value = filter.value

        if filter.operator == Operator.EQUALS:
            df = df.filter(pl.col(filter.column_name) == pl.lit(filter_value))
        elif filter.operator == Operator.NOT_EQUALS:
            df = df.filter(pl.col(filter.column_name) != pl.lit(filter_value))
        elif filter.operator == Operator.GREATER_THAN:
            df = df.filter(pl.col(filter.column_name) > pl.lit(filter_value))
        elif filter.operator == Operator.LESS_THAN:
            df = df.filter(pl.col(filter.column_name) < pl.lit(filter_value))
        elif filter.operator == Operator.GREATER_THAN_OR_EQUALS:
            df = df.filter(pl.col(filter.column_name) >= pl.lit(filter_value))
        elif filter.operator == Operator.LESS_THAN_OR_EQUALS:
            df = df.filter(pl.col(filter.column_name) <= pl.lit(filter_value))
        elif filter.operator == Operator.IN:
            df = df.filter(pl.col(filter.column_name).is_in(filter_value))
        elif filter.operator == Operator.NOT_IN:
            df = df.filter(~pl.col(filter.column_name).is_in(filter_value))

    return df


def apply_filters(df: FrameT, filters: list[Filter]) -> FrameT:
    for filter in filters:

        if df[filter.column_name].dtype in (nw.Datetime, nw.Date) and isinstance(
                filter.value, str
        ):
            filter_value = datetime.datetime.fromisoformat(filter.value).replace(
                tzinfo=datetime.timezone.utc
            )
        else:
            filter_value = filter.value

        if filter.operator == Operator.EQUALS:
            df = df.filter(nw.col(filter.column_name) == nw.lit(filter_value))
        elif filter.operator == Operator.NOT_EQUALS:
            df = df.filter(nw.col(filter.column_name) != nw.lit(filter_value))
        elif filter.operator == Operator.GREATER_THAN:
            df = df.filter(nw.col(filter.column_name) > nw.lit(filter_value))
        elif filter.operator == Operator.LESS_THAN:
            df = df.filter(nw.col(filter.column_name) < nw.lit(filter_value))
        elif filter.operator == Operator.GREATER_THAN_OR_EQUALS:
            df = df.filter(nw.col(filter.column_name) >= nw.lit(filter_value))
        elif filter.operator == Operator.LESS_THAN_OR_EQUALS:
            df = df.filter(nw.col(filter.column_name) <= nw.lit(filter_value))
        elif filter.operator == Operator.IN:
            df = df.filter(nw.col(filter.column_name).is_in(filter_value))
        elif filter.operator == Operator.NOT_IN:
            df = df.filter(~nw.col(filter.column_name).is_in(filter_value))

    return df


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
        self.filters = filters or []
        if validation_column:
            self.filters.append(
                Filter(
                    column_name=self.validation_column,
                    value=1,
                    operator=Operator.EQUALS,
                )
            )
        self.granularity = granularity

    @abstractmethod
    def score(self, df: FrameT) -> float:
        pass


class MeanBiasScorer(BaseScorer):
    def __init__(
            self,
            pred_column: str,
            target: str,
            validation_column: Optional[str] = None,
            granularity: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
    ):
        """
        :param pred_column: The column name of the predictions
        :param target: The column name of the target
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param granularity: The columns to group by before calculating the score
        :param filters: The filters to apply before calculating
        """

        self.pred_column_name = pred_column
        super().__init__(
            target=target,
            pred_column=pred_column,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
        )

    @narwhals.narwhalify
    def score(self, df: FrameT) -> float:
        df = apply_filters(df, self.filters)
        if self.granularity:
            grouped = df.group_by(self.granularity).agg(
                [
                    nw.col(self.pred_column_name).sum().alias(self.pred_column_name),
                    nw.col(self.target).sum().alias(self.target),
                ]
            )
            return (grouped[self.pred_column] - grouped[self.target]).mean()
        return (df[self.pred_column] - df[self.target]).mean()


class SklearnScorer(BaseScorer):

    def __init__(
            self,
            scorer_function: Callable,
            pred_column: str,
            target: str,
            drop_rows_where_target_is_nan: bool = True,
            validation_column: Optional[str] = None,
            granularity: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
            labels: list[int] | None = None

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
        self.drop_rows_where_target_is_nan = drop_rows_where_target_is_nan
        self.pred_column_name = pred_column
        self.scorer_function = scorer_function
        self.labels = labels

        super().__init__(
            target=target,
            pred_column=pred_column,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
        )

    @narwhals.narwhalify
    def score(self, df: FrameT) -> float:

        df = apply_filters(df=df, filters=self.filters)
        if self.drop_rows_where_target_is_nan:
            df = df.filter(~nw.col(self.target).is_nan())
        if isinstance(df[self.pred_column_name].to_list()[0], dict):
            assert (
                not self.granularity
            ), "Granularity is not supported for dict predictions"
            df = df.to_polars()
            df = df.with_columns(
                pl.col(self.pred_column)
                .struct.unnest()
                .pipe(pl.concat_list)
                .alias(self.pred_column)
            )

        if self.granularity:
            grouped = df.group_by(self.granularity).agg(
                [
                    nw.col(self.pred_column_name).sum().alias(self.pred_column_name),
                    nw.col(self.target).sum().alias(self.target),
                ]
            )
        else:
            grouped = df
        kwargs = {}
        if self.labels:
            kwargs = {
                'labels': self.labels
            }
        if isinstance(grouped[self.pred_column_name].to_list()[0], list):
            return self.scorer_function(
                grouped[self.target],
                [item for item in grouped[self.pred_column_name].to_list()],
                **kwargs
            )

        return self.scorer_function(
            grouped[self.target].to_list(), grouped[self.pred_column_name].to_list(), **kwargs
        )


class ProbabilisticMeanBias(BaseScorer):

    def __init__(
            self,
            pred_column: str,
            target: str,
            class_column_name: str = "classes",
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

        distinct_classes_variations = df.drop_duplicates(
            subset=[self.class_column_name]
        )[self.class_column_name].tolist()

        sum_lrs = [0 for _ in range(len(distinct_classes_variations))]
        sum_lr = 0
        for variation_idx, distinct_class_variation in enumerate(
                distinct_classes_variations
        ):

            if not isinstance(distinct_class_variation, list):
                if math.isnan(distinct_class_variation):
                    continue

            rows_target_group = df[
                df[self.class_column_name].apply(
                    lambda x: x == distinct_class_variation
                )
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
                        rows_target_group.groupby(self.granularity + ["__target"])[
                            prob_under
                        ]
                        .mean()
                        .reset_index()
                    )
                else:
                    grouped = rows_target_group

                grouped["min"] = 0.0001
                grouped["max"] = 0.9999
                grouped[prob_under] = np.minimum(grouped["max"], grouped[prob_under])
                grouped[prob_under] = np.maximum(grouped["min"], grouped[prob_under])

                grouped.loc[grouped["__target"] <= class_, "__went_under"] = 1
                grouped.loc[grouped["__target"] > class_, "__went_under"] = 0

                under_prob_mean = grouped[prob_under].mean()
                under_actual_mean = grouped["__went_under"].mean()

                overbias = under_prob_mean - under_actual_mean
                sum_lrs[variation_idx] += overbias * weight_class

                last_column_name = prob_under
            sum_lr += sum_lrs[variation_idx] * len(rows_target_group) / len(df)
        return sum_lr


class OrdinalLossScorer(BaseScorer):
    def __init__(
            self,
            pred_column: str,
            target: str,
            validation_column: Optional[str] = None,
            granularity: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
            classes: list[int] | None = None,
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
        self.classes = classes

    @narwhals.narwhalify
    def score(self, df: FrameT) -> float:
        df = apply_filters(df, self.filters)
        df = df.to_polars()

        pred_dtype = df.schema[self.pred_column]

        if pred_dtype == pl.Struct:
            class_labels = [int(f) for f in df[self.pred_column].struct.fields]
            class_labels.sort()
            df = df.with_columns(
                [
                    pl.col(self.pred_column).struct.field(str(c)).alias(f"prob_{c}")
                    for c in class_labels
                ]
            )
        else:
            if pred_dtype != pl.List and pred_dtype != pl.Array:
                raise TypeError(
                    f"Unsupported pred_column dtype: {pred_dtype}. Expected Struct or List/Array."
                )
            if not self.classes:
                raise ValueError(
                    "OrdinalLossScorer: `classes` must be provided when pred_column is List/Array (sklearn-style probabilities)."
                )

            class_labels = [int(c) for c in self.classes]
            class_labels.sort()
            expected_len = len(class_labels)

            if pred_dtype == pl.Array:
                width = int(pred_dtype.shape[0])
                if width != expected_len:
                    raise ValueError(
                        f"OrdinalLossScorer: pred_column Array width ({width}) does not match len(classes) ({expected_len})."
                    )
                get_expr = lambda i: pl.col(self.pred_column).arr.get(i)
            else:
                max_len = df.select(pl.col(self.pred_column).list.len().max()).item()
                if max_len is not None and int(max_len) != expected_len:
                    raise ValueError(
                        f"OrdinalLossScorer: pred_column List length ({int(max_len)}) does not match len(classes) ({expected_len})."
                    )
                get_expr = lambda i: pl.col(self.pred_column).list.get(i)

            df = df.with_columns(
                [get_expr(i).alias(f"prob_{c}") for i, c in enumerate(class_labels)]
            )

        if len(class_labels) < 2:
            raise ValueError("OrdinalLossScorer: need at least 2 classes.")

        if class_labels != list(range(class_labels[0], class_labels[0] + len(class_labels))):
            raise ValueError(
                f"OrdinalLossScorer: classes must be consecutive integers. Got: {class_labels[:10]}..."
            )

        min_field = class_labels[0]
        prob_col_under = "sum_prob_under"
        df = df.with_columns(pl.col(f"prob_{min_field}").alias(prob_col_under))

        counts = df.group_by(self.target).len().rename({"len": "n"})
        total = counts.filter(pl.col(self.target) < class_labels[-1])["n"].sum()
        total = int(total) if total is not None else 0
        if total <= 0:
            return 0.0

        sum_lr = 0.0

        for class_ in class_labels[1:]:
            n_exact = counts.filter(pl.col(self.target) == class_ - 1)["n"].sum()
            n_exact = int(n_exact) if n_exact is not None else 0
            weight_class = n_exact / total
            if weight_class == 0.0:
                df = df.with_columns(
                    (pl.col(f"prob_{class_}") + pl.col(prob_col_under))
                    .clip(0.0, 1.0)
                    .alias(prob_col_under)
                )
                continue

            df = df.with_columns(
                pl.col(prob_col_under).clip(0.0001, 0.9999).alias(prob_col_under)
            )

            log_loss = df.select(
                pl.when(pl.col(self.target) < class_)
                .then(pl.col(prob_col_under).log())
                .otherwise((1 - pl.col(prob_col_under)).log())
                .mean()
            ).item()

            sum_lr -= float(log_loss) * float(weight_class)

            df = df.with_columns(
                (pl.col(f"prob_{class_}") + pl.col(prob_col_under))
                .clip(0.0, 1.0)
                .alias(prob_col_under)
            )

        return float(sum_lr)


class OrdinalLossScorerOld(BaseScorer):

    def __init__(
            self,
            pred_column: str,
            target: str,
            class_column_name: str = "classes",
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

        distinct_classes_variations = df.drop_duplicates(
            subset=[self.class_column_name]
        )[self.class_column_name].tolist()

        sum_lrs = [0 for _ in range(len(distinct_classes_variations))]
        sum_lr = 0
        for variation_idx, distinct_class_variation in enumerate(
                distinct_classes_variations
        ):

            if not isinstance(distinct_class_variation, list):
                if math.isnan(distinct_class_variation):
                    continue

            rows_target_group = df[
                df[self.class_column_name].apply(
                    lambda x: x == distinct_class_variation
                )
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
