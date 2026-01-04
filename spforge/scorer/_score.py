import datetime
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import narwhals
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
from narwhals.typing import IntoFrameT

_logger = logging.getLogger(__name__)


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
    value: Any | list[Any]
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

        if df[filter.column_name].dtype in (pl.Datetime, pl.Date) and isinstance(filter.value, str):
            filter_value = datetime.datetime.fromisoformat(filter.value).replace(
                tzinfo=datetime.UTC
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


@narwhals.narwhalify
def apply_filters(df: IntoFrameT, filters: list[Filter]) -> IntoFrameT:
    for filter in filters:

        if df[filter.column_name].dtype in (nw.Datetime, nw.Date) and isinstance(filter.value, str):
            parsed_dt = datetime.datetime.fromisoformat(filter.value)

            try:
                first_val = df[filter.column_name].to_list()[0] if len(df) > 0 else None
                if first_val is not None:
                    if isinstance(first_val, pd.Timestamp):
                        if first_val.tz is None:
                            filter_value = parsed_dt.replace(tzinfo=None)
                        else:
                            filter_value = parsed_dt.replace(tzinfo=datetime.UTC)
                    else:

                        dtype_str = str(df[filter.column_name].dtype)
                        if "UTC" in dtype_str or "timezone" in dtype_str.lower():
                            filter_value = parsed_dt.replace(tzinfo=datetime.UTC)
                        else:
                            filter_value = parsed_dt.replace(tzinfo=None)
                else:
                    filter_value = parsed_dt.replace(tzinfo=None)
            except:
                filter_value = parsed_dt.replace(tzinfo=None)
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
        validation_column: str | None,
        filters: list[Filter] | None = None,
        aggregation_level: list[str] | None = None,
        granularity: list[str] | None = None,
    ):
        """
        :param target: The column name of the target
        :param pred_column: The column name of the predictions
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param filters: The filters to apply before calculating
        :param aggregation_level: The columns to group by before calculating the score (e.g., group from game-player to game-team)
        :param granularity: The columns to calculate separate scores for each unique combination (e.g., different scores for each team)
        """
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
        self.aggregation_level = aggregation_level
        self.granularity = granularity

    def _apply_aggregation_level(self, df: IntoFrameT) -> IntoFrameT:
        """Apply aggregation_level grouping if set"""
        if self.aggregation_level:
            # Determine aggregation method based on column types
            # For numeric columns, use sum; for others, use first or mean
            agg_exprs = []
            for col in [self.pred_column, self.target]:
                # Try to determine if numeric
                try:
                    # Use sum for aggregation
                    agg_exprs.append(nw.col(col).sum().alias(col))
                except:
                    # Fallback to mean or first
                    agg_exprs.append(nw.col(col).mean().alias(col))

            df = df.group_by(self.aggregation_level).agg(agg_exprs)
        return df

    def _get_granularity_groups(self, df: IntoFrameT) -> list[tuple]:
        """Get list of granularity tuples from dataframe"""
        if not self.granularity:
            return []
        granularity_values = df.select(self.granularity).unique().to_dict(as_series=False)
        return list(zip(*[granularity_values[col] for col in self.granularity]))

    def _filter_to_granularity(self, df: IntoFrameT, gran_tuple: tuple) -> IntoFrameT:
        """Filter dataframe to specific granularity combination"""
        mask = None
        for i, col in enumerate(self.granularity):
            col_mask = nw.col(col) == gran_tuple[i]
            mask = col_mask if mask is None else (mask & col_mask)
        return df.filter(mask)

    @abstractmethod
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        """
        Calculate the score(s).

        :param df: The dataframe to score
        :return: If granularity is None, returns a single float score.
                 If granularity is set, returns a dict mapping granularity combinations (as tuples) to scores.
        """
        pass


class PWMSE(BaseScorer):
    def __init__(
        self,
        pred_column: str,
        target: str,
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
        labels: list[int] | None = None,
    ):
        self.pred_column_name = pred_column
        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
        )
        self.labels = labels

    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        df = apply_filters(df, self.filters)
        before = len(df)
        # Filter out null targets - ensure df is properly wrapped as Narwhals DataFrame
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)
        df = df.filter(~nw.col(self.target).is_null())
        after = len(df)
        _logger.info(
            "Dropped %d rows with NaN target (%d → %d)",
            before - after,
            before,
            after,
        )

        # Apply aggregation_level if set
        # For PWMSE, predictions are lists/arrays, which can't be aggregated with mean
        # Skip aggregation_level for PWMSE with list predictions (or implement custom aggregation)
        # For now, we'll skip aggregation if pred_column contains lists
        if self.aggregation_level:
            # Check if pred_column contains lists
            first_pred = df[self.pred_column].to_list()[0] if len(df) > 0 else None
            if isinstance(first_pred, list):
                # Can't aggregate lists easily - skip aggregation_level for PWMSE
                # In practice, you'd want custom list aggregation logic here
                pass  # Skip aggregation for list predictions
            else:
                # Numeric predictions can be aggregated
                df = df.group_by(self.aggregation_level).agg(
                    [
                        nw.col(self.pred_column).mean().alias(self.pred_column),
                        nw.col(self.target).mean().alias(self.target),
                    ]
                )

        # If granularity is set, calculate separate scores per group
        if self.granularity:
            results = {}
            granularity_values = df.select(self.granularity).unique().to_dict(as_series=False)
            granularity_tuples = list(zip(*[granularity_values[col] for col in self.granularity]))

            for gran_tuple in granularity_tuples:
                # Filter to this granularity combination
                mask = None
                for i, col in enumerate(self.granularity):
                    col_mask = nw.col(col) == gran_tuple[i]
                    mask = col_mask if mask is None else (mask & col_mask)
                gran_df = df.filter(mask)

                # Calculate score for this group
                labels = np.asarray(self.labels, dtype=np.float64)
                targets = gran_df[self.target].to_numpy().astype(np.float64)
                preds = np.asarray(gran_df[self.pred_column].to_list(), dtype=np.float64)
                diffs_sqd = (labels[None, :] - targets[:, None]) ** 2
                results[gran_tuple] = float((diffs_sqd * preds).sum(axis=1).mean())

            return results

        # Single score calculation
        labels = np.asarray(self.labels, dtype=np.float64)
        targets = df[self.target].to_numpy().astype(np.float64)
        preds = np.asarray(df[self.pred_column].to_list(), dtype=np.float64)
        diffs_sqd = (labels[None, :] - targets[:, None]) ** 2
        return float((diffs_sqd * preds).sum(axis=1).mean())


class MeanBiasScorer(BaseScorer):
    def __init__(
        self,
        pred_column: str,
        target: str,
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
    ):
        """
        :param pred_column: The column name of the predictions
        :param target: The column name of the target
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param aggregation_level: The columns to group by before calculating the score (e.g., group from game-player to game-team)
        :param granularity: The columns to calculate separate scores for each unique combination (e.g., different scores for each team)
        :param filters: The filters to apply before calculating
        """

        self.pred_column_name = pred_column
        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
        )

    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        df = apply_filters(df, self.filters)
        # Ensure df is a Narwhals DataFrame
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)

        # Apply aggregation_level if set
        if self.aggregation_level:
            df = df.group_by(self.aggregation_level).agg(
                [
                    nw.col(self.pred_column_name).sum().alias(self.pred_column_name),
                    nw.col(self.target).sum().alias(self.target),
                ]
            )
            # After group_by, ensure df is still a Narwhals DataFrame
            if not hasattr(df, "to_native"):
                df = nw.from_native(df)

        # If granularity is set, calculate separate scores per group
        if self.granularity:
            results = {}
            # Get unique granularity combinations - convert to native first to use select
            df_native = df.to_native()
            if isinstance(df_native, pd.DataFrame):
                # For pandas, use unique() and convert to list of tuples
                gran_df_unique = df_native[self.granularity].drop_duplicates()
                granularity_tuples = [tuple(row) for row in gran_df_unique.values]
            else:
                # For polars, use select and unique
                gran_df_unique = df_native.select(self.granularity).unique()
                granularity_tuples = [tuple(row) for row in gran_df_unique.iter_rows()]

            for gran_tuple in granularity_tuples:
                # Filter to this granularity combination using Narwhals
                mask = None
                for i, col in enumerate(self.granularity):
                    col_mask = nw.col(col) == nw.lit(gran_tuple[i])
                    mask = col_mask if mask is None else (mask & col_mask)
                gran_df = df.filter(mask)

                # Calculate score for this group
                score = float((gran_df[self.pred_column] - gran_df[self.target]).mean())
                results[gran_tuple] = score

            return results

        # Single score calculation
        mean_score = (df[self.pred_column] - df[self.target]).mean()
        # Handle NaN/None from empty dataframes
        if mean_score is None or (isinstance(mean_score, float) and pd.isna(mean_score)):
            return 0.0
        return float(mean_score)


class SklearnScorer(BaseScorer):

    def __init__(
        self,
        scorer_function: Callable,
        pred_column: str,
        target: str,
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
        params: dict[str, Any] = None,
    ):
        """
        :param pred_column: The column name of the predictions
        :param scorer_function: SKlearn scorer function, e.g. los_loss
        :param target: The column name of the target
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param aggregation_level: The columns to group by before calculating the score (e.g., group from game-player to game-team)
        :param granularity: The columns to calculate separate scores for each unique combination (e.g., different scores for each team)
        :param filters: The filters to apply before calculating
        """

        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
        )
        self.pred_column_name = pred_column
        self.scorer_function = scorer_function
        self.params = params or {}


    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:

        df = nw.from_native(apply_filters(df=df, filters=self.filters))

        if self.aggregation_level:
            df = df.group_by(self.aggregation_level).agg(
                [
                    nw.col(self.pred_column_name).sum().alias(self.pred_column_name),
                    nw.col(self.target).sum().alias(self.target),
                ]
            )
            if not hasattr(df, "to_native"):
                df = nw.from_native(df)

        if self.granularity:
            results = {}

            gran_df_unique = df.select(self.granularity).unique()
            granularity_tuples = [tuple(row) for row in gran_df_unique.iter_rows()]

            for gran_tuple in granularity_tuples:
                mask = None
                for i, col in enumerate(self.granularity):
                    col_mask = nw.col(col) == nw.lit(gran_tuple[i])
                    mask = col_mask if mask is None else (mask & col_mask)
                gran_df = df.filter(mask)

                if len(gran_df) > 0 and isinstance(
                    gran_df[self.pred_column_name].to_list()[0], list
                ):
                    score = self.scorer_function(
                        gran_df[self.target],
                        [item for item in gran_df[self.pred_column_name].to_list()],
                    )
                else:
                    score = self.scorer_function(
                        gran_df[self.target].to_list(),
                        gran_df[self.pred_column_name].to_list(),
                        **self.params,
                    )
                results[gran_tuple] = float(score)

            return results

        if len(df) > 0 and isinstance(df[self.pred_column_name].to_list()[0], list):
            return float(
                self.scorer_function(
                    df[self.target],
                    [item for item in df[self.pred_column_name].to_list()],
                )
            )

        return float(
            self.scorer_function(
                df[self.target].to_list(), df[self.pred_column_name].to_list(), **self.params
            )
        )


class ProbabilisticMeanBias(BaseScorer):

    def __init__(
        self,
        pred_column: str,
        target: str,
        class_column_name: str = "classes",
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
    ):

        self.pred_column_name = pred_column
        self.class_column_name = class_column_name
        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
        )

    def _calculate_score_for_group(self, df: pd.DataFrame) -> float:
        """Calculate score for a single group (used for granularity)"""
        df = df.copy()
        df.reset_index(drop=True, inplace=True)

        distinct_classes_variations = df.drop_duplicates(subset=[self.class_column_name])[
            self.class_column_name
        ].tolist()

        sum_lrs = [0 for _ in range(len(distinct_classes_variations))]
        sum_lr = 0
        for variation_idx, distinct_class_variation in enumerate(distinct_classes_variations):

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
                    probs.apply(lambda x: x[idx + 1]) + rows_target_group[last_column_name]
                )

                count_exact = len(rows_target_group[rows_target_group[self.target] == class_])
                weight_class = count_exact / len(rows_target_group)

                if self.aggregation_level:
                    grouped = (
                        rows_target_group.groupby(self.aggregation_level + [self.target])[
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

                grouped.loc[grouped[self.target] <= class_, "__went_under"] = 1
                grouped.loc[grouped[self.target] > class_, "__went_under"] = 0

                under_prob_mean = grouped[prob_under].mean()
                under_actual_mean = grouped["__went_under"].mean()

                overbias = under_prob_mean - under_actual_mean
                sum_lrs[variation_idx] += overbias * weight_class

                last_column_name = prob_under
            sum_lr += sum_lrs[variation_idx] * len(rows_target_group) / len(df)
        return sum_lr

    def score(self, df: pd.DataFrame) -> float | dict[tuple, float]:
        df = df.copy()
        df = apply_filters(df, self.filters)

        # Apply aggregation_level if set
        if self.aggregation_level:
            df = (
                df.groupby(self.aggregation_level)
                .agg(
                    {self.pred_column: "mean", self.target: "mean", self.class_column_name: "first"}
                )
                .reset_index()
            )

        # If granularity is set, calculate separate scores per group
        if self.granularity:
            results = {}
            granularity_groups = df.groupby(self.granularity)
            for gran_tuple, gran_df in granularity_groups:
                if isinstance(gran_tuple, tuple):
                    results[gran_tuple] = self._calculate_score_for_group(gran_df)
                else:
                    # Single column granularity
                    results[(gran_tuple,)] = self._calculate_score_for_group(gran_df)
            return results

        # Single score calculation
        return self._calculate_score_for_group(df)


class OrdinalLossScorer(BaseScorer):
    def __init__(
        self,
        pred_column: str,
        target: str,
        classes: list[int],
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
    ):
        self.pred_column_name = pred_column
        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
        )
        self.classes = classes

    def _calculate_score_for_group(self, df: pl.DataFrame) -> float:
        """Calculate score for a single group (used for granularity)"""
        pred_dtype = df.schema[self.pred_column]

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

        df = df.with_columns([get_expr(i).alias(f"prob_{c}") for i, c in enumerate(class_labels)])

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

            df = df.with_columns(pl.col(prob_col_under).clip(0.0001, 0.9999).alias(prob_col_under))

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

    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        df = apply_filters(df, self.filters)
        # Ensure df is a Narwhals DataFrame
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)

        # Convert to Polars DataFrame for _calculate_score_for_group
        df_native = df.to_native()
        if isinstance(df_native, pd.DataFrame):
            df_pl = pl.DataFrame(df_native)
        else:
            df_pl = df_native

        # Apply aggregation_level if set
        if self.aggregation_level:
            df_pl = df_pl.group_by(self.aggregation_level).agg(
                [
                    pl.col(self.pred_column).mean().alias(self.pred_column),
                    pl.col(self.target).mean().alias(self.target),
                ]
            )

        # If granularity is set, calculate separate scores per group
        if self.granularity:
            results = {}
            granularity_values = df_pl.select(self.granularity).unique().to_dict(as_series=False)
            granularity_tuples = list(zip(*[granularity_values[col] for col in self.granularity]))

            for gran_tuple in granularity_tuples:
                # Filter to this granularity combination
                mask = None
                for i, col in enumerate(self.granularity):
                    col_mask = pl.col(col) == gran_tuple[i]
                    mask = col_mask if mask is None else (mask & col_mask)
                gran_df = df_pl.filter(mask)

                # Calculate score for this group
                results[gran_tuple] = self._calculate_score_for_group(gran_df)

            return results

        # Single score calculation
        return self._calculate_score_for_group(df_pl)
