from typing import Literal

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge.data_structures import ColumnNames
from spforge.feature_generator._base import LagGenerator
from spforge.feature_generator._utils import (
    future_lag_transformations_wrapper,
    future_validator,
    historical_lag_transformations_wrapper,
    required_lag_column_names,
    transformation_validator,
)


class RollingWindowTransformer(LagGenerator):
    """
    Calculates the rolling mean for a list of features over a window of matches.
    Rolling Mean Values are also shifted by one match to avoid data leakage.

    Use .generate_historical() to generate rolling mean for historical data.
        The historical data is stored as instance-variables after calling .generate_historical()
    Use .generate_future() to generate rolling mean for future data after having called .generate_historical()
    """

    def __init__(
        self,
        features: list[str],
        window: int,
        granularity: list[str] | str,
        add_opponent: bool = False,
        scale_by_participation_weight: bool = False,
        min_periods: int = 1,
        are_estimator_features=True,
        prefix: str = "rolling_mean",
        aggregation: Literal["mean", "sum", "var"] = "mean",
        group_to_granularity: list[str] | None = None,
        unique_constraint: list[str] | None = None,
        match_id_column: str | None = None,
        update_column: str | None = None,
        max_days: int | None = None,
    ):
        """
        :param features:   Features to create rolling mean for
        :param window: Window size for rolling mean.
            If 10 will calculate rolling mean over the prior 10 observations
        :param min_periods: Minimum number of observations in window required to have a non-null result
        :param granularity: Columns to group by before rolling mean. E.g. player_id or [player_id, position].
             In the latter case it will get the rolling mean for each player_id and position combination.
             Defaults to player_id
        :param add_opponent: If True, will add new columns containing rolling mean for the opponent team.
        :param are_estimator_features: If True, the features will be added to the estimator features.
            If false, it makes it possible for the outer layer (Pipeline) to exclude these features from the estimator.
        :param prefix: Prefix for the new rolling mean columns
        :param max_days: Optional max lookback horizon in days. If set, observations older than max_days
            from the current row are ignored in the rolling calculation.
        """
        if prefix == "rolling_mean":
            prefix = (
                "rolling_mean"
                if aggregation == "mean"
                else "rolling_sum" if aggregation == "sum" else "rolling_var"
            )
        super().__init__(
            features=features,
            add_opponent=add_opponent,
            iterations=[window],
            prefix=prefix,
            granularity=granularity,
            are_estimator_features=are_estimator_features,
            unique_constraint=unique_constraint,
            group_to_granularity=group_to_granularity,
            update_column=update_column,
            match_id_column=match_id_column,
        )
        self.aggregation = aggregation
        self.scale_by_participation_weight = scale_by_participation_weight
        if self.aggregation == "var" and self.scale_by_participation_weight:
            raise NotImplementedError(
                "Rolling variance with participation weight is not implemented yet."
            )
        self.window = window
        self.min_periods = min_periods
        if max_days is not None and max_days <= 0:
            raise ValueError("max_days must be positive when provided")
        self.max_days = max_days

    @nw.narwhalify
    @historical_lag_transformations_wrapper
    @required_lag_column_names
    @transformation_validator
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        """
        Generates rolling mean for historical data
        Stored the historical data as instance-variables so it's possible to generate future data afterwards

        :param df: Historical data
        :param column_names: Column names
        """

        grouped = self._maybe_group(df)
        if self.column_names:
            self._store_df(grouped_df=grouped, ori_df=df)
            grouped_with_feats = self._generate_features(grouped, ori_df=df)
            df = self._merge_into_input_df(
                df=df,
                concat_df=grouped_with_feats,
            )

        else:
            join_on_cols = (
                self.group_to_granularity
                if self.group_to_granularity
                else [*self.granularity, self.update_column]
            )
            grouped_with_feats = self._generate_features(grouped, ori_df=df).sort("__row_index")
            df = df.join(
                grouped_with_feats.select([*join_on_cols, *self.features_out]),
                on=join_on_cols,
                how="left",
            ).unique("__row_index")

        return self._post_features_generated(df)

    @nw.narwhalify
    @future_lag_transformations_wrapper
    @future_validator
    @transformation_validator
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        """
        Generates rolling mean for future data
        Assumes that .generate_historical() has been called before
        Regardless of the scheduled data of the future match, all future matches are perceived as being played in the same date.
        That is to ensure that a team's 2nd future match has the same rolling means as the 1st future match.
        :param df: Future data
        """

        sort_col = self.column_names.start_date if self.column_names else "__row_index"
        grouped = self._group_to_granularity_level(df=df, sort_col=sort_col)
        grouped_df_with_feats = self._generate_features(df=grouped, ori_df=df)
        df = self._merge_into_input_df(df=df, concat_df=grouped_df_with_feats)
        df = self._post_features_generated(df)
        return self._forward_fill_future_features(df=df)

    def _generate_features(self, df: IntoFrameT, ori_df: IntoFrameT) -> IntoFrameT:

        if self.column_names and self._df is not None:
            sort_col = self.column_names.start_date
            concat_df = self._concat_with_stored(group_df=df, ori_df=ori_df)

        else:
            concat_df = df
            if "__row_index" not in concat_df.columns:
                concat_df = concat_df.with_row_index(name="__row_index")
            sort_col = "__row_index"
        concat_df = concat_df.sort(sort_col)

        if self.max_days is not None:
            concat_df = self._generate_features_with_max_days(concat_df)
            return concat_df

        agg_method = {
            "sum": lambda col: col.rolling_sum(
                window_size=self.window, min_samples=self.min_periods
            ),
            "mean": lambda col: col.rolling_mean(
                window_size=self.window, min_samples=self.min_periods
            ),
            "var": lambda col: col.rolling_var(
                window_size=self.window, min_samples=self.min_periods
            ),
        }
        if self.scale_by_participation_weight:

            concat_df = concat_df.with_columns(
                (nw.col(feature) * nw.col(self.column_names.participation_weight)).alias(
                    f"__scaled_{feature}"
                )
                for feature in self.features
            )
            scaled_feats = [f"__scaled_{feature}" for feature in self.features]

            rolling_sums = [
                agg_method["sum"](nw.col(feature_name).shift(n=1))
                .over(self.granularity)
                .alias(f"{self.prefix}_{feature_name}{self.window}__sum")
                for feature_name in [
                    *scaled_feats,
                    self.column_names.participation_weight,
                ]
            ]
            concat_df = concat_df.with_columns(rolling_sums)
            if self.aggregation == "mean":
                rolling_values = [
                    (
                        nw.col(f"{self.prefix}___scaled_{feature}{self.window}__sum")
                        / nw.col(
                            f"{self.prefix}_{self.column_names.participation_weight}{self.window}__sum"
                        )
                    ).alias(f"{self.prefix}_{feature}{self.window}")
                    for feature in self.features
                ]
                concat_df = concat_df.with_columns(rolling_values)

        else:
            rolling_values = [
                agg_method[self.aggregation](nw.col(feature_name).shift(n=1))
                .over(self.granularity)
                .alias(f"{self.prefix}_{feature_name}{self.window}")
                for feature_name in self.features
            ]

            concat_df = concat_df.with_columns(rolling_values)

        return concat_df.with_columns(
            [
                nw.col(f).fill_null(strategy="forward").over(self.granularity).alias(f)
                for f in self._entity_features_out
            ]
        )

    def _generate_features_with_max_days(self, concat_df: IntoFrameT) -> IntoFrameT:
        if self.column_names is None:
            raise ValueError("max_days requires column_names to be provided")

        if self.aggregation == "var":
            raise NotImplementedError("max_days is not supported for aggregation='var'")

        if self.scale_by_participation_weight and self.aggregation != "mean":
            raise NotImplementedError(
                "max_days with scale_by_participation_weight is only supported for aggregation='mean'"
            )

        date_col = self.column_names.start_date
        if date_col not in concat_df.columns:
            raise ValueError(f"Date column '{date_col}' not found in input dataframe")

        days_diff_divisor = 60 * 24
        for feature in self.features:
            sum_expr = nw.lit(0.0)
            count_expr = nw.lit(0)
            weight_sum_expr = nw.lit(0.0)
            temp_cols = []

            for lag in range(1, self.window + 1):
                lag_date_col = f"__max_days_date_lag_{feature}_{lag}"
                lag_feature_col = f"__max_days_feature_lag_{feature}_{lag}"
                temp_cols.extend([lag_date_col, lag_feature_col])

                if self.scale_by_participation_weight:
                    lag_weight_col = f"__max_days_weight_lag_{feature}_{lag}"
                    temp_cols.append(lag_weight_col)

                    concat_df = concat_df.with_columns(
                        [
                            nw.col(date_col).shift(lag).over(self.granularity).alias(lag_date_col),
                            (nw.col(feature) * nw.col(self.column_names.participation_weight))
                            .shift(lag)
                            .over(self.granularity)
                            .alias(lag_feature_col),
                            nw.col(self.column_names.participation_weight)
                            .shift(lag)
                            .over(self.granularity)
                            .alias(lag_weight_col),
                        ]
                    )
                else:
                    concat_df = concat_df.with_columns(
                        [
                            nw.col(date_col).shift(lag).over(self.granularity).alias(lag_date_col),
                            nw.col(feature).shift(lag).over(self.granularity).alias(lag_feature_col),
                        ]
                    )

                within_max_days = (
                    (
                        (
                            nw.col(date_col).cast(nw.Date) - nw.col(lag_date_col).cast(nw.Date)
                        ).dt.total_minutes()
                        / days_diff_divisor
                    )
                    <= self.max_days
                )

                concat_df = concat_df.with_columns(
                    nw.when(within_max_days).then(nw.col(lag_feature_col)).otherwise(nw.lit(None)).alias(
                        lag_feature_col
                    )
                )
                count_expr = count_expr + nw.when(nw.col(lag_feature_col).is_null()).then(0).otherwise(1)

                if self.scale_by_participation_weight:
                    concat_df = concat_df.with_columns(
                        nw.when(within_max_days)
                        .then(nw.col(lag_weight_col))
                        .otherwise(nw.lit(None))
                        .alias(lag_weight_col)
                    )
                    sum_expr = sum_expr + nw.col(lag_feature_col).fill_null(0.0)
                    weight_sum_expr = weight_sum_expr + nw.col(lag_weight_col).fill_null(0.0)
                else:
                    sum_expr = sum_expr + nw.col(lag_feature_col).fill_null(0.0)

            if self.scale_by_participation_weight:
                value_expr = nw.when(
                    (count_expr >= self.min_periods) & (weight_sum_expr > 0.0)
                ).then(sum_expr / weight_sum_expr).otherwise(nw.lit(None))
            elif self.aggregation == "sum":
                value_expr = nw.when(count_expr >= self.min_periods).then(sum_expr).otherwise(
                    nw.lit(None)
                )
            else:
                value_expr = nw.when(count_expr >= self.min_periods).then(sum_expr / count_expr).otherwise(
                    nw.lit(None)
                )

            concat_df = concat_df.with_columns(
                value_expr.alias(f"{self.prefix}_{feature}{self.window}")
            ).drop(temp_cols)

        return concat_df

    def _store_df(
        self,
        grouped_df: IntoFrameT,
        ori_df: nw.DataFrame | None = None,
        additional_cols: list[str] | None = None,
    ):
        super()._store_df(grouped_df=grouped_df, ori_df=ori_df, additional_cols=additional_cols)
        self._trim_stored_history_to_window()

    def _trim_stored_history_to_window(self) -> None:
        if self._df is None:
            return

        stored_df = nw.from_native(self._df)
        if not self.granularity:
            self._df = stored_df.tail(self.window).to_native()
            return

        sort_cols = [self.column_names.start_date]
        if self.update_column and self.update_column in stored_df.columns:
            sort_cols.append(self.update_column)

        self._df = (
            stored_df.sort(sort_cols, descending=True)
            .with_columns(
                nw.col(sort_cols[0]).cum_count().over(self.granularity).alias("__state_row_rank")
            )
            .filter(nw.col("__state_row_rank") <= self.window)
            .drop("__state_row_rank")
            .sort(sort_cols)
            .to_native()
        )
