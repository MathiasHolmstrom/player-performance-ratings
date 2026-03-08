import narwhals.stable.v2 as nw
import pandas as pd
from narwhals.typing import IntoFrameT

from spforge.data_structures import ColumnNames
from spforge.feature_generator._base import LagGenerator
from spforge.feature_generator._utils import (
    future_lag_transformations_wrapper,
    future_validator,
    historical_lag_transformations_wrapper,
    numeric_null_literal,
    required_lag_column_names,
    transformation_validator,
)

_MISSING_STATE_MSG = (
    "No stored rolling state found. Call fit_transform() before future_transform()."
)


class RollingMeanDaysTransformer(LagGenerator):
    def __init__(
        self,
        features: list[str],
        days: int,
        granularity: list[str] | str,
        min_games: int | None = None,
        scale_by_participation_weight: bool = False,
        add_count: bool = False,
        add_opponent: bool = False,
        prefix: str = "rolling_mean_days",
        column_names: ColumnNames | None = None,
        date_column: str | None = None,
        update_column: str | None = None,
        unique_constraint: list[str] | None = None,
    ):
        self.days = days
        if min_games is not None and min_games <= 0:
            raise ValueError("min_games must be positive when provided")
        self.min_games = min_games
        super().__init__(
            column_names=column_names,
            features=features,
            iterations=[self.days],
            prefix=prefix,
            add_opponent=add_opponent,
            granularity=granularity,
            update_column=update_column,
            unique_constraint=unique_constraint,
            scale_by_participation_weight=scale_by_participation_weight,
        )

        self.add_count = add_count
        self.date_column = date_column
        self._fitted_game_ids = []
        self._future_state_df = None

        self._count_column_name = f"{self.prefix}_count{str(self.days)}"
        if self.add_count:
            self._features_out.append(self._count_column_name)
            self._entity_features_out.append(self._count_column_name)

            if self.add_opponent:
                self._features_out.append(f"{self._count_column_name}_opponent")

    @nw.narwhalify
    @historical_lag_transformations_wrapper
    @required_lag_column_names
    @transformation_validator
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        if not self.column_names and not self.date_column:
            raise ValueError("column_names or date_column must be provided")

        if self.column_names:
            self.date_column = self.column_names.start_date or self.date_column
            self.match_id_update_column = self.column_names.update_match_id or self.update_column
            self.match_id_column = self.column_names.match_id

        if self.column_names:
            if df.schema[self.date_column] not in (nw.Date, nw.Datetime):
                df = df.with_columns(nw.col(self.date_column).alias("__ori_date"))
                df = df.with_columns(nw.col("__ori_date").str.to_datetime().alias(self.date_column))
                df = df.with_columns(
                    nw.when(nw.col(self.date_column).is_null())
                    .then(nw.col("__ori_date").str.to_date().cast(nw.Datetime))
                    .otherwise(nw.col(self.date_column))
                    .alias(self.date_column)
                )
            self._store_df(nw.from_native(df))
            self._store_future_state()

            concat_df = self._concat_with_stored_and_calculate_feats(nw.from_native(df))
            if "__ori_date" in df.columns:
                df = df.with_columns(nw.col("__ori_date").alias(self.date_column))
            transformed_df = self._merge_into_input_df(
                df=nw.from_native(df),
                concat_df=concat_df,
                match_id_join_on=self.match_id_update_column,
            )

        else:
            concat_df = self._concat_with_stored_and_calculate_feats(df)
            transformed_df = df.join(
                concat_df.select(["__row_index", *self.features_out]),
                on="__row_index",
                how="left",
            ).sort("__row_index")

        transformed_df = self._post_features_generated(transformed_df)
        if self.add_opponent and self.add_count:
            transformed_df = transformed_df.with_columns(
                nw.col(f"{self._count_column_name}_opponent")
                .fill_null(0)
                .alias(f"{self._count_column_name}_opponent")
            )
        return transformed_df

    @nw.narwhalify
    @future_validator
    @future_lag_transformations_wrapper
    @transformation_validator
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        return self._future_transform_from_state(df)

    def _future_transform_from_state(self, df: IntoFrameT) -> IntoFrameT:
        if self._future_state_df is None:
            raise ValueError(_MISSING_STATE_MSG)

        state_df = self._align_backend(df, nw.from_native(self._future_state_df))
        joined_df = df.join(state_df, on=self.granularity, how="left")

        if self.add_count:
            joined_df = joined_df.with_columns(
                nw.col(self._count_column_name).fill_null(0).alias(self._count_column_name)
            )

        return self._post_features_generated(joined_df)

    def _concat_with_stored_and_calculate_feats(self, df: IntoFrameT) -> IntoFrameT:
        concat_df = self._concat_with_stored(df) if self.column_names else df
        concat_df = concat_df.sort([*self.granularity, self.date_column])
        rolling_numerator_cols = self.features.copy()
        weight_col = (
            self.column_names.participation_weight
            if self.column_names and self.scale_by_participation_weight
            else None
        )

        concat_df = concat_df.with_columns(nw.lit(1).alias("__count1"))
        aggregation_columns = [nw.col("__count1").sum().alias("__count1")]

        if self.scale_by_participation_weight:
            assert self.column_names is not None
            weighted_feature_columns = [f"__weighted_{col}" for col in self.features]
            concat_df = concat_df.with_columns(
                (nw.col(feature) * nw.col(weight_col)).alias(weighted_feature)
                for feature, weighted_feature in zip(
                    self.features, weighted_feature_columns, strict=True
                )
            )
            rolling_numerator_cols = weighted_feature_columns
            aggregation_columns.extend(
                [nw.col(weight_col).sum().alias(weight_col)]
                + [
                    nw.col(weighted_feature).sum().alias(weighted_feature)
                    for weighted_feature in weighted_feature_columns
                ]
            )
        else:
            aggregation_columns.extend(
                [nw.col(feature).sum().alias(feature) for feature in self.features]
            )

        grp_cols = [self.date_column, *self.granularity]
        daily = (
            concat_df.group_by(grp_cols)
            .agg(aggregation_columns)
            .sort([*self.granularity, self.date_column])
        )
        daily = daily.with_columns(
            nw.col("__count1")
            .cum_sum()
            .over(self.granularity, order_by=[self.date_column])
            .alias("__cum_count")
        )
        daily = daily.with_columns(
            (nw.col("__cum_count") - nw.col("__count1")).alias("__prev_count")
        )

        daily = daily.with_columns(
            [
                nw.col(col)
                .cum_sum()
                .over(self.granularity, order_by=[self.date_column])
                .alias(f"__cum_{col}")
                for col in rolling_numerator_cols
            ]
        )
        daily = daily.with_columns(
            [
                (nw.col(f"__cum_{col}") - nw.col(col)).alias(f"__prev_{col}")
                for col in rolling_numerator_cols
            ]
        )

        if self.scale_by_participation_weight:
            daily = daily.with_columns(
                nw.col(weight_col)
                .cum_sum()
                .over(self.granularity, order_by=[self.date_column])
                .alias("__cum_weight")
            )
            daily = daily.with_columns(
                (nw.col("__cum_weight") - nw.col(weight_col)).alias("__prev_weight")
            )

        lookup_cols = [
            *self.granularity,
            self.date_column,
            "__prev_count",
            *[f"__prev_{col}" for col in rolling_numerator_cols],
        ]
        if self.scale_by_participation_weight:
            lookup_cols.append("__prev_weight")

        lookup = daily.select(lookup_cols).sort([self.date_column, *self.granularity])

        daily = daily.with_columns(
            nw.col(self.date_column).dt.offset_by(f"-{self.days}d").alias("__cutoff_date")
        ).sort(["__cutoff_date", *self.granularity])

        daily = daily.join_asof(
            lookup,
            left_on="__cutoff_date",
            right_on=self.date_column,
            by=self.granularity,
            strategy="forward",
            suffix="_lower",
        )

        daily = daily.with_columns(
            (nw.col("__prev_count") - nw.col("__prev_count_lower").fill_null(0)).alias(
                self._count_column_name
            )
        )

        if self.scale_by_participation_weight:
            daily = daily.with_columns(
                (nw.col("__prev_weight") - nw.col("__prev_weight_lower").fill_null(0.0)).alias(
                    "__window_weight"
                )
            )
            rolling_means = [
                self._float_output(
                    nw.when(
                        (nw.col(self._count_column_name) >= (self.min_games or 1))
                        & (nw.col("__window_weight") > 0)
                    ).then(
                        (
                            nw.col(f"__prev_{weighted_feature}")
                            - nw.col(f"__prev_{weighted_feature}_lower").fill_null(0.0)
                        )
                        / nw.col("__window_weight")
                    ),
                    f"{self.prefix}_{feature}{self.days}",
                    daily,
                )
                for feature, weighted_feature in zip(
                    self.features, rolling_numerator_cols, strict=True
                )
            ]
        else:
            rolling_means = [
                self._float_output(
                    nw.when(nw.col(self._count_column_name) >= (self.min_games or 1)).then(
                        (nw.col(f"__prev_{col}") - nw.col(f"__prev_{col}_lower").fill_null(0.0))
                        / nw.col(self._count_column_name)
                    ),
                    f"{self.prefix}_{feature}{self.days}",
                    daily,
                )
                for feature, col in zip(self.features, rolling_numerator_cols, strict=True)
            ]

        daily = daily.with_columns(rolling_means)
        df = concat_df.join(
            daily.select([*grp_cols, *self._entity_features_out]),
            on=grp_cols,
            how="left",
        )
        if self.add_count:
            df = df.with_columns(
                nw.col(self._count_column_name).fill_null(0).alias(self._count_column_name)
            )
        return df

    def _store_future_state(self) -> None:
        """Precompute per-granularity rolling mean from the trimmed stored history.

        rolling_sum_by uses a left-exclusive window: (D - window, D].
        For a future game at max_date + 1, that means date > max_date - days.
        _trim_stored_history_to_days_window uses >=, so we re-filter here to >
        to match the exact rolling window semantics.
        """
        if self._df is None:
            self._future_state_df = None
            return

        history_df = nw.from_native(self._df)

        max_date_val = history_df[self.date_column].max()
        if max_date_val is None:
            self._future_state_df = None
            return

        exclusive_cutoff = max_date_val - pd.Timedelta(days=self.days)
        history_df = history_df.filter(nw.col(self.date_column) > nw.lit(exclusive_cutoff))
        feature_state_cols = [f"{self.prefix}_{feature}{self.days}" for feature in self.features]

        if self.scale_by_participation_weight:
            weight_col = self.column_names.participation_weight
            total_weight = nw.col(weight_col).sum().over(self.granularity)
            state_exprs = [
                nw.when(total_weight > 0)
                .then(
                    (nw.col(feature) * nw.col(weight_col)).sum().over(self.granularity)
                    / total_weight
                )
                .otherwise(numeric_null_literal(history_df))
                .alias(state_col)
                for feature, state_col in zip(self.features, feature_state_cols, strict=True)
            ]
        else:
            state_exprs = [
                nw.col(feature).mean().over(self.granularity).alias(state_col)
                for feature, state_col in zip(self.features, feature_state_cols, strict=True)
            ]

        history_df = history_df.with_columns(state_exprs)

        agg_cols = list(feature_state_cols)

        if self.min_games is not None or self.add_count:
            count_tmp = "__state_count"
            history_df = history_df.with_columns(
                nw.col(self.features[0]).count().over(self.granularity).alias(count_tmp)
            )
            if self.min_games is not None:
                history_df = history_df.with_columns(
                    [
                        nw.when(nw.col(count_tmp) >= self.min_games)
                        .then(nw.col(state_col))
                        .otherwise(numeric_null_literal(history_df))
                        .alias(state_col)
                        for state_col in feature_state_cols
                    ]
                )
            if self.add_count:
                history_df = history_df.rename({count_tmp: self._count_column_name})
                agg_cols.append(self._count_column_name)
            else:
                history_df = history_df.drop(count_tmp)

        self._future_state_df = (
            history_df.group_by(self.granularity)
            .agg([nw.col(col).last().alias(col) for col in agg_cols])
            .to_native()
        )

    def _store_df(
        self,
        grouped_df: IntoFrameT,
        ori_df: nw.DataFrame | None = None,
        additional_cols: list[str] | None = None,
    ):
        super()._store_df(grouped_df=grouped_df, ori_df=ori_df, additional_cols=additional_cols)
        self._trim_stored_history_to_days_window()

    def _trim_stored_history_to_days_window(self) -> None:
        if self._df is None or not self.date_column:
            return

        stored_df = nw.from_native(self._df)
        if self.date_column not in stored_df.columns:
            return

        max_date = stored_df[self.date_column].max()
        if max_date is None:
            return

        cutoff = max_date - pd.Timedelta(days=self.days)
        self._df = stored_df.filter(nw.col(self.date_column) >= nw.lit(cutoff)).to_native()

    def _float_output(self, expr: nw.Expr, alias: str, df: IntoFrameT) -> nw.Expr:
        return expr.otherwise(numeric_null_literal(df)).cast(nw.Float64).alias(alias)

    def reset(self):
        self._df = None
        self._fitted_game_ids = []
        self._future_state_df = None

    @property
    def features_out(self) -> list[str]:
        return self._features_out
