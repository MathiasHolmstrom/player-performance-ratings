import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
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
        self._force_polars_backend = True

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

        if self.column_names:
            if df[self.date_column].dtype not in (nw.Date, nw.Datetime):
                df = df.with_columns(nw.col(self.date_column).alias("__ori_date"))
                try:
                    df = df.with_columns(
                        nw.col("__ori_date")
                        .str.to_datetime(format="%Y-%m-%d %H:%M:%S")
                        .alias(self.date_column)
                    )
                except nw.exceptions.InvalidOperationError:
                    df = df.with_columns(nw.col("__ori_date").cast(nw.Date).alias(self.date_column))
            self._store_df(nw.from_native(df))
            self._store_future_state()

            concat_df = self._concat_with_stored_and_calculate_feats(df.to_polars())
            if "__ori_date" in df.columns:
                df = df.with_columns(nw.col("__ori_date").alias(self.date_column))
            transformed_df = self._merge_into_input_df(
                df=nw.from_native(df),
                concat_df=nw.from_native(concat_df),
                match_id_join_on=self.match_id_update_column,
            )

        else:
            concat_df = nw.from_native(self._concat_with_stored_and_calculate_feats(df.to_polars()))
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

        state_df = nw.from_native(self._future_state_df)
        joined_df = df.join(state_df, on=self.granularity, how="left")

        if self.add_count:
            joined_df = joined_df.with_columns(
                nw.col(self._count_column_name).fill_null(0).alias(self._count_column_name)
            )

        return self._post_features_generated(joined_df)

    def _concat_with_stored_and_calculate_feats(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.column_names:
            concat_df = self._concat_with_stored(nw.from_native(df)).to_native()
        else:
            concat_df = df

        concat_df = concat_df.sort(self.date_column)
        days_str = str(self.days + 1) + "d"
        rolling_numerator_cols = self.features.copy()
        weight_col = (
            self.column_names.participation_weight
            if self.column_names and self.scale_by_participation_weight
            else None
        )

        concat_df = concat_df.with_columns(pl.lit(1).alias("__count1"))
        aggregation_columns = [pl.col("__count1").sum()]

        if self.scale_by_participation_weight:
            assert self.column_names is not None
            weighted_feature_columns = [f"__weighted_{col}" for col in self.features]
            concat_df = concat_df.with_columns(
                (pl.col(feature) * pl.col(weight_col)).alias(weighted_feature)
                for feature, weighted_feature in zip(
                    self.features, weighted_feature_columns, strict=True
                )
            )
            rolling_numerator_cols = weighted_feature_columns
            aggregation_columns.extend(
                [pl.col(weight_col).sum(), pl.col(weighted_feature_columns).sum()]
            )
        else:
            aggregation_columns.append(pl.col(self.features).sum())

        grp_cols = [self.date_column, *self.granularity]
        grp = concat_df.group_by(grp_cols).agg(aggregation_columns).sort(grp_cols)
        grp = grp.with_columns(
            pl.col("__count1")
            .rolling_sum_by(self.date_column, window_size=days_str)
            .over(self.granularity)
            .alias(self._count_column_name)
        ).with_columns(
            (pl.col(self._count_column_name) - pl.col("__count1")).alias(self._count_column_name)
        )

        grp = grp.with_columns(
            pl.col(col).sum().over([self.date_column, *self.granularity]).alias(f"days_sum_{col}")
            for col in rolling_numerator_cols
        )

        if self.scale_by_participation_weight:
            rolling_weight_col = "__rolling_participation_weight"
            grp = grp.with_columns(
                (
                    pl.col(weight_col)
                    .rolling_sum_by(self.date_column, window_size=days_str)
                    .over(self.granularity)
                    - pl.col(weight_col).sum().over([self.date_column, *self.granularity])
                ).alias(rolling_weight_col)
            )
            rolling_means = [
                pl.when(pl.col(rolling_weight_col) > 0)
                .then(
                    (
                        pl.col(weighted_feature)
                        .rolling_sum_by(self.date_column, window_size=days_str)
                        .over(self.granularity)
                        - pl.col(f"days_sum_{weighted_feature}")
                    )
                    / pl.col(rolling_weight_col)
                )
                .otherwise(pl.lit(None))
                .alias(f"{self.prefix}_{feature}{str(self.days)}")
                for feature, weighted_feature in zip(
                    self.features, rolling_numerator_cols, strict=True
                )
            ]
        else:
            rolling_means = [
                (
                    (
                        pl.col(col)
                        .rolling_sum_by(self.date_column, window_size=days_str)
                        .over(self.granularity)
                        - pl.col(f"days_sum_{col}")
                    )
                    / pl.col(self._count_column_name)
                ).alias(f"{self.prefix}_{feature}{str(self.days)}")
                for feature, col in zip(self.features, rolling_numerator_cols, strict=True)
            ]

        grp = grp.with_columns(rolling_means)
        grp = grp.with_columns(
            [
                pl.when(pl.col(f"{self.prefix}_{col}{str(self.days)}").is_nan())
                .then(pl.lit(None))
                .otherwise(pl.col(f"{self.prefix}_{col}{str(self.days)}"))
                .alias(f"{self.prefix}_{col}{str(self.days)}")
                for col in self.features
            ]
        )
        if self.min_games is not None:
            grp = grp.with_columns(
                [
                    pl.when(pl.col(self._count_column_name) >= self.min_games)
                    .then(pl.col(f"{self.prefix}_{col}{str(self.days)}"))
                    .otherwise(pl.lit(None))
                    .alias(f"{self.prefix}_{col}{str(self.days)}")
                    for col in self.features
                ]
            )

        df = df.join(grp, on=grp_cols, how="left")
        if self.add_count:
            df = df.with_columns(
                pl.col(self._count_column_name).fill_null(0).alias(self._count_column_name)
            )
        if "__ori_date" in df.columns:
            return df.with_columns(pl.col("__ori_date").alias(self.date_column))
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
                .otherwise(nw.lit(None))
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
                        .otherwise(nw.lit(None))
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

        stored_df = nw.from_native(self._df).to_polars()
        if self.date_column not in stored_df.columns:
            return

        max_date = stored_df.select(pl.col(self.date_column).max()).item()
        if max_date is None:
            return

        cutoff = max_date - pd.Timedelta(days=self.days)
        self._df = stored_df.filter(pl.col(self.date_column) >= cutoff)

    def reset(self):
        self._df = None
        self._fitted_game_ids = []
        self._future_state_df = None

    @property
    def features_out(self) -> list[str]:
        return self._features_out
