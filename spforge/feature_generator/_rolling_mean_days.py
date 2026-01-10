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


class RollingMeanDaysTransformer(LagGenerator):

    def __init__(
        self,
        features: list[str],
        days: int,
        granularity: list[str] | str,
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
        self.scale_by_participation_weight = scale_by_participation_weight
        super().__init__(
            column_names=column_names,
            features=features,
            iterations=[self.days],
            prefix=prefix,
            add_opponent=add_opponent,
            granularity=granularity,
            update_column=update_column,
            unique_constraint=unique_constraint,
        )

        self.add_count = add_count
        self.date_column = date_column
        self._fitted_game_ids = []

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
        ori_cols = df.columns
        if self.column_names and df[self.date_column].dtype not in (nw.Date, nw.Datetime):
            df = df.with_columns(nw.col(self.date_column).alias("__ori_date"))
            try:
                df = df.with_columns(
                    nw.col("__ori_date")
                    .str.to_datetime(format="%Y-%m-%d %H:%M:%S")
                    .alias(self.date_column)
                )
            except nw.exceptions.InvalidOperationError:
                df = df.with_columns(nw.col("__ori_date").cast(nw.Date).alias(self.date_column))

        if isinstance(df.to_native(), pd.DataFrame):
            ori_type = "pd"
            df = pl.DataFrame(df.to_native())
        else:
            df = df.to_native()
            ori_type = "pl"

        concat_df = self._concat_with_stored_and_calculate_feats(df=df)
        concat_df = concat_df.filter(
            pl.col(self.column_names.match_id).is_in(
                df[self.column_names.match_id].unique().to_list()
            )
        )
        concat_df = self._post_features_generated(nw.from_native(concat_df))
        transformed_future = self._forward_fill_future_features(df=nw.from_native(concat_df))
        transformed_future = transformed_future.select(ori_cols + self.features_out)

        if ori_type == "pd":
            transformed_future = nw.from_native(transformed_future.to_pandas())

        return transformed_future

    def _concat_with_stored_and_calculate_feats(self, df: pl.DataFrame) -> pl.DataFrame:

        if self.column_names:
            concat_df = self._concat_with_stored(nw.from_native(df)).to_native()
        else:
            concat_df = df

        concat_df = concat_df.sort(self.date_column)
        days_str = str(self.days + 1) + "d"
        cols = self.features.copy()

        concat_df = concat_df.with_columns(pl.lit(1).alias("__count1"))
        cols.append("__count1")

        grp_cols = [self.date_column, *self.granularity]
        grp = (
            concat_df.group_by(grp_cols)
            .agg([pl.col(self.features).sum(), pl.col("__count1").sum()])
            .sort(grp_cols)
        )
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
            for col in self.features
        )

        rolling_means = [
            (
                (
                    pl.col(col)
                    .rolling_sum_by(self.date_column, window_size=days_str)
                    .over(self.granularity)
                    - pl.col(f"days_sum_{col}")
                )
                / pl.col(self._count_column_name)
            ).alias(f"{self.prefix}_{col}{str(self.days)}")
            for col in self.features
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

        df = df.join(grp, on=grp_cols, how="left")
        if self.add_count:
            df = df.with_columns(
                pl.col(self._count_column_name).fill_null(0).alias(self._count_column_name)
            )
        if "__ori_date" in df.columns:
            return df.with_columns(pl.col("__ori_date").alias(self.date_column))
        return df

    def reset(self):
        self._df = None
        self._fitted_game_ids = []

    @property
    def features_out(self) -> list[str]:
        return self._features_out
