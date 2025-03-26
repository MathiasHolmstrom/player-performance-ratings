import logging
from typing import Optional

import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT
import pandas as pd
import polars as pl

from spforge import ColumnNames
from spforge.transformers.base_transformer import (
    BaseLagGenerator,
    required_lag_column_names,
    row_count_validator,
)
from spforge.utils import validate_sorting


class LagTransformer(BaseLagGenerator):

    def __init__(
        self,
        features: list[str],
        lag_length: int,
        granularity: list[str],
        days_between_lags: Optional[list[int]] = None,
        future_lag: bool = False,
        prefix: str = "lag",
        add_opponent: bool = False,
        match_id_update_column: Optional[str] = None,
        column_names: Optional[ColumnNames] = None,
    ):
        """
        :param features. List of features to create lags for
        :param lag_length: Number of lags
        :param granularity: Columns to group by before lagging. E.g. player_id or [player_id, position].
            In the latter case it will get the lag for each player_id and position combination.
            Defaults to
        :param days_between_lags. Adds a column with the number of days between the lagged date and the current date.
        :param future_lag: If True, the lag will be calculated for the future instead of the past.
        :param prefix:
            Prefix for the new lag columns
        """

        super().__init__(
            features=features,
            add_opponent=add_opponent,
            prefix=prefix,
            iterations=[i for i in range(1, lag_length + 1)],
            granularity=granularity,
            column_names=column_names,
            match_id_update_column=match_id_update_column,
        )
        self.days_between_lags = days_between_lags or []
        for days_lag in self.days_between_lags:
            self._features_out.append(f"{prefix}{days_lag}_days_ago")

        self.lag_length = lag_length
        self.future_lag = future_lag
        self._df = None

    @nw.narwhalify
    @required_lag_column_names
    @row_count_validator
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        """ """
        input_cols = df.columns
        df = df.drop([f for f in self.features_out if f in df.columns])
        self.column_names = column_names or self.column_names

        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            ori_native = "pd"
            df = nw.from_native(pl.DataFrame(native))
        else:
            ori_native = "pl"

        if self.column_names:
            df = df.with_columns(nw.lit(0).alias("is_future"))
            validate_sorting(df=df, column_names=self.column_names)
            self._store_df(df)
            concat_df = self._concat_with_stored_and_calculate_feats(df)
            df = self._create_transformed_df(
                df=df, concat_df=concat_df, match_id_join_on=self.column_names.match_id
            )
        else:
            df = self._concat_with_stored_and_calculate_feats(df).sort("__row_index")
        df = df.select(list(set(input_cols + self.features_out)))
        if "__row_index" in df.columns:
            df = df.drop("__row_index")
        if ori_native == "pd":
            return df.to_pandas()
        return df

    @nw.narwhalify
    @row_count_validator
    def transform_future(self, df: FrameT) -> IntoFrameT:
        df = df.drop([f for f in self.features_out if f in df.columns])
        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            df = nw.from_native(pl.DataFrame(native))
            ori_native = "pd"
        else:
            ori_native = "pl"
        df = df.with_columns(nw.lit(1).alias("is_future"))
        concat_df = self._concat_with_stored_and_calculate_feats(df=df)

        transformed_df = concat_df.filter(
            nw.col(self.match_id_update_column).is_in(
                df.select(
                    nw.col(self.match_id_update_column)
                    #   .cast(nw.String)
                )
                .unique()[self.match_id_update_column]
                .to_list()
            )
        )

        transformed_future = self._generate_future_feats(
            transformed_df=transformed_df, ori_df=df
        )
        if "is_future" in transformed_future.columns:
            transformed_future = transformed_future.drop("is_future")
        if ori_native == "pd":
            return transformed_future.to_pandas()
        return transformed_future

    def _concat_with_stored_and_calculate_feats(self, df: FrameT) -> FrameT:
        grp_cols = self.granularity + [self.match_id_update_column]
        if self.column_names and self._df is not None:
            concat_df = self._concat_with_stored(df=df)
            sort_col = self.column_names.start_date

        else:
            concat_df = df
            if "__row_index" not in concat_df.columns:
                concat_df = concat_df.with_row_index(name="__row_index")
            sort_col = "__row_index"

        grouped = (
            concat_df.group_by(grp_cols)
            .agg(
                [nw.col(feature).mean().alias(feature) for feature in self.features]
                + [nw.col(sort_col).min().alias(sort_col)]
            )
            .sort(sort_col)
        )

        for days_lag in self.days_between_lags:
            if self.future_lag:
                grouped = grouped.with_columns(
                    nw.col(self.column_names.start_date)
                    .shift(-days_lag)
                    .over(self.granularity)
                    .alias("shifted_days")
                )
                grouped = grouped.with_columns(
                    (
                        (
                            nw.col("shifted_days").cast(nw.Date)
                            - nw.col(self.column_names.start_date).cast(nw.Date)
                        ).dt.total_minutes()
                        / 60
                        / 24
                    ).alias(f"{self.prefix}{days_lag}_days_ago")
                )
            else:
                grouped = grouped.with_columns(
                    nw.col(self.column_names.start_date)
                    .shift(days_lag)
                    .over(self.granularity)
                    .alias("shifted_days")
                )
                grouped = grouped.with_columns(
                    (
                        (
                            nw.col(self.column_names.start_date).cast(nw.Date)
                            - nw.col("shifted_days").cast(nw.Date)
                        ).dt.total_minutes()
                        / 60
                        / 24
                    ).alias(f"{self.prefix}{days_lag}_days_ago")
                )
            grouped = grouped.drop("shifted_days")

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f"{self.prefix}_{feature_name}{lag}"
                if self.future_lag:
                    grouped = grouped.with_columns(
                        nw.col(feature_name)
                        .shift(-lag)
                        .over(self.granularity)
                        .alias(output_column_name)
                    )
                else:
                    grouped = grouped.with_columns(
                        nw.col(feature_name)
                        .shift(lag)
                        .over(self.granularity)
                        .alias(output_column_name)
                    )

        feats_out = []
        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                feats_out.append(f"{self.prefix}_{feature_name}{lag}")

        for days_lag in self.days_between_lags:
            feats_out.append(f"{self.prefix}{days_lag}_days_ago")

        return concat_df.join(
            grouped.select(grp_cols + feats_out),
            on=grp_cols,
            how="left",
        ).sort(sort_col)

    @property
    def features_out(self) -> list[str]:
        return self._features_out
