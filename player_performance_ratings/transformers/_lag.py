import logging
from typing import Optional, Union

import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT
import pandas as pd
import polars as pl

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformers.base_transformer import (
    BaseLagGenerator, required_lag_column_names,
)
from player_performance_ratings.utils import validate_sorting


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
            update_match_id_column: Optional[str] = None,
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
            column_names=column_names
        )
        self.days_between_lags = days_between_lags or []
        for days_lag in self.days_between_lags:
            self._features_out.append(f"{prefix}{days_lag}_days_ago")

        self.match_id_update_column = update_match_id_column
        self.lag_length = lag_length
        self.future_lag = future_lag
        self._df = None

    @nw.narwhalify
    @required_lag_column_names
    def transform_historical(self, df: FrameT, column_names: Optional[ColumnNames] = None) -> IntoFrameT:
        """ """
        input_cols = df.columns

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
            df = self._concat_with_stored_and_calculate_feats(df).sort('__row_index')

        if "is_future" in df.columns:
            df = df.drop("is_future")
        if ori_native == "pd":
            return df.select(list(set(input_cols + self.features_out))).to_pandas()
        return df.select(list(set(input_cols + self.features_out)))

    @nw.narwhalify
    def transform_future(self, df: FrameT) -> IntoFrameT:
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
            if '__row_index' not in concat_df.columns:
                concat_df = concat_df.with_row_index(name='__row_index')
            sort_col = '__row_index'

        grouped = concat_df.group_by(
            grp_cols
        ).agg(
            [nw.col(feature).mean().alias(feature) for feature in self.features]
            + [
                nw.col(sort_col)
            .min()
            .alias(sort_col)
            ]
        ).sort(sort_col)

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
                if output_column_name in concat_df.columns:
                    raise ValueError(
                        f"Column {output_column_name} already exists. Choose a different prefix or ensure no duplication was performed."
                    )

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
            grouped.select(
                grp_cols + feats_out
            ),
            on=grp_cols,
            how="left",
        ).sort(sort_col)

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class BinaryOutcomeRollingMeanTransformer(BaseLagGenerator):

    def __init__(
            self,
            features: list[str],
            window: int,
            binary_column: str,
            granularity: list[str] = None,
            prob_column: Optional[str] = None,
            min_periods: int = 1,
            add_opponent: bool = False,
            prefix: str = "rolling_mean_binary",
    ):
        super().__init__(
            features=features,
            add_opponent=add_opponent,
            prefix=prefix,
            iterations=[],
            granularity=granularity,
        )
        self.window = window
        self.min_periods = min_periods
        self.binary_column = binary_column
        self.prob_column = prob_column
        for feature_name in self.features:
            feature1 = f"{self.prefix}_{feature_name}{self.window}_1"
            feature2 = f"{self.prefix}_{feature_name}{self.window}_0"
            self._features_out.append(feature1)
            self._features_out.append(feature2)
            self._entity_features.append(feature1)
            self._entity_features.append(feature2)

            if self.add_opponent:
                self._features_out.append(
                    f"{self.prefix}_{feature_name}{self.window}_1_opponent"
                )
                self._features_out.append(
                    f"{self.prefix}_{feature_name}{self.window}_0_opponent"
                )

        if self.prob_column:
            for feature_name in self.features:
                prob_feature = (
                    f"{self.prefix}_{feature_name}_{self.prob_column}{self.window}"
                )
                self._features_out.append(prob_feature)

        self._estimator_features_out = self._features_out.copy()

    @nw.narwhalify
    def transform_historical(self, df: FrameT, column_names: ColumnNames) -> IntoFrameT:

        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            df = nw.from_native(pl.DataFrame(native))
            ori_native = "pd"
        else:
            ori_native = "pl"

        if df.schema[self.binary_column] in [nw.Float64, nw.Float32]:
            df = df.with_columns(nw.col(self.binary_column).cast(nw.Int64))

        df = df.with_columns(nw.lit(0).alias("is_future"))
        self.column_names = column_names
        self.granularity = self.granularity or [self.column_names.player_id]
        validate_sorting(df=df, column_names=self.column_names)
        additional_cols_to_use = [self.binary_column] + (
            [self.prob_column] if self.prob_column else []
        )
        self._store_df(df, additional_cols_to_use=additional_cols_to_use)
        concat_df = self._generate_concat_df_with_feats(df)
        concat_df = self._add_weighted_prob(transformed_df=concat_df)
        transformed_df = self._create_transformed_df(df=df, concat_df=concat_df)
        if "is_future" in transformed_df.columns:
            transformed_df = transformed_df.drop("is_future")
        transformed_df = self._add_weighted_prob(transformed_df=transformed_df)

        unique_cols = (
            [
                self.column_names.player_id,
                self.column_names.match_id,
                self.column_names.team_id,
            ]
            if self.column_names.team_id
            else [self.column_names.team_id, self.column_names.match_id]
        )

        if (
                transformed_df.unique(subset=unique_cols).shape[0]
                != transformed_df.shape[0]
        ):
            raise ValueError(
                f"Duplicated rows in df. Df must be a unique combination of {unique_cols}"
            )
        if ori_native == "pd":
            return transformed_df.to_pandas()
        return transformed_df

    @nw.narwhalify
    def transform_future(self, df: FrameT) -> IntoFrameT:

        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            df = nw.from_native(pl.DataFrame(native))
            ori_native = "pd"
        else:
            ori_native = "pl"

        if self._df is None:
            raise ValueError(
                "generate_historical needs to be called before generate_future"
            )

        if self.binary_column in df.columns:
            if df.schema[self.binary_column] in [nw.Float64, nw.Float32]:
                df = df.with_columns(nw.col(self.binary_column).cast(nw.Int64))
        df = df.with_columns(nw.lit(1).alias("is_future"))
        concat_df = self._generate_concat_df_with_feats(df=df)
        unique_match_ids = (
            df.select(nw.col(self.column_names.match_id))
            .unique()[self.column_names.match_id]
            .to_list()
        )
        transformed_df = concat_df.filter(
            nw.col(self.column_names.match_id).is_in(unique_match_ids)
        )
        transformed_future = self._generate_future_feats(
            transformed_df=transformed_df,
            ori_df=df,
            known_future_features=self._get_known_future_features(),
        )
        if "is_future" in transformed_future.columns:
            transformed_future = transformed_future.drop("is_future")
        transformed_future = self._add_weighted_prob(transformed_df=transformed_future)
        if ori_native == "pd":
            return transformed_future.to_pandas()
        return transformed_future

    def _get_known_future_features(self) -> list[str]:
        known_future_features = []
        if self.prob_column:
            for idx, feature_name in enumerate(self.features):
                weighted_prob_feat_name = (
                    f"{self.prefix}_{feature_name}_{self.prob_column}{self.window}"
                )
                known_future_features.append(weighted_prob_feat_name)

        return known_future_features

    def _add_weighted_prob(self, transformed_df: FrameT) -> FrameT:

        if self.prob_column:
            for idx, feature_name in enumerate(self.features):
                weighted_prob_feat_name = (
                    f"{self.prefix}_{feature_name}_{self.prob_column}{self.window}"
                )
                transformed_df = transformed_df.with_columns(
                    (
                            nw.col(f"{self.prefix}_{feature_name}{self.window}_1")
                            * nw.col(self.prob_column)
                            + nw.col(f"{self.prefix}_{feature_name}{self.window}_0")
                            * (1 - nw.col(self.prob_column))
                    ).alias(weighted_prob_feat_name)
                )
        return transformed_df

    def _generate_concat_df_with_feats(self, df: FrameT) -> FrameT:

        concat_df = self._concat_with_stored(df)

        groupby_cols = [
            self.column_names.update_match_id,
            *self.granularity,
            self.column_names.start_date,
            "is_future",
        ]

        aggregation = {f: nw.mean(f).alias(f) for f in self.features}
        aggregation[self.binary_column] = nw.median(self.binary_column)
        grouped = concat_df.group_by(groupby_cols).agg(list(aggregation.values()))

        # Sort the DataFrame
        grouped = grouped.sort(
            by=[
                self.column_names.start_date,
                "is_future",
                self.column_names.update_match_id,
            ]
        )

        feats_added = []

        for feature in self.features:
            grouped = grouped.with_columns(
                [
                    nw.when(nw.col(self.binary_column) == 1)
                    .then(nw.col(feature))
                    .alias("value_result_1"),
                    nw.when(nw.col(self.binary_column) == 0)
                    .then(nw.col(feature))
                    .alias("value_result_0"),
                ]
            )

            grouped = grouped.with_columns(
                [
                    nw.col("value_result_0")
                    .shift(1)
                    .over(self.granularity)
                    .alias("value_result_0_shifted"),
                    nw.col("value_result_1")
                    .shift(1)
                    .over(self.granularity)
                    .alias("value_result_1_shifted"),
                ]
            ).with_columns(
                [
                    nw.col("value_result_1_shifted")
                    .rolling_mean(window_size=self.window, min_samples=self.min_periods)
                    .over(self.granularity)
                    .alias(f"{self.prefix}_{feature}{self.window}_1"),
                    nw.col("value_result_0_shifted")
                    .rolling_mean(window_size=self.window, min_samples=self.min_periods)
                    .over(self.granularity)
                    .alias(f"{self.prefix}_{feature}{self.window}_0"),
                ]
            )

            feats_added.extend(
                [
                    f"{self.prefix}_{feature}{self.window}_1",
                    f"{self.prefix}_{feature}{self.window}_0",
                ]
            )

        concat_df = concat_df.join(
            grouped.select(
                [
                    self.column_names.update_match_id,
                    *self.granularity,
                    *feats_added,
                ]
            ),
            on=[self.column_names.update_match_id, *self.granularity],
            how="left",
        )

        concat_df = concat_df.with_columns(
            [nw.col(feats_added).fill_null(strategy="forward").over(self.granularity)]
        )

        return concat_df

