from typing import Optional, Union

import numpy as np
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT
import pandas as pd
import polars as pl

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformers.base_transformer import (
    BaseLagGenerator,
)
from player_performance_ratings.utils import validate_sorting


class LagTransformer(BaseLagGenerator):

    def __init__(
        self,
        features: list[str],
        lag_length: int,
        granularity: Optional[list[str]] = None,
        days_between_lags: Optional[list[int]] = None,
        scale_by_participation_weight: bool = False,
        future_lag: bool = False,
        prefix: str = "lag",
        add_opponent: bool = False,
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
        )
        self.scale_by_participation_weight = scale_by_participation_weight
        self.days_between_lags = days_between_lags or []
        for days_lag in self.days_between_lags:
            self._features_out.append(f"{prefix}{days_lag}_days_ago")

        self.lag_length = lag_length
        self.future_lag = future_lag
        self._df = None

    @nw.narwhalify
    def generate_historical(self, df: FrameT, column_names: ColumnNames) -> IntoFrameT:
        """ """
        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            ori_native = "pd"
            df = nw.from_native(pl.DataFrame(native))
        else:
            ori_native = "pl"

        df = df.with_columns(nw.lit(0).alias("is_future"))
        self.column_names = column_names
        self.granularity = self.granularity or [self.column_names.player_id]
        validate_sorting(df=df, column_names=self.column_names)
        self._store_df(df)
        concat_df = self._generate_concat_df_with_feats(df)
        df = self._create_transformed_df(df=df, concat_df=concat_df)
        if "is_future" in df.columns:
            df = df.drop("is_future")
        if ori_native == "pd":
            return df.to_pandas()
        return df

    @nw.narwhalify
    def generate_future(self, df: FrameT) -> IntoFrameT:
        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            df = nw.from_native(pl.DataFrame(native))
            ori_native = "pd"
        else:
            ori_native = "pl"
        df = df.with_columns(nw.lit(1).alias("is_future"))
        concat_df = self._generate_concat_df_with_feats(df=df)

        transformed_df = concat_df.filter(
            nw.col(self.column_names.match_id).is_in(
                df.select(
                    nw.col(self.column_names.match_id)
                    #   .cast(nw.String)
                )
                .unique()[self.column_names.match_id]
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

    def _generate_concat_df_with_feats(self, df: FrameT) -> FrameT:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        if len(
            df.unique(
                subset=[
                    self.column_names.player_id,
                    self.column_names.team_id,
                    self.column_names.match_id,
                ]
            )
        ) != len(df):
            raise ValueError(
                f"Duplicated rows in df. Df must be a unique combination of {self.column_names.player_id} and {self.column_names.update_match_id}"
            )

        concat_df = self._concat_df(df=df)

        if (
            self.column_names.participation_weight
            and self.scale_by_participation_weight
        ):
            for feature in self.features:
                concat_df = concat_df.with_columns(
                    (
                        nw.col(feature) * nw.col(self.column_names.participation_weight)
                    ).alias(feature)
                )

        grouped = concat_df.group_by(
            self.granularity
            + [self.column_names.update_match_id, self.column_names.start_date]
        ).agg([nw.col(feature).mean().alias(feature) for feature in self.features])

        grouped = grouped.sort(
            [self.column_names.start_date, self.column_names.update_match_id]
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

        concat_df = concat_df.join(
            grouped.select(
                self.granularity
                + [self.column_names.update_match_id, self.column_names.start_date]
                + feats_out
            ),
            on=self.granularity
            + [self.column_names.update_match_id, self.column_names.start_date],
            how="left",
        )

        return concat_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class RollingMeanDaysTransformer(BaseLagGenerator):

    def __init__(
        self,
        features: list[str],
        days: Union[int, list[int]],
        granularity: Union[list[str], str] = None,
        scale_by_participation_weight: bool = False,
        add_count: bool = False,
        add_opponent: bool = False,
        prefix: str = "rolling_mean_days_",
    ):
        self.days = days
        self.scale_by_participation_weight = scale_by_participation_weight
        if isinstance(self.days, int):
            self.days = [self.days]
        super().__init__(
            features=features,
            iterations=[i for i in self.days],
            prefix=prefix,
            add_opponent=add_opponent,
            granularity=granularity,
        )

        self.add_count = add_count
        self._fitted_game_ids = []

        for day in self.days:
            if self.add_count:
                feature = f"{self.prefix}{day}_count"
                self._features_out.append(feature)
                self._entity_features.append(feature)

                if self.add_opponent:
                    self._features_out.append(f"{self.prefix}{day}_count_opponent")

    def generate_historical(
        self, df: pd.DataFrame, column_names: ColumnNames
    ) -> pd.DataFrame:
        ori_types = {c: df[c].dtype for c in df.columns}
        df = df.assign(is_future=0)
        self.column_names = column_names
        self.granularity = self.granularity or [self.column_names.player_id]
        validate_sorting(df=df, column_names=self.column_names)
        self._store_df(df)
        concat_df = self._generate_concat_df_with_feats(df)
        df = self._create_transformed_df(df=df, concat_df=concat_df)

        if self.add_count:
            for day in self.days:
                df = df.assign(
                    **{
                        f"{self.prefix}{day}_count": df[
                            f"{self.prefix}{day}_count"
                        ].fillna(0)
                    }
                )
                if self.add_opponent:
                    df = df.assign(
                        **{
                            f"{self.prefix}{day}_count_opponent": df[
                                f"{self.prefix}{day}_count_opponent"
                            ].fillna(0)
                        }
                    )
        if "is_future" in df.columns:
            df = df.drop(columns="is_future")

        for c, t in ori_types.items():
            if c not in self.features_out and df[c].dtype != t:
                df[c] = df[c].astype(t)
        return df

    def generate_future(self, df: pd.DataFrame) -> pd.DataFrame:

        ori_types = {c: df[c].dtype for c in df.columns}
        df = df.assign(is_future=1)
        concat_df = self._generate_concat_df_with_feats(df=df)
        transformed_df = concat_df[
            concat_df[self.column_names.match_id].isin(
                df[self.column_names.match_id].astype("str").unique().tolist()
            )
        ]
        transformed_future = self._generate_future_feats(
            transformed_df=transformed_df, ori_df=df
        )
        if "is_future" in transformed_future.columns:
            transformed_future = transformed_future.drop(columns="is_future")

        for c, t in ori_types.items():
            if c not in self.features_out and transformed_future[c].dtype != t:
                transformed_future[c] = transformed_future[c].astype(t)
        return transformed_future

    def _generate_concat_df_with_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        concat_df = self._concat_df(df)

        for feature_name in self.features:

            if (
                self.column_names.participation_weight
                and self.scale_by_participation_weight
            ):
                concat_df = concat_df.assign(
                    **{
                        feature_name: concat_df[feature_name]
                        * concat_df[self.column_names.participation_weight]
                    }
                )

        concat_df = concat_df.assign(
            **{"__date_day": lambda x: x[self.column_names.start_date].dt.date}
        )

        aggregations = {feature: "mean" for feature in self.features}
        aggregations["__date_day"] = "first"
        grouped = (
            concat_df.groupby([*self.granularity, self.column_names.update_match_id])
            .agg(aggregations)
            .reset_index()
        )
        count_feats = []
        for day in self.days:
            prefix_day = f"{self.prefix}{day}"
            if self.add_count:
                count_feats.append(f"{prefix_day}_count")
            grouped = self._add_rolling_feature(
                concat_df=grouped,
                day=day,
                granularity=self.granularity,
                prefix_day=prefix_day,
            )

        feats_created = [f for f in self.features_out if f in grouped.columns]
        concat_df = concat_df.merge(
            grouped[
                [*self.granularity, *feats_created, self.column_names.update_match_id]
            ],
            on=self.granularity + [self.column_names.update_match_id],
            how="left",
        )
        concat_df = concat_df.sort_values(
            by=[
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
        )
        return concat_df.drop(columns="__date_day")

    def _add_rolling_feature(
        self, concat_df: pd.DataFrame, day: int, granularity: list[str], prefix_day: str
    ):

        if len(granularity) > 1:
            granularity_concat = "__".join(granularity)
            temporary_str_df = concat_df[granularity].astype(str)
            concat_df[granularity_concat] = temporary_str_df.agg("__".join, axis=1)
        else:
            granularity_concat = granularity[0]

        concat_df["is_nan"] = concat_df[self.features[0]].isna().astype(float)

        df1 = (
            concat_df.groupby(["__date_day", granularity_concat])[
                [*self.features, "is_nan"]
            ]
            .agg(["sum", "size"])
            .unstack()
            .asfreq("d", fill_value=np.nan)
            .rolling(window=day, min_periods=1)
            .sum()
            .shift()
            .stack()
        )
        feats = []
        for feature_name in self.features:
            feats.append(f"{prefix_day}_{feature_name}_sum")
            feats.append(f"{prefix_day}_{feature_name}_count")

        df1.columns = feats + ["is_nan_sum", "is_nan_count"]
        for feature_name in self.features:
            df1[f"{prefix_day}_{feature_name}_count"] = (
                df1[f"{prefix_day}_{feature_name}_count"] - df1["is_nan_sum"]
            )
            df1[f"{prefix_day}_{feature_name}"] = df1[
                f"{prefix_day}_{feature_name}_sum"
            ] / (df1[f"{prefix_day}_{feature_name}_count"])

            df1.loc[
                df1[f"{prefix_day}_{feature_name}_count"] == 0,
                f"{prefix_day}_{feature_name}",
            ] = np.nan

        if self.add_count:
            df1[f"{prefix_day}_count"] = df1[f"{prefix_day}_{self.features[0]}_count"]
            df1 = df1.drop(
                columns=[
                    f"{prefix_day}_{feature_name}_count"
                    for feature_name in self.features
                ]
            )

        concat_df["__date_day"] = pd.to_datetime(concat_df["__date_day"])
        concat_df = concat_df.merge(
            df1[[c for c in df1.columns if c in self.features_out]].reset_index(),
            on=["__date_day", granularity_concat],
        )

        return concat_df

    def reset(self):
        self._df = None
        self._fitted_game_ids = []

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
    def generate_historical(self, df: FrameT, column_names: ColumnNames) -> IntoFrameT:

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
        if ori_native == "pd":
            return transformed_df.to_pandas()
        return transformed_df

    @nw.narwhalify
    def generate_future(self, df: FrameT) -> IntoFrameT:

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
            df.select(
                nw.col(self.column_names.match_id)
            )
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

        concat_df = self._concat_df(df)

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
                    .rolling_mean(window_size=self.window, min_periods=self.min_periods)
                    .over(self.granularity)
                    .alias(f"{self.prefix}_{feature}{self.window}_1"),
                    nw.col("value_result_0_shifted")
                    .rolling_mean(window_size=self.window, min_periods=self.min_periods)
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


class RollingMeanTransformer(BaseLagGenerator):
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
        granularity: Union[list[str], str] = None,
        add_opponent: bool = False,
        scale_by_participation_weight: bool = False,
        min_periods: int = 1,
        are_estimator_features=True,
        prefix: str = "rolling_mean",
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
        """

        super().__init__(
            features=features,
            add_opponent=add_opponent,
            iterations=[window],
            prefix=prefix,
            granularity=granularity,
            are_estimator_features=are_estimator_features,
        )
        self.scale_by_participation_weight = scale_by_participation_weight
        self.window = window
        self.min_periods = min_periods

    @nw.narwhalify
    def generate_historical(self, df: FrameT, column_names: ColumnNames) -> IntoFrameT:
        """
        Generates rolling mean for historical data
        Stored the historical data as instance-variables so it's possible to generate future data afterwards
        The calculation is done using Polars.
         However, Pandas Dataframe can be used as input and it will also output a pandas dataframe in that case.

        :param df: Historical data
        :param column_names: Column names
        """

        if isinstance(nw.to_native(df), pd.DataFrame):
            ori_type = "pd"
            df = nw.from_native(pl.DataFrame(nw.to_native(df)))
        else:
            ori_type = "pl"

        unique_cols = (
            [column_names.player_id, column_names.team_id, column_names.match_id]
            if column_names.player_id in df.columns
            else [column_names.team_id, column_names.match_id]
        )
        if df.unique(subset=unique_cols).shape[0] != df.shape[0]:
            raise ValueError(
                f"Duplicated rows in df. Df must be a unique combination of {column_names.player_id} and {column_names.update_match_id}"
            )

        df = df.with_columns(nw.lit(0).alias("is_future"))

        self.column_names = column_names
        self.granularity = self.granularity or [self.column_names.player_id]
        validate_sorting(df=df, column_names=self.column_names)
        self._store_df(df)
        concat_df = self._generate_concat_df_with_feats(df)
        transformed_df = self._create_transformed_df(df=df, concat_df=concat_df)
        cn = self.column_names

        df = df.join(
            transformed_df.select(*unique_cols, *self.features_out),
            on=unique_cols,
            how="left",
        )
        if "is_future" in df.columns:
            df = df.drop("is_future")

        if ori_type == "pd":
            return df.to_pandas()
        return df.to_native()

    @nw.narwhalify
    def generate_future(self, df: FrameT) -> IntoFrameT:
        """
        Generates rolling mean for future data
        Assumes that .generate_historical() has been called before
        The calculation is done using Polars.
         However, Pandas Dataframe can be used as input and it will also output a pandas dataframe in that case.

        Regardless of the scheduled data of the future match, all future matches are perceived as being played in the same date.
        That is to ensure that a team's 2nd future match has the same rolling means as the 1st future match.
        :param df: Future data
        """

        if isinstance(nw.to_native(df), pd.DataFrame):
            ori_type = "pd"
            df = nw.from_native(pl.DataFrame(nw.to_native(df)))
        else:
            ori_type = "pl"

        df = df.with_columns(nw.lit(1).alias("is_future"))
        concat_df = self._generate_concat_df_with_feats(df=df)
        unique_match_ids = df.select(nw.col(self.column_names.match_id).unique())[
            self.column_names.match_id
        ].to_list()
        transformed_df = concat_df.filter(
            nw.col(self.column_names.match_id).is_in(unique_match_ids)
        )
        transformed_df = self._generate_future_feats(
            transformed_df=transformed_df, ori_df=df
        )
        cn = self.column_names

        df = df.join(
            transformed_df.select(
                cn.player_id, cn.team_id, cn.match_id, *self.features_out
            ),
            on=[cn.player_id, cn.team_id, cn.match_id],
            how="left",
        )
        if "is_future" in df.columns:
            df = df.drop("is_future")

        if ori_type == "pd":
            return df.to_pandas()

        return df.to_native()

    def _generate_concat_df_with_feats(self, df: FrameT) -> FrameT:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        concat_df = self._concat_df(df)
        if (
            self.column_names.participation_weight
            and self.scale_by_participation_weight
        ):
            concat_df = concat_df.with_columns(
                [
                    (
                        concat_df[feature_name]
                        * concat_df[self.column_names.participation_weight]
                    ).alias(feature_name)
                    for feature_name in self.features
                ]
            )

        grp = concat_df.group_by(
            self.granularity
            + [self.column_names.update_match_id, self.column_names.start_date]
        ).agg([nw.col(feature).mean().alias(feature) for feature in self.features])

        grp = grp.filter(~nw.col(self.column_names.start_date).is_null())
        grp = grp.sort(
            [self.column_names.start_date, self.column_names.update_match_id]
        )

        rolling_means = [
            nw.col(feature_name)
            .shift(n=1)
            .rolling_mean(window_size=self.window, min_periods=self.min_periods)
            .over(self.granularity)
            .alias(f"{self.prefix}_{feature_name}{self.window}")
            for feature_name in self.features
        ]
        grp = grp.with_columns(rolling_means)

        selection_columns = (
            self.granularity
            + [self.column_names.update_match_id]
            + [f"{self.prefix}_{feature}{self.window}" for feature in self.features]
        )
        concat_df = concat_df.join(
            grp.select(selection_columns),
            on=self.granularity + [self.column_names.update_match_id],
            how="left",
        )

        sort_cols = (
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id in concat_df.columns
            else [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )
        concat_df = concat_df.sort(sort_cols)

        feats_added = [f for f in self.features_out if f in concat_df.columns]

        concat_df = concat_df.with_columns(
            [
                nw.col(f).fill_null(strategy="forward").over(self.granularity).alias(f)
                for f in feats_added
            ]
        )
        return concat_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out
