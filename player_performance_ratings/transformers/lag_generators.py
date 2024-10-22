from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformers.base_transformer import (
    BaseLagGenerator,
    BaseLagGeneratorPolars,
)
from player_performance_ratings.utils import validate_sorting


def create_output_column_by_game_group(
        data: pd.DataFrame,
        feature_name: str,
        weight_column: str,
        game_id: str,
        granularity: list[str],
) -> pd.DataFrame:
    if weight_column:
        data = data.assign(
            **{f"weighted_{feature_name}": data[feature_name] * data[weight_column]}
        )
        data = data.assign(
            **{
                "sum_weighted": data.groupby(granularity + [game_id])[
                    weight_column
                ].transform("sum")
            }
        )
        data = data.assign(
            **{feature_name: data[f"weighted_{feature_name}"] / data["sum_weighted"]}
        )
        data = data.groupby(granularity + [game_id])[feature_name].sum().reset_index()

    else:
        data = data.groupby(granularity + [game_id])[feature_name].mean().reset_index()
    return data


class LagTransformer(BaseLagGenerator):

    def __init__(
            self,
            features: list[str],
            lag_length: int,
            granularity: Optional[list[str]] = None,
            days_between_lags: Optional[list[int]] = None,
            scale_by_participation_weight: bool = False,
            future_lag: bool = False,
            prefix: str = "lag_",
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

    def generate_historical(
            self, df: pd.DataFrame, column_names: ColumnNames
    ) -> pd.DataFrame:
        """ """
        df = df.assign(is_future=0)
        self.column_names = column_names
        self.granularity = self.granularity or [self.column_names.player_id]
        validate_sorting(df=df, column_names=self.column_names)
        self._store_df(df)
        concat_df = self._generate_concat_df_with_feats(df)
        df = self._create_transformed_df(df=df, concat_df=concat_df)
        if "is_future" in df.columns:
            df = df.drop(columns="is_future")
        return df

    def generate_future(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return transformed_future

    def _generate_concat_df_with_feats(self, df: pd.DataFrame) -> pd.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        if len(
                df.drop_duplicates(
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

        if self.column_names.participation_weight and self.scale_by_participation_weight:
            for feature in self.features:
                concat_df = concat_df.assign(
                    **{
                        feature: concat_df[feature]
                                 * concat_df[self.column_names.participation_weight]
                    }
                )

        grouped = (
            concat_df.groupby(
                self.granularity
                + [self.column_names.update_match_id, self.column_names.start_date]
            )[self.features]
            .mean()
            .reset_index()
        )

        grouped = grouped.sort_values(
            [self.column_names.start_date, self.column_names.update_match_id]
        )

        for days_lag in self.days_between_lags:
            if self.future_lag:
                grouped["shifted_days"] = grouped.groupby(self.granularity)[
                    self.column_names.start_date
                ].shift(-days_lag)
                grouped[f"{self.prefix}{days_lag}_days_ago"] = (
                        pd.to_datetime(grouped["shifted_days"])
                        - pd.to_datetime(grouped[self.column_names.start_date])
                ).dt.days
            else:
                grouped["shifted_days"] = grouped.groupby(self.granularity)[
                    self.column_names.start_date
                ].shift(days_lag)
                grouped[f"{self.prefix}{days_lag}_days_ago"] = (
                        pd.to_datetime(grouped[self.column_names.start_date])
                        - pd.to_datetime(grouped["shifted_days"])
                ).dt.days

            grouped = grouped.drop(columns=["shifted_days"])

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f"{self.prefix}{lag}_{feature_name}"
                if output_column_name in concat_df.columns:
                    raise ValueError(
                        f"Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed"
                    )

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f"{self.prefix}{lag}_{feature_name}"
                if self.future_lag:
                    grouped = grouped.assign(
                        **{
                            output_column_name: grouped.groupby(self.granularity)[
                                feature_name
                            ].shift(-lag)
                        }
                    )
                else:
                    grouped = grouped.assign(
                        **{
                            output_column_name: grouped.groupby(self.granularity)[
                                feature_name
                            ].shift(lag)
                        }
                    )

        feats_out = []
        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                feats_out.append(f"{self.prefix}{lag}_{feature_name}")

        for days_lag in self.days_between_lags:
            feats_out.append(f"{self.prefix}{days_lag}_days_ago")

        concat_df = concat_df.merge(
            grouped[
                self.granularity
                + [
                    self.column_names.update_match_id,
                    self.column_names.start_date,
                    *feats_out,
                ]
                ],
            on=self.granularity
               + [self.column_names.update_match_id, self.column_names.start_date],
            how="left",
        )

        return concat_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out


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
        min_periods: int = 1,
        granularity: Union[list[str], str] = None,
        scale_by_participation_weight: bool = False,
        add_opponent: bool = False,
        are_estimator_features=True,
        prefix: str = "rolling_mean_",
    ):
        """
        Calculates the rolling mean for a list of features over a window of matches.
        Rolling Mean Values are also shifted by one match to avoid data leakage.

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

    def generate_historical(
            self, df: pd.DataFrame, column_names: ColumnNames
    ) -> pd.DataFrame:
        """
        Generates rolling mean for historical data
        Stored the historical data as instance-variables so it's possible to generate future data afterwards
        The calculation is done using Polars.
         However, Pandas Dataframe can be used as input and it will also output a pandas dataframe in that case.

        :param df: Historical data
        :param column_names: Column names
        """
        df = df.assign(is_future=0)
        self.column_names = column_names
        self.granularity = self.granularity or [self.column_names.player_id]
        validate_sorting(df=df, column_names=self.column_names)
        self._store_df(df)
        concat_df = self._generate_concat_df_with_feats(df)
        df = self._create_transformed_df(df=df, concat_df=concat_df)
        if "is_future" in df.columns:
            df = df.drop(columns="is_future")
        return df

    def generate_future(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates rolling mean for future data
        Assumes that .generate_historical() has been called before

        Regardless of the scheduled data of the future match, all future matches are perceived as being played in the same date.
        That is to ensure that a team's 2nd future match has the same rolling means as the 1st future match.
        :param df: Future data
        """

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
        return transformed_future

    def _generate_concat_df_with_feats(self, df: pd.DataFrame) -> pd.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        concat_df = self._concat_df(df)

        for feature_name in self.features:

            if self.column_names.participation_weight and self.scale_by_participation_weight:
                concat_df = concat_df.assign(
                    **{
                        feature_name: concat_df[feature_name]
                                      * concat_df[self.column_names.participation_weight]
                    }
                )

            output_column_name = f"{self.prefix}{self.window}_{feature_name}"
            if output_column_name in concat_df.columns:
                raise ValueError(
                    f"Column {output_column_name} already exists. Choose different prefix or ensure no duplication was performed"
                )

            agg_dict = {feature_name: "mean", self.column_names.start_date: "first"}
            grp = (
                concat_df.groupby(
                    self.granularity + [self.column_names.update_match_id]
                )
                .agg(agg_dict)
                .reset_index()
            )
            grp.sort_values(
                by=[self.column_names.start_date, self.column_names.update_match_id],
                inplace=True,
            )

            grp = grp.assign(
                **{
                    output_column_name: grp.groupby(self.granularity)[
                        feature_name
                    ].transform(
                        lambda x: x.shift()
                        .rolling(self.window, min_periods=self.min_periods)
                        .mean()
                    )
                }
            )

            concat_df = concat_df.merge(
                grp[
                    self.granularity
                    + [self.column_names.update_match_id, output_column_name]
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

        feats_added = [f for f in self.features_out if f in concat_df.columns]

        concat_df[feats_added] = concat_df.groupby(self.granularity)[
            feats_added
        ].fillna(method="ffill")
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

            if self.column_names.participation_weight and self.scale_by_participation_weight:
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

class BinaryOutcomeRollingMeanTransformerPolars(BaseLagGeneratorPolars):
    # IN PROGRESS NOT WORKING
    def __init__(
            self,
            features: list[str],
            window: int,
            binary_column: str,
            granularity: list[str] = None,
            prob_column: Optional[str] = None,
            min_periods: int = 1,
            add_opponent: bool = False,
            prefix: str = "rolling_mean_binary_",
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
            feature1 = f"{self.prefix}{self.window}_{feature_name}_1"
            feature2 = f"{self.prefix}{self.window}_{feature_name}_0"
            self._features_out.append(feature1)
            self._features_out.append(feature2)
            self._entity_features.append(feature1)
            self._entity_features.append(feature2)

            if self.add_opponent:
                self._features_out.append(
                    f"{self.prefix}{self.window}_{feature_name}_1_opponent"
                )
                self._features_out.append(
                    f"{self.prefix}{self.window}_{feature_name}_0_opponent"
                )

        if self.prob_column:
            for feature_name in self.features:
                prob_feature = (
                    f"{self.prefix}{self.window}_{self.prob_column}_{feature_name}"
                )
                self._features_out.append(prob_feature)

        self._estimator_features_out = self._features_out.copy()

    def generate_historical(
            self, df: pl.DataFrame, column_names: ColumnNames
    ) -> pl.DataFrame:
        if df.schema[self.binary_column] in [pl.Float64, pl.Float32]:
            df = df.with_columns(
                pl.col(self.binary_column).cast(pl.Int64)
            )

        df = df.with_columns(pl.lit(0).alias("is_future"))
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
        return self._add_weighted_prob(transformed_df=transformed_df)

    def generate_future(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._df is None:
            raise ValueError(
                "generate_historical needs to be called before generate_future"
            )

        if self.binary_column in df.columns:
            if df.schema[self.binary_column] in [pl.Float64, pl.Float32]:
                df = df.with_columns(
                    pl.col(self.binary_column).cast(pl.Int64)
                )
        df = df.with_columns(pl.lit(1).alias("is_future"))
        concat_df = self._generate_concat_df_with_feats(df=df)
        unique_match_ids = df.select(pl.col(self.column_names.match_id).cast(pl.Utf8)).unique()
        transformed_df = concat_df.filter(
            pl.col(self.column_names.match_id).is_in(unique_match_ids[self.column_names.match_id])
        )
        transformed_future = self._generate_future_feats(
            transformed_df=transformed_df,
            ori_df=df,
            known_future_features=self._get_known_future_features(),
        )
        if "is_future" in transformed_future.columns:
            transformed_future = transformed_future.drop("is_future")
        return self._add_weighted_prob(transformed_df=transformed_future)

    def _get_known_future_features(self) -> list[str]:
        known_future_features = []
        if self.prob_column:
            for idx, feature_name in enumerate(self.features):
                weighted_prob_feat_name = (
                    f"{self.prefix}{self.window}_{self.prob_column}_{feature_name}"
                )
                known_future_features.append(weighted_prob_feat_name)

        return known_future_features

    def _add_weighted_prob(self, transformed_df: pl.DataFrame) -> pl.DataFrame:

        if self.prob_column:
            for idx, feature_name in enumerate(self.features):
                weighted_prob_feat_name = (
                    f"{self.prefix}{self.window}_{self.prob_column}_{feature_name}"
                )
                transformed_df = transformed_df.with_columns(
                    (
                            pl.col(f"{self.prefix}{self.window}_{feature_name}_1")
                            * pl.col(self.prob_column)
                            + pl.col(f"{self.prefix}{self.window}_{feature_name}_0")
                            * (1 - pl.col(self.prob_column))
                    ).alias(weighted_prob_feat_name)
                )
        return transformed_df

    def _generate_concat_df_with_feats(self, df: pl.DataFrame) -> pl.DataFrame:
        additional_cols_to_use = [self.binary_column] + (
            [self.prob_column] if self.prob_column else []
        )
        concat_df = self._concat_df(df, additional_cols_to_use=additional_cols_to_use)

        # Define groupby columns
        groupby_cols = [
            self.column_names.update_match_id,
            *self.granularity,
            self.column_names.start_date,
            "is_future",
        ]

        # Perform initial aggregation
        aggregation = {f: pl.mean(f).alias(f) for f in self.features}
        aggregation[self.binary_column] = pl.first(self.binary_column)
        grouped = concat_df.group_by(groupby_cols).agg(list(aggregation.values()))

        # Sort the DataFrame
        grouped = grouped.sort(
            by=[
                self.column_names.start_date,
                "is_future",
                self.column_names.update_match_id,
            ]
        )

        # Initialize a list to keep track of new features added
        feats_added = []

        for feature in self.features:
            # Define shifted feature and binary columns
            shifted_feature = pl.col(feature).shift(1).over(self.granularity)
            shifted_binary = pl.col(self.binary_column).shift(1).over(self.granularity)

            # Calculate rolling means conditioned on binary outcome
            rolling_mean_1 = (
                pl.when(shifted_binary == 1)
                .then(shifted_feature)
                .rolling_mean(
                    window_size=self.window, min_periods=self.min_periods
                )
                .over(self.granularity)
                .alias(f"{self.prefix}{self.window}_{feature}_1")
            )

            rolling_mean_0 = (
                pl.when(shifted_binary == 0)
                .then(shifted_feature)
                .rolling_mean(
                    window_size=self.window, min_periods=self.min_periods
                )
                .over(self.granularity)
                .alias(f"{self.prefix}{self.window}_{feature}_0")
            )

            # Add the new rolling mean columns to the DataFrame
            grouped = grouped.with_columns([rolling_mean_1, rolling_mean_0])

            # Keep track of the features added
            feats_added.extend([
                f"{self.prefix}{self.window}_{feature}_1",
                f"{self.prefix}{self.window}_{feature}_0",
            ])

        # Merge the rolling mean features back into concat_df
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

        # Forward fill missing values within granularity groups
        concat_df = concat_df.with_columns([
            pl.col(feats_added).fill_null(strategy="forward").over(self.granularity)
        ])

        return concat_df

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
            prefix: str = "rolling_mean_binary_",
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
            feature1 = f"{self.prefix}{self.window}_{feature_name}_1"
            feature2 = f"{self.prefix}{self.window}_{feature_name}_0"
            self._features_out.append(f"{self.prefix}{self.window}_{feature_name}_1")
            self._features_out.append(f"{self.prefix}{self.window}_{feature_name}_0")
            self._entity_features.append(feature1)
            self._entity_features.append(feature2)

            if self.add_opponent:
                self._features_out.append(
                    f"{self.prefix}{self.window}_{feature_name}_1_opponent"
                )
                self._features_out.append(
                    f"{self.prefix}{self.window}_{feature_name}_0_opponent"
                )

        if self.prob_column:
            for feature_name in self.features:
                prob_feature = (
                    f"{self.prefix}{self.window}_{self.prob_column}_{feature_name}"
                )
                self._features_out.append(prob_feature)

        self._estimator_features_out = self._features_out.copy()

    def generate_historical(
            self, df: pd.DataFrame, column_names: ColumnNames
    ) -> pd.DataFrame:
        if df[self.binary_column].dtype in ("float64", "float32", "float16", "float"):
            df = df.assign(**{self.binary_column: df[self.binary_column].astype(int)})

        df = df.assign(is_future=0)
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
            transformed_df = transformed_df.drop(columns="is_future")
        return self._add_weighted_prob(transformed_df=transformed_df)

    def generate_future(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._df is None:
            raise ValueError(
                "generate_historical needs to be called before generate_future"
            )

        if self.binary_column in df.columns:
            if df[self.binary_column].dtype in (
                    "float64",
                    "float32",
                    "float16",
                    "float",
            ):
                df = df.assign(
                    **{self.binary_column: df[self.binary_column].astype(int)}
                )
        df = df.assign(is_future=1)
        concat_df = self._generate_concat_df_with_feats(df=df)
        transformed_df = concat_df[
            concat_df[self.column_names.match_id].isin(
                df[self.column_names.match_id].astype("str").unique().tolist()
            )
        ]
        transformed_future = self._generate_future_feats(
            transformed_df=transformed_df,
            ori_df=df,
            known_future_features=self._get_known_future_features(),
        )
        if "is_future" in transformed_future.columns:
            transformed_future = transformed_future.drop(columns="is_future")
        return self._add_weighted_prob(transformed_df=transformed_future)

    def _get_known_future_features(self) -> list[str]:
        known_future_features = []
        if self.prob_column:
            for idx, feature_name in enumerate(self.features):
                weighted_prob_feat_name = (
                    f"{self.prefix}{self.window}_{self.prob_column}_{feature_name}"
                )
                known_future_features.append(weighted_prob_feat_name)

        return known_future_features

    def _add_weighted_prob(self, transformed_df: pd.DataFrame) -> pd.DataFrame:

        if self.prob_column:
            for idx, feature_name in enumerate(self.features):
                weighted_prob_feat_name = (
                    f"{self.prefix}{self.window}_{self.prob_column}_{feature_name}"
                )
                transformed_df[weighted_prob_feat_name] = transformed_df[
                                                              f"{self.prefix}{self.window}_{feature_name}_1"
                                                          ] * transformed_df[self.prob_column] + transformed_df[
                                                              f"{self.prefix}{self.window}_{feature_name}_0"
                                                          ] * (
                                                                  1 - transformed_df[self.prob_column]
                                                          )
        return transformed_df

    def _generate_concat_df_with_feats(self, df: pd.DataFrame) -> pd.DataFrame:

        additional_cols_to_use = [self.binary_column] + (
            [self.prob_column] if self.prob_column else []
        )
        concat_df = self._concat_df(df, additional_cols_to_use=additional_cols_to_use)
        aggregation = {
            **{f: "mean" for f in self.features},
            **{self.binary_column: "first"},
        }
        grouped = (
            concat_df.groupby(
                [
                    self.column_names.update_match_id,
                    *self.granularity,
                    self.column_names.start_date,
                    "is_future",
                ]
            )
            .agg(aggregation)
            .reset_index()
        )

        grouped = grouped.sort_values(
            by=[
                self.column_names.start_date,
                "is_future",
                self.column_names.update_match_id,
            ]
        )

        feats_added = []
        for feature in self.features:
            mask_result_1 = grouped[self.binary_column] == 1
            mask_result_0 = grouped[self.binary_column] == 0

            grouped["value_result_1"] = grouped[feature].where(mask_result_1)
            grouped["value_result_0"] = grouped[feature].where(mask_result_0)

            grouped[f"{self.prefix}{self.window}_{feature}_1"] = grouped.groupby(
                self.granularity
            )["value_result_1"].transform(
                lambda x: x.shift().rolling(window=self.window, min_periods=1).mean()
            )
            grouped[f"{self.prefix}{self.window}_{feature}_0"] = grouped.groupby(
                self.granularity
            )["value_result_0"].transform(
                lambda x: x.shift().rolling(window=self.window, min_periods=1).mean()
            )

            feats_added.append(f"{self.prefix}{self.window}_{feature}_1")
            feats_added.append(f"{self.prefix}{self.window}_{feature}_0")

        grouped["count_result_1"] = (
                grouped[grouped[self.binary_column] == 1]
                .groupby([*self.granularity])
                .cumcount()
                + 1
        )
        grouped["count_result_0"] = (
                grouped[grouped[self.binary_column] == 0]
                .groupby([*self.granularity])
                .cumcount()
                + 1
        )
        grouped[["count_result_0", "count_result_1"]] = grouped.groupby(
            self.granularity
        )[["count_result_0", "count_result_1"]].fillna(method="ffill")
        for feature in self.features:
            grouped.loc[
                grouped["count_result_1"] < self.min_periods,
                f"{self.prefix}{self.window}_{feature}_1",
            ] = np.nan
            grouped.loc[
                grouped["count_result_0"] < self.min_periods,
                f"{self.prefix}{self.window}_{feature}_0",
            ] = np.nan

        concat_df = concat_df.merge(
            grouped[
                [self.column_names.update_match_id, *self.granularity, *feats_added]
            ],
            on=[self.column_names.update_match_id, *self.granularity],
            how="left",
        )

        concat_df[feats_added] = concat_df.groupby(self.granularity)[
            feats_added
        ].fillna(method="ffill")
        return concat_df


class RollingMeanTransformerPolars(BaseLagGeneratorPolars):
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
            prefix: str = "rolling_mean_",
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

    def generate_historical(
            self, df: Union[pd.DataFrame, pl.DataFrame], column_names: ColumnNames
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Generates rolling mean for historical data
        Stored the historical data as instance-variables so it's possible to generate future data afterwards
        The calculation is done using Polars.
         However, Pandas Dataframe can be used as input and it will also output a pandas dataframe in that case.

        :param df: Historical data
        :param column_names: Column names
        """
        if isinstance(df, pd.DataFrame):
            ori_type = "pd"
            df = pl.DataFrame(df)
        else:
            ori_type = "pl"
        if (
                df.unique(
                    subset=[
                        column_names.player_id,
                        column_names.team_id,
                        column_names.match_id,
                    ]
                ).shape[0]
                != df.shape[0]
        ):
            raise ValueError(
                f"Duplicated rows in df. Df must be a unique combination of {column_names.player_id} and {column_names.update_match_id}"
            )

        df = df.with_columns(pl.lit(0).alias("is_future"))

        self.column_names = column_names
        self.granularity = self.granularity or [self.column_names.player_id]
        self._store_df(df)
        concat_df = self._generate_concat_df_with_feats(df)
        transformed_df = self._create_transformed_df(df=df, concat_df=concat_df)
        cn = self.column_names
        recasts_mapping = {}
        for c in [cn.player_id, cn.team_id, cn.match_id]:
            if transformed_df[c].dtype != df[c].dtype:
                recasts_mapping[c] = df[c].dtype
        transformed_df = transformed_df.cast(recasts_mapping)

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
        return df

    def generate_future(
            self, df: Union[pd.DataFrame, pl.DataFrame]
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Generates rolling mean for future data
        Assumes that .generate_historical() has been called before
        The calculation is done using Polars.
         However, Pandas Dataframe can be used as input and it will also output a pandas dataframe in that case.

        Regardless of the scheduled data of the future match, all future matches are perceived as being played in the same date.
        That is to ensure that a team's 2nd future match has the same rolling means as the 1st future match.
        :param df: Future data
        """

        if isinstance(df, pd.DataFrame):
            ori_type = "pd"
            df = pl.DataFrame(df)
        else:
            ori_type = "pl"

        df = df.with_columns(pl.lit(1).alias("is_future"))
        concat_df = self._generate_concat_df_with_feats(df=df)
        unique_match_ids = df.select(
            pl.col(self.column_names.match_id).cast(pl.Utf8).unique()
        ).to_series()
        transformed_df = concat_df.filter(
            pl.col(self.column_names.match_id).cast(pl.Utf8).is_in(unique_match_ids)
        )
        transformed_df = self._generate_future_feats(
            transformed_df=transformed_df, ori_df=df
        )
        cn = self.column_names
        recasts_mapping = {}
        for c in [cn.player_id, cn.team_id, cn.match_id]:
            if transformed_df[c].dtype != df[c].dtype:
                recasts_mapping[c] = df[c].dtype
        transformed_df = transformed_df.cast(recasts_mapping)

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
        return df

    def _generate_concat_df_with_feats(self, df: pl.DataFrame) -> pl.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        concat_df = self._concat_df(df)
        if self.column_names.participation_weight and self.scale_by_participation_weight:
            concat_df = concat_df.with_columns(
                [
                    (
                            concat_df[feature_name]
                            * concat_df[self.column_names.participation_weight]
                    ).alias(feature_name)
                    for feature_name in self.features
                ]
            )

        agg_dict = [
            pl.col(feature_name).mean().alias(feature_name)
            for feature_name in self.features
        ]
        agg_dict.append(
            pl.col(self.column_names.start_date)
            .first()
            .alias(self.column_names.start_date)
        )

        grp = concat_df.group_by(
            self.granularity + [self.column_names.update_match_id]
        ).agg(agg_dict)
        grp = grp.filter(pl.col(self.column_names.start_date).is_not_null())
        grp = grp.sort(
            [self.column_names.start_date, self.column_names.update_match_id]
        )

        rolling_means = [
            pl.col(feature_name)
            .shift()
            .rolling_mean(window_size=self.window, min_periods=self.min_periods)
            .over(self.granularity)
            .alias(f"{self.prefix}{self.window}_{feature_name}")
            for feature_name in self.features
        ]
        grp = grp.with_columns(rolling_means)

        selection_columns = (
                self.granularity
                + [self.column_names.update_match_id]
                + [f"{self.prefix}{self.window}_{feature}" for feature in self.features]
        )
        concat_df = concat_df.join(
            grp.select(selection_columns),
            on=self.granularity + [self.column_names.update_match_id],
            how="left",
        )

        concat_df = concat_df.sort(
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
        )

        feats_added = [f for f in self.features_out if f in concat_df.columns]

        concat_df = concat_df.with_columns(
            [
                pl.col(f).forward_fill().over(self.granularity).alias(f)
                for f in feats_added
            ]
        )
        return concat_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out
