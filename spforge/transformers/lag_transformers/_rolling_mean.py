from typing import Union, Optional
import narwhals as nw

from narwhals.typing import FrameT, IntoFrameT

from spforge import ColumnNames
from spforge.transformers.lag_transformers._utils import (
    required_lag_column_names,
    historical_lag_transformations_wrapper,
    transformation_validator,
    future_validator,
    future_lag_transformations_wrapper,
)
from spforge.transformers.lag_transformers import BaseLagTransformer


class RollingMeanTransformer(BaseLagTransformer):
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
        granularity: Union[list[str], str],
        add_opponent: bool = False,
        scale_by_participation_weight: bool = False,
        min_periods: int = 1,
        are_estimator_features=True,
        prefix: str = "rolling_mean",
        match_id_update_column: Optional[str] = None,
        unique_constraint: Optional[list[str]] = None,
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
            match_id_update_column=match_id_update_column,
            unique_constraint=unique_constraint,
        )
        self.scale_by_participation_weight = scale_by_participation_weight
        self.window = window
        self.min_periods = min_periods

    @nw.narwhalify
    @historical_lag_transformations_wrapper
    @required_lag_column_names
    @transformation_validator
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        """
        Generates rolling mean for historical data
        Stored the historical data as instance-variables so it's possible to generate future data afterwards

        :param df: Historical data
        :param column_names: Column names
        """

        if self.column_names:
            self._store_df(df)
            df_with_feats = self._generate_features(df)
            df = self._merge_into_input_df(
                df=df,
                concat_df=df_with_feats,
                match_id_join_on=self.match_id_update_column,
            )

        else:
            df_with_feats = self._generate_features(df).sort("__row_index")
            df = df.join(
                df_with_feats.select(
                    [*self.granularity, self.match_id_update_column, *self.features_out]
                ),
                on=[*self.granularity, self.match_id_update_column],
                how="left",
            ).unique("__row_index")
        if self.add_opponent:
            return self._add_opponent_features(df).sort("__row_index")
        return df

    @nw.narwhalify
    @future_lag_transformations_wrapper
    @future_validator
    @transformation_validator
    def transform_future(self, df: FrameT) -> IntoFrameT:
        """
        Generates rolling mean for future data
        Assumes that .generate_historical() has been called before
        Regardless of the scheduled data of the future match, all future matches are perceived as being played in the same date.
        That is to ensure that a team's 2nd future match has the same rolling means as the 1st future match.
        :param df: Future data
        """

        df_with_feats = self._generate_features(df=df)
        df_with_feats = self._merge_into_input_df(df=df, concat_df=df_with_feats)
        if self.add_opponent:
            df_with_feats = self._add_opponent_features(df_with_feats).sort(
                "__row_index"
            )

        df_with_feats = self._forward_fill_future_features(df=df_with_feats)

        return df_with_feats

    def _generate_features(self, df: FrameT) -> FrameT:

        if self.column_names and self._df is not None:
            sort_col = self.column_names.start_date
            concat_df = self._concat_with_stored(df)

        else:
            concat_df = df
            if "__row_index" not in concat_df.columns:
                concat_df = concat_df.with_row_index(name="__row_index")
            sort_col = "__row_index"

        aggr_cols = (
            [*self.features] + [self.column_names.participation_weight]
            if self.scale_by_participation_weight
            else self.features
        )
        grp = (
            concat_df.group_by(self.granularity + [self.match_id_update_column]).agg(
                [nw.col(feature).mean().alias(feature) for feature in [*aggr_cols]]
                + [nw.col(sort_col).min()]
            )
        ).sort(sort_col)

        if self.scale_by_participation_weight:

            grp = grp.with_columns(
                (
                    nw.col(feature) * nw.col(self.column_names.participation_weight)
                ).alias(f"__scaled_{feature}")
                for feature in self.features
            )
            scaled_feats = [f"__scaled_{feature}" for feature in self.features]

            rolling_sums = [
                nw.col(feature_name)
                .shift(n=1)
                .rolling_sum(window_size=self.window, min_samples=self.min_periods)
                .over(self.granularity)
                .alias(f"{self.prefix}_{feature_name}{self.window}__sum")
                for feature_name in [
                    *scaled_feats,
                    self.column_names.participation_weight,
                ]
            ]
            grp = grp.with_columns(rolling_sums)
            rolling_means = [
                (
                    nw.col(f"{self.prefix}___scaled_{feature}{self.window}__sum")
                    / nw.col(
                        f"{self.prefix}_{self.column_names.participation_weight}{self.window}__sum"
                    )
                ).alias(f"{self.prefix}_{feature}{self.window}")
                for feature in self.features
            ]
            grp = grp.with_columns(rolling_means)

        else:

            rolling_means = [
                nw.col(feature_name)
                .shift(n=1)
                .rolling_mean(window_size=self.window, min_samples=self.min_periods)
                .over(self.granularity)
                .alias(f"{self.prefix}_{feature_name}{self.window}")
                for feature_name in self.features
            ]
            grp = grp.with_columns(rolling_means)

        return grp.with_columns(
            [
                nw.col(f).fill_null(strategy="forward").over(self.granularity).alias(f)
                for f in self._entity_features_out
            ]
        )
