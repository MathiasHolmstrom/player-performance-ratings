import logging
from typing import Optional

import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from spforge import ColumnNames

from spforge.transformers.lag_transformers._utils import (
    required_lag_column_names,
    transformation_validator,
    historical_lag_transformations_wrapper,
    future_lag_transformations_wrapper,
    future_validator,
)
from spforge.transformers.lag_transformers import BaseLagTransformer


class LagTransformer(BaseLagTransformer):

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
    @historical_lag_transformations_wrapper
    @required_lag_column_names
    @transformation_validator
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        """ """

        if self.column_names:
            self._store_df(df)
            df = self._generate_features(df)
        else:
            df = self._generate_features(df)
        if self.add_opponent:
            return self._add_opponent_features(df).sort("__row_index")
        return df

    @nw.narwhalify
    @future_lag_transformations_wrapper
    @future_validator
    @transformation_validator
    def transform_future(self, df: FrameT) -> IntoFrameT:

        df_with_feats = self._generate_features(df=df)
        if self.add_opponent:
            df_with_feats = self._add_opponent_features(df_with_feats).sort(
                "__row_index"
            )

        df_with_feats = self._forward_fill_future_features(df=df_with_feats)

        return df_with_feats

    def _generate_features(self, df: FrameT) -> FrameT:
        grp_cols = self.granularity + [self.match_id_update_column]
        if self.column_names and self._df is not None:
            concat_df = self._concat_with_stored(df=df)
            sort_col = self.column_names.start_date

        else:
            concat_df = df
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

        return df.join(
            grouped.select(
                [
                    *self.granularity,
                    self.match_id_update_column,
                    *self._entity_features_out,
                ]
            ),
            on=[*self.granularity, self.match_id_update_column],
            how="left",
        ).sort("__row_index")

    @property
    def features_out(self) -> list[str]:
        return self._features_out
