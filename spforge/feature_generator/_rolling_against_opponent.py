from typing import Literal

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
from narwhals.typing import IntoFrameT

from spforge.data_structures import ColumnNames
from spforge.feature_generator._base import LagGenerator
from spforge.feature_generator._rolling_window import RollingWindowTransformer
from spforge.feature_generator._utils import (
    future_lag_transformations_wrapper,
    future_validator,
    historical_lag_transformations_wrapper,
    required_lag_column_names,
    transformation_validator,
)


class RollingAgainstOpponentTransformer(LagGenerator):
    """
    Performs a rolling calculation of the features from the entities perspective against the opponent.

    Unlike `RollingWindowTransformer` with `add_opponent=True`, the `RollingAgainstOpponentTransformer`
    does not compute a rolling mean for the entity itself. Instead, it aggregates how opponents
    perform *against* the specified granularity over a defined rolling window.

    This allows for creation of features that reflect opponent tendencies in various contexts.

    Example use case:
        - Do opponents tend to allow more points to centers than to players in other positions?

    Example:
        df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)
        transformer = RollingAgainstOpponentTransformer(
            features=["points"],
            window=15,
            granularity=["position"],
            opponent_column="opponent_team_id"
        )
        df = transformer.generate_historical(df)

    Use .fit_transform() to generate rolling mean for historical data.
        The historical data is stored as instance-variables after calling .generate_historical()
    Use .transform() to generate rolling mean for future data after having called .generate_historical()
    """

    def __init__(
        self,
        features: list[str],
        window: int,
        granularity: list[str] | str,
        min_periods: int = 1,
        are_estimator_features=True,
        prefix: str = "rolling_against_opponent",
        update_column: str | None = None,
        match_id_column: str | None = None,
        team_column: str | None = None,
        opponent_column: str = "__opponent",
        transformation: Literal["rolling_mean"] = "rolling_mean",
    ):
        """
        :param features:   Features to create rolling mean for
        :param window: Window size for rolling mean.
            If 10 will calculate rolling mean over the prior 10 observations
        :param min_periods: Minimum number of observations in window required to have a non-null result
        :param granularity: Columns to group by before rolling mean. E.g. player_id or [player_id, position].
             In the latter case it will get the rolling mean for each player_id and position combination.
             Defaults to player_id
        :param are_estimator_features: If True, the features will be added to the estimator features.
            If false, it makes it possible for the outer layer (Pipeline) to exclude these features from the estimator.
        :param prefix: Prefix for the new rolling mean columns
        :match_id_update_column: Column to join on when updating the match_id. Must be set if column_names is not passed.
        :opponent_column: Column name for the opponent team_id. Must be set if column_names is not passed.
        """

        super().__init__(
            features=features,
            add_opponent=False,
            iterations=[window],
            prefix=prefix,
            granularity=granularity,
            are_estimator_features=are_estimator_features,
            update_column=update_column,
            match_id_column=match_id_column,
        )
        self.window = window
        self.min_periods = min_periods
        self.team_column = team_column
        self.opponent_column = opponent_column
        self.transformation = transformation
        self._transformer: LagGenerator

    @nw.narwhalify
    @historical_lag_transformations_wrapper
    @required_lag_column_names
    @transformation_validator
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        """
        Generates rolling mean for historical data

        :param df: Historical data
        :param column_names: Column names
        """

        if self.column_names:
            self.update_column = self.column_names.update_match_id or self.update_column
            self.team_column = self.column_names.team_id or self.team_column
        else:
            assert (
                self.team_column is not None
            ), "team_column must be set if column names is not passed"
            assert (
                self.update_column is not None
            ), "match_id_update_column must be set if column names is not passed"
        if self.transformation == "rolling_mean":
            self._transformer = RollingWindowTransformer(
                granularity=[self.opponent_column, *self.granularity],
                features=self.features,
                window=self.window,
                min_periods=self.min_periods,
                update_column=self.update_column,
                match_id_column=self.match_id_column,
                group_to_granularity=[
                    self.opponent_column,
                    self.update_column,
                    *self.granularity,
                ],
            )
        else:
            raise NotImplementedError("Only rolling_mean transformation is supported")

        if self.column_names:
            self._store_df(df, ori_df=df)
            concat_df = self._concat_with_stored_and_calculate_feats(df, is_future=False)
            concat_df = self._rename_features(concat_df)
            return self._merge_into_input_df(
                df=df, concat_df=concat_df, match_id_join_on=self.update_column
            )

        else:
            concat_df = self._concat_with_stored_and_calculate_feats(df, is_future=False).sort(
                "__row_index"
            )
            concat_df = self._rename_features(concat_df)
            return (
                df.join(
                    concat_df.select(
                        [
                            *self.granularity,
                            self.team_column,
                            self.update_column,
                            *self.features_out,
                        ]
                    ),
                    on=[
                        *self.granularity,
                        self.update_column,
                        self.team_column,
                    ],
                    how="left",
                )
                .unique("__row_index")
                .sort("__row_index")
            )

    @nw.narwhalify
    @future_lag_transformations_wrapper
    @future_validator
    @transformation_validator
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        """
        Generates rolling mean opponent for future data
        Assumes that .generate_historical() has been called before
        Ensure that a team's 2nd future match has the same rolling means as the 1st future match.
        :param df: Future data
        """

        if isinstance(nw.to_native(df), pd.DataFrame):
            ori_type = "pd"
            df = nw.from_native(pl.DataFrame(nw.to_native(df)))
        else:
            ori_type = "pl"

        concat_df = self._concat_with_stored_and_calculate_feats(df=df, is_future=True)
        concat_df = self._rename_features(concat_df)

        unique_match_ids = df.select(nw.col(self.column_names.match_id).unique())[
            self.column_names.match_id
        ].to_list()
        transformed_df = concat_df.filter(
            nw.col(self.column_names.match_id).is_in(unique_match_ids)
        )
        transformed_df = self._forward_fill_future_features(df=transformed_df)

        cn = self.column_names

        df = df.join(
            transformed_df.select(cn.player_id, cn.team_id, cn.match_id, *self.features_out),
            on=[cn.player_id, cn.team_id, cn.match_id],
            how="left",
        )

        if ori_type == "pd":
            return df.to_pandas()

        return df.to_native()

    def _concat_with_stored_and_calculate_feats(
        self, df: IntoFrameT, is_future: bool
    ) -> IntoFrameT:

        cols_to_drop = [c for c in self.features_out if c in df.columns]
        df = df.drop(cols_to_drop)
        concat_df = df.clone()

        if self.opponent_column not in concat_df.columns:
            gt = concat_df.unique([self.update_column, self.team_column]).select(
                [self.update_column, self.team_column]
            )
            gt_opponent = gt.join(gt, on=self.update_column, how="left", suffix="__opp")
            gt_opponent = gt_opponent.filter(
                nw.col(self.team_column) != nw.col(f"{self.team_column}__opp")
            )
            gt_opponent = gt_opponent.with_columns(
                nw.col(f"{self.team_column}__opp").alias(self.opponent_column)
            ).drop(f"{self.team_column}__opp")
            concat_df = concat_df.join(
                gt_opponent,
                on=[self.update_column, self.team_column],
                how="left",
            ).sort("__row_index")

        if is_future:
            concat_df = nw.from_native(self._transformer.future_transform(concat_df))

        else:
            concat_df = nw.from_native(
                self._transformer.fit_transform(concat_df, column_names=self.column_names)
            )
        return concat_df

    def _merge_into_input_df(
        self, df: IntoFrameT, concat_df: IntoFrameT, match_id_join_on: str | None = None
    ) -> IntoFrameT:
        sort_cols = (
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id
            else [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )
        return (
            df.join(
                concat_df.select(
                    [
                        self.team_column,
                        self.update_column,
                        *self.features_out,
                        *self.granularity,
                    ]
                ),
                on=[self.update_column, self.team_column, *self.granularity],
                how="left",
            )
            .unique(self.unique_constraint)
            .sort(sort_cols)
        )

    def _rename_features(self, df: IntoFrameT) -> IntoFrameT:
        rename_cols = {
            feature: self.features_out[idx]
            for idx, feature in enumerate(self._transformer.features_out)
        }
        return df.rename(rename_cols)
