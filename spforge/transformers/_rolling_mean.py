from typing import Union, Optional
import narwhals as nw
import pandas as pd
import polars as pl
from narwhals.typing import FrameT, IntoFrameT

from spforge import ColumnNames
from spforge.transformers.base_transformer import (
    BaseLagGenerator,
    required_lag_column_names,
    row_count_validator,
    future_validator,
)
from spforge.utils import validate_sorting


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
    @required_lag_column_names
    @row_count_validator
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        """
        Generates rolling mean for historical data
        Stored the historical data as instance-variables so it's possible to generate future data afterwards

        :param df: Historical data
        :param column_names: Column names
        """
        self.column_names = column_names or self.column_names
        input_cols = df.columns
        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            ori_native = "pd"
            df = nw.from_native(pl.DataFrame(native))
        else:
            ori_native = "pl"

        if self.column_names:
            df = df.with_columns(nw.lit(0).alias("is_future"))
            self._store_df(df)
            concat_df = self._concat_with_stored_and_calculate_feats(df)
            transformed_df = self._create_transformed_df(
                df=df, concat_df=concat_df, match_id_join_on=self.match_id_update_column
            )

            join_cols = (
                [
                    self.column_names.match_id,
                    self.column_names.player_id,
                    self.column_names.team_id,
                ]
                if self.column_names.player_id
                else [self.column_names.match_id, self.column_names.team_id]
            )
            df = df.join(
                transformed_df.select([*join_cols, *self.features_out]),
                on=join_cols,
                how="left",
            )
        else:
            transformed_df = self._concat_with_stored_and_calculate_feats(df).sort(
                "__row_index"
            )
            df = df.join(
                transformed_df.select(["__row_index", *self.features_out]),
                on="__row_index",
                how="left",
            )

        if "is_future" in df.columns:
            df = df.drop("is_future")
        if "__row_index" in df.columns:
            df = df.drop("__row_index")
            input_cols = [c for c in input_cols if c != "__row_index"]
        if ori_native == "pd":
            return df.select(list(set(input_cols + self.features_out))).to_pandas()
        return df.select(list(set(input_cols + self.features_out)))

    @nw.narwhalify
    @future_validator
    @row_count_validator
    def transform_future(self, df: FrameT) -> IntoFrameT:
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
        concat_df = self._concat_with_stored_and_calculate_feats(df=df)
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
        join_cols = self.unique_constraint or [cn.match_id, cn.player_id, cn.team_id]

        df = df.join(
            transformed_df.select(*join_cols, *self.features_out),
            on=join_cols,
            how="left",
        )
        if "is_future" in df.columns:
            df = df.drop("is_future")

        if ori_type == "pd":
            return df.to_pandas()

        return df.to_native()

    def _concat_with_stored_and_calculate_feats(self, df: FrameT) -> FrameT:

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

        selection_columns = (
            self.granularity
            + [self.match_id_update_column]
            + [f"{self.prefix}_{feature}{self.window}" for feature in self.features]
        )
        concat_df = concat_df.join(
            grp.select(selection_columns),
            on=self.granularity + [self.match_id_update_column],
            how="left",
        ).sort(sort_col)

        feats_added = [f for f in self.features_out if f in concat_df.columns]

        concat_df = concat_df.with_columns(
            [
                nw.col(f).fill_null(strategy="forward").over(self.granularity).alias(f)
                for f in feats_added
            ]
        )
        return concat_df
