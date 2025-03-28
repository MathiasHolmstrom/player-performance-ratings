import logging
from typing import Union, Optional, Literal
import narwhals as nw
import pandas as pd
import polars as pl
from narwhals.typing import FrameT, IntoFrameT

from spforge import ColumnNames
from spforge.transformers import RollingMeanTransformer
from spforge.transformers.base_transformer import (
    BaseLagGenerator,
    required_lag_column_names,
    row_count_validator,
    BaseTransformer,
    future_validator,
)
from spforge.utils import validate_sorting


class OpponentTransformer(BaseLagGenerator):
    """
    Calculates the rolling mean of a list of features from the opponents perspective.
    In contrast to RollingMeanTransformer and setting add_opponent = True, the OpponentRollingMeanTransformer does not calculate the mean of the rolling mean for the team itself.
    Rather it calculates the rolling-mean performance of every entity that has faced the opponent over the window period.

    This is useful for creating features that indicate how the opponent performs in different contexts.
    Example:
        - Does the opponent allow more points against centers than other positions?
    df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)
    transformer = OpponentRollingMeanTransformer(
        features=["points"],
        window=15,
        granularity=['position'],
        opponent_column='opponent_team_id'
        )
    df = transformer.generate_historical(df)




    Use .transform_historical() to generate rolling mean for historical data.
        The historical data is stored as instance-variables after calling .generate_historical()
    Use .transform_future() to generate rolling mean for future data after having called .generate_historical()
    """

    def __init__(
        self,
        features: list[str],
        window: int,
        granularity: Union[list[str], str],
        min_periods: int = 1,
        are_estimator_features=True,
        prefix: str = "opponent_rolling_mean",
        match_id_update_column: Optional[str] = None,
        team_column: Optional[str] = None,
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
            match_id_update_column=match_id_update_column,
        )
        self.window = window
        self.min_periods = min_periods
        self.team_column = team_column
        self.opponent_column = opponent_column
        self.transformation = transformation
        self._transformer: BaseTransformer

    @nw.narwhalify
    @required_lag_column_names
    @row_count_validator
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        """
        Generates rolling mean for historical data

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
            self.match_id_update_column = (
                self.column_names.update_match_id or self.match_id_update_column
            )
            self.team_column = self.column_names.team_id or self.team_column
        else:
            assert self.team_column is not None, "team_column must be set if column names is not passed"
            assert self.match_id_update_column is not None, "match_id_update_column must be set if column names is not passed"
        if self.transformation == "rolling_mean":
            self._transformer = RollingMeanTransformer(
                granularity=[self.opponent_column, *self.granularity],
                features=self.features,
                window=self.window,
                min_periods=self.min_periods,
                match_id_update_column=self.match_id_update_column,
                unique_constraint=[
                    self.opponent_column,
                    self.match_id_update_column,
                    *self.granularity,
                ],
            )
        else:
            raise NotImplementedError("Only rolling_mean transformation is supported")

        if self.column_names:
            df = df.with_columns(nw.lit(0).alias("is_future"))
            self._store_df(df)
            concat_df = self._concat_with_stored_and_calculate_feats(
                df, is_future=False
            )
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
            transformed_df = self._concat_with_stored_and_calculate_feats(
                df, is_future=False
            ).sort("__row_index")
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
        concat_df = self._concat_with_stored_and_calculate_feats(df=df, is_future=True)
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

    def _concat_with_stored_and_calculate_feats(
        self, df: FrameT, is_future: bool
    ) -> FrameT:

        cols_to_drop = [c for c in self.features_out if c in df.columns]
        df = df.drop(cols_to_drop)

        if self.opponent_column not in df.columns:
            gt = df.unique([self.match_id_update_column, self.team_column]).select(
                [self.match_id_update_column, self.team_column]
            )
            gt_opponent = gt.join(
                gt, on=self.match_id_update_column, how="left", suffix="__opp"
            )
            gt_opponent = gt_opponent.filter(
                nw.col(self.team_column) != nw.col(f"{self.team_column}__opp")
            )
            gt_opponent = gt_opponent.with_columns(
                nw.col(f"{self.team_column}__opp").alias(self.opponent_column)
            ).drop(f"{self.team_column}__opp")
            df = df.join(
                gt_opponent,
                on=[self.match_id_update_column, self.team_column],
                how="left",
            )

        if self.column_names and self._df is not None:
            sort_col = self.column_names.start_date
            grouped = (
                df.group_by(
                    [
                        self.match_id_update_column,
                        self.team_column,
                        self.opponent_column,
                        *self.granularity,
                        sort_col,
                    ]
                )
                .agg(nw.col(self.features).mean())
                .sort(sort_col)
            )
        else:
            sort_col = "__row_index"

            if "__row_index" not in df.columns:
                df = df.with_row_index(name="__row_index")

            grouped = (
                df.group_by(
                    [
                        self.match_id_update_column,
                        self.team_column,
                        self.opponent_column,
                        *self.granularity,
                    ]
                )
                .agg([nw.col(self.features).mean(), nw.col("__row_index").min()])
                .sort("__row_index")
            )

        if is_future:
            grouped = nw.from_native(self._transformer.transform_future(grouped))
        else:
            if self.column_names:
                column_names_transformer = ColumnNames(**self.column_names.__dict__)
                column_names_transformer.player_id = None
            else:
                column_names_transformer = None
            grouped = nw.from_native(
                self._transformer.transform_historical(
                    grouped, column_names=column_names_transformer
                )
            )

        df = df.join(
            grouped.select(
                [
                    self.team_column,
                    self.match_id_update_column,
                    *self._transformer.features_out,
                    *self.granularity
                ]
            ),
            on=[self.match_id_update_column, self.team_column, *self.granularity],
        ).sort(sort_col)
        rename_cols = {
            feature: self.features_out[idx]
            for idx, feature in enumerate(self._transformer.features_out)
        }
        return df.rename(rename_cols)
