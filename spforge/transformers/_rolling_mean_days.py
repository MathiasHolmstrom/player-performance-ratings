from typing import Union, Optional

import pandas as pd
import polars as pl
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from spforge import ColumnNames
from spforge.transformers.base_transformer import (
    BaseLagGenerator,
    required_lag_column_names,
    row_count_validator,
    future_validator,
)
from spforge.utils import validate_sorting


class RollingMeanDaysTransformer(BaseLagGenerator):

    def __init__(
        self,
        features: list[str],
        days: int,
        granularity: Union[list[str], str],
        scale_by_participation_weight: bool = False,
        add_count: bool = False,
        add_opponent: bool = False,
        prefix: str = "rolling_mean_days",
        column_names: Optional[ColumnNames] = None,
        date_column: Optional[str] = None,
        match_id_update_column: Optional[str] = None,
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
            match_id_update_column=match_id_update_column,
        )

        self.add_count = add_count
        self.date_column = date_column
        self._fitted_game_ids = []

        self._count_column_name = f"{self.prefix}_count{str(self.days)}"
        if self.add_count:
            self._features_out.append(self._count_column_name)
            self._entity_features.append(self._count_column_name)

            if self.add_opponent:
                self._features_out.append(f"{self._count_column_name}_opponent")

    @nw.narwhalify
    @row_count_validator
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:

        ori_cols = df.columns
        self.column_names = column_names or self.column_names
        if not self.column_names and not self.date_column:
            raise ValueError("column_names or date_column must be provided")

        if (
            self.scale_by_participation_weight
            and not self.column_names
            or self.scale_by_participation_weight
            and not self.column_names.participation_weight
        ):
            raise ValueError(
                "scale_by_participation_weight requires column_names to be provided"
            )
        if self.column_names:
            self.date_column = self.column_names.start_date or self.date_column
            self.match_id_update_column = (
                self.column_names.update_match_id or self.match_id_update_column
            )
        else:
            if "__row_index" not in df.columns:
                df = df.with_row_index(name="__row_index")

        if isinstance(df.to_native(), pd.DataFrame):
            ori_type = "pd"
            df = pl.DataFrame(df.to_native())
        else:
            df = df.to_native()
            ori_type = "pl"
        df = df.with_columns(pl.lit(0).alias("is_future"))
        if self.column_names:
            validate_sorting(df=df, column_names=self.column_names)
            self._store_df(nw.from_native(df))

            concat_df = self._concat_with_stored_and_calculate_feats(df)

            transformed_df = pl.DataFrame(
                self._create_transformed_df(
                    df=nw.from_native(df),
                    concat_df=nw.from_native(concat_df),
                    match_id_join_on=self.match_id_update_column,
                )
            )
            if self.add_opponent and self.add_count:
                transformed_df = transformed_df.with_columns(
                    pl.col(f"{self._count_column_name}_opponent")
                    .fill_null(0)
                    .alias(f"{self._count_column_name}_opponent")
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
            df = df.join(
                transformed_df.select([*join_cols, *self.features_out]),
                on=join_cols,
                how="left",
            ).sort(sort_cols)

        else:
            transformed_df = self._concat_with_stored_and_calculate_feats(df)
            df = df.join(
                transformed_df.select(["__row_index", *self.features_out]),
                on="__row_index",
                how="left",
            ).sort("__row_index")

        df = df.select(ori_cols + self.features_out)

        if "is_future" in df.schema:
            df = df.drop("is_future")
        if ori_type == "pd":
            df = nw.from_native(df.to_pandas())
        return df

    @nw.narwhalify
    @future_validator
    @row_count_validator
    def transform_future(self, df: FrameT) -> IntoFrameT:
        ori_cols = df.columns
        if isinstance(df.to_native(), pd.DataFrame):
            ori_type = "pd"
            df = pl.DataFrame(df.to_native())
        else:
            df = df.to_native()
            ori_type = "pl"

        df = df.with_columns(pl.lit(1).alias("is_future"))
        concat_df = self._concat_with_stored_and_calculate_feats(df=df)
        concat_df = concat_df.filter(
            pl.col(self.column_names.match_id).is_in(
                df[self.column_names.match_id].unique().to_list()
            )
        )
        transformed_future = pl.DataFrame(
            self._generate_future_feats(
                transformed_df=nw.from_native(concat_df), ori_df=nw.from_native(df)
            )
        )
        if "is_future" in transformed_future.columns:
            transformed_future = transformed_future.drop("is_future")

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
            (pl.col(self._count_column_name) - pl.col("__count1")).alias(
                self._count_column_name
            )
        )

        grp = grp.with_columns(
            pl.col(col)
            .sum()
            .over([self.date_column, *self.granularity])
            .alias(f"days_sum_{col}")
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

        concat_df = concat_df.join(grp, on=grp_cols, how="left")
        if self.add_count:
            concat_df = concat_df.with_columns(
                pl.col(self._count_column_name)
                .fill_null(0)
                .alias(self._count_column_name)
            )
        return concat_df

    def reset(self):
        self._df = None
        self._fitted_game_ids = []

    @property
    def features_out(self) -> list[str]:
        return self._features_out
