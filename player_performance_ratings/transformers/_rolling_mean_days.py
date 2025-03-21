from typing import Union, Optional

import pandas as pd
import polars as pl
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformers.base_transformer import BaseLagGenerator
from player_performance_ratings.utils import validate_sorting


class RollingMeanDaysTransformer(BaseLagGenerator):

    def __init__(
            self,
            features: list[str],
            days: Union[int, list[int]],
            granularity: Union[list[str], str] = None,
            scale_by_participation_weight: bool = False,
            add_count: bool = False,
            add_opponent: bool = False,
            prefix: str = "rolling_mean_days",
            column_names: Optional[ColumnNames] = None,
            date_column: Optional[str] = None,
    ):
        self.days = days
        self.scale_by_participation_weight = scale_by_participation_weight
        if isinstance(self.days, int):
            self.days = [self.days]
        super().__init__(
            column_names=column_names,
            features=features,
            iterations=[i for i in self.days],
            prefix=prefix,
            add_opponent=add_opponent,
            granularity=granularity,
        )

        self.add_count = add_count
        self.date_column = date_column
        self._fitted_game_ids = []

        for day in self.days:
            if self.add_count:
                feature = f"{self.prefix}_count{day}"
                self._features_out.append(feature)
                self._entity_features.append(feature)

                if self.add_opponent:
                    self._features_out.append(f"{self.prefix}_count{day}_opponent")

    @nw.narwhalify
    def transform_historical(self, df: FrameT, column_names: Optional[ColumnNames] = None) -> IntoFrameT:
        self.column_names = column_names or self.column_names
        if not self.column_names and not self.date_column:
            raise ValueError("column_names or date_column must be provided")

        if self.scale_by_participation_weight and not self.column_names or self.scale_by_participation_weight and not self.column_names.participation_weight:
            raise ValueError(
                "scale_by_participation_weight requires column_names to be provided"
            )
        ori_cols = df.columns
        if isinstance(df.to_native(), pd.DataFrame):
            ori_type = "pd"
            df = pl.DataFrame(df.to_native())
        else:
            df = df.to_native()
            ori_type = "pl"

        df = df.with_columns(pl.lit(0).alias("is_future"))
        self.granularity = self.granularity or [self.column_names.player_id]
        if self.column_names:
            validate_sorting(df=df, column_names=self.column_names)
            self._store_df(nw.from_native(df))

        concat_df = self._generate_concat_df_with_feats(df)

        df = pl.DataFrame(
            self._create_transformed_df(
                df=nw.from_native(df),
                concat_df=nw.from_native(concat_df),
                match_id_join_on=self.column_names.match_id,
            )
        )

        if self.add_count:
            for day in self.days:
                df = df.with_columns(
                    pl.col(f"{self.prefix}_count{day}")
                    .fill_null(0)
                    .alias(f"{self.prefix}_count{day}")
                )
                if self.add_opponent:
                    df = df.with_columns(
                        pl.col(f"{self.prefix}_count{day}_opponent")
                        .fill_null(0)
                        .alias(f"{self.prefix}_count{day}_opponent")
                    )

        df = df.select(ori_cols + self.features_out)
        unique_cols = (
            [
                self.column_names.player_id,
                self.column_names.match_id,
                self.column_names.team_id,
            ]
            if self.column_names.team_id
            else [self.column_names.team_id, self.column_names.match_id]
        )

        if df.unique(subset=unique_cols).shape[0] != df.shape[0]:
            raise ValueError(
                f"Duplicated rows in df. Df must be a unique combination of {unique_cols}"
            )
        if "is_future" in df.schema:
            df = df.drop("is_future")
        if ori_type == "pd":
            df = nw.from_native(df.to_pandas())
        validate_sorting(df=df, column_names=self.column_names)
        return df

    @nw.narwhalify
    def transform_future(self, df: FrameT) -> IntoFrameT:
        ori_cols = df.columns
        if isinstance(df.to_native(), pd.DataFrame):
            ori_type = "pd"
            df = pl.DataFrame(df.to_native())
        else:
            df = df.to_native()
            ori_type = "pl"

        df = df.with_columns(pl.lit(1).alias("is_future"))
        concat_df = self._generate_concat_df_with_feats(df=df)
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

    def _generate_concat_df_with_feats(self, df: pl.DataFrame) -> pl.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        concat_df = self._concat_with_stored(nw.from_native(df)).to_native()

        for feature_name in self.features:
            if (
                    self.column_names.participation_weight
                    and self.scale_by_participation_weight
            ):
                concat_df = concat_df.with_columns(
                    nw.col(self.column_names.participation_weight)
                    .mean()
                    .over(self.granularity)
                    .alias("__mean_participation_weight")
                )

                concat_df = concat_df.with_columns(
                    (
                            pl.col(feature_name)
                            * pl.col(self.column_names.participation_weight)
                            / pl.col("__mean_participation_weight")
                    ).alias(feature_name)
                ).drop("__mean_participation_weight")

        concat_df = concat_df.with_columns(
            pl.col(self.column_names.start_date).dt.date().alias("__date_day")
        )

        grouped = concat_df.group_by(
            ["__date_day", *self.granularity, self.column_names.team_id]
        ).agg(
            [
                *[pl.col(feature).mean().alias(feature) for feature in self.features],
                pl.count().alias("__count"),
            ]
        )

        for day in self.days:
            grouped = self._add_rolling_feature(
                concat_df=grouped,
                day=day,
                granularity=self.granularity,
            )
        unique_cols = (
            [
                self.column_names.player_id,
                self.column_names.match_id,
                self.column_names.team_id,
            ]
            if self.column_names.team_id
            else [self.column_names.team_id, self.column_names.match_id]
        )

        concat_df = concat_df.join(
            grouped, on=self.granularity + ["__date_day"], how="left"
        ).unique(subset=unique_cols)

        concat_df = concat_df.sort(
            by=[
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
        )

        return concat_df

    def _add_rolling_feature(
            self,
            concat_df: pl.DataFrame,
            day: int,
            granularity: list[str],
    ) -> pl.DataFrame:
        full_calendar = (
            concat_df.select(granularity)
            .unique()
            .join(
                pl.DataFrame(
                    {
                        "__date_day": pl.date_range(
                            concat_df["__date_day"].min(),
                            concat_df["__date_day"].max(),
                            eager=True,
                        )
                    }
                ),
                how="cross",
            )
        )
        full_df = (
            full_calendar.join(concat_df, on=[*granularity, "__date_day"], how="left")
            .sort([*granularity, "__date_day"])
            .unique([*granularity, "__date_day"], maintain_order=True)
        )

        full_df = full_df.with_columns(
            (pl.col("__count") / pl.col("__count").max()).alias("weight")
        )

        rolling_result = (
            full_df.group_by(*granularity, maintain_order=True)
            .agg(
                [
                    pl.col("__date_day").alias("__date_day"),
                ]
                + [
                    (
                            (pl.col(feature) * pl.col("weight")).rolling_sum(
                                window_size=day, min_periods=1
                            )
                            / pl.col("weight").rolling_sum(window_size=day, min_periods=1)
                    )
                .shift(1)
                .alias(f"{self.prefix}_{feature}{day}")
                    for feature in self.features
                ]
                + [
                    pl.col("__count")
                .rolling_sum(window_size=day, weights=None, min_periods=1)
                .shift(1)
                .alias(f"{self.prefix}_count{day}")
                ]
            )
            .explode(["__date_day", *self._entity_features])
        )

        concat_df = concat_df.join(
            rolling_result, on=[*granularity, "__date_day"], how="left"
        )

        return concat_df

    def reset(self):
        self._df = None
        self._fitted_game_ids = []

    @property
    def features_out(self) -> list[str]:
        return self._features_out
