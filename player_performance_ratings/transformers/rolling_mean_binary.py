from typing import Optional

import pandas as pd
import polars as pl
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformers.base_transformer import BaseLagGenerator


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
            match_id_update_column: Optional[str] = None,
    ):
        super().__init__(
            features=features,
            add_opponent=add_opponent,
            prefix=prefix,
            iterations=[],
            granularity=granularity,
            match_id_update_column=match_id_update_column,
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
