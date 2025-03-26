from typing import Optional

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
        column_names: Optional[ColumnNames] = None,
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
        self.column_names = column_names
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
    @required_lag_column_names
    @row_count_validator
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        input_cols = df.columns
        self.column_names = column_names or self.column_names

        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            df = nw.from_native(pl.DataFrame(native))
            ori_native = "pd"
        else:
            ori_native = "pl"

        if df.schema[self.binary_column] in [nw.Float64, nw.Float32]:
            df = df.with_columns(nw.col(self.binary_column).cast(nw.Int64))

        df = df.with_columns(nw.lit(0).alias("is_future"))
        if self.column_names:
            self.match_id_update_column = (
                self.column_names.update_match_id or self.match_id_update_column
            )
            validate_sorting(df=df, column_names=self.column_names)
            additional_cols_to_use = [self.binary_column] + (
                [self.prob_column] if self.prob_column else []
            )
            self._store_df(df, additional_cols_to_use=additional_cols_to_use)
            concat_df = self._concat_with_stored(df)
            concat_df = self._concat_with_stored_and_calculate_feats(concat_df)
            transformed_df = self._create_transformed_df(
                df=df,
                concat_df=concat_df,
                match_id_join_on=self.column_names.update_match_id,
            )
        else:
            transformed_df = self._concat_with_stored_and_calculate_feats(df).sort(
                "__row_index"
            )
            transformed_df = df.join(
                transformed_df.select(["__row_index", *self.features_out]),
                on="__row_index",
                how="left",
            )

        # transformed_df = self._add_weighted_prob(transformed_df=transformed_df)

        transformed_df = transformed_df.select([*input_cols, *self.features_out])
        if "__row_index" in transformed_df.columns:
            transformed_df = transformed_df.drop("__row_index")
        if ori_native == "pd":
            return transformed_df.to_pandas()
        return transformed_df

    @nw.narwhalify
    @future_validator
    @row_count_validator
    def transform_future(self, df: FrameT) -> IntoFrameT:
        input_cols = df.columns
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
        concat_df = self._concat_with_stored_and_calculate_feats(df=df)
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

        transformed_future = transformed_future.select(
            [*input_cols, *self.features_out]
        )

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

    def _concat_with_stored_and_calculate_feats(self, df: FrameT) -> FrameT:
        if self.column_names and self._df is not None:
            sort_col = self.column_names.start_date
            concat_df = self._concat_with_stored(df)

        else:
            concat_df = df
            if "__row_index" not in concat_df.columns:
                concat_df = concat_df.with_row_index(name="__row_index")
            sort_col = "__row_index"

        groupby_cols = [
            self.match_id_update_column,
            *self.granularity,
            #   self.column_names.start_date,
            #   "is_future",
        ]

        aggregation = {f: nw.mean(f).alias(f) for f in self.features}
        aggregation[self.binary_column] = nw.median(self.binary_column)
        aggregation[sort_col] = nw.mean(sort_col)
        grouped = concat_df.group_by(groupby_cols).agg(list(aggregation.values()))

        grouped = grouped.sort(
            by=[
                sort_col,
                # "is_future",
                self.match_id_update_column,
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
                    self.match_id_update_column,
                    *self.granularity,
                    *feats_added,
                ]
            ),
            on=[self.match_id_update_column, *self.granularity],
            how="left",
        )

        concat_df = concat_df.with_columns(
            [nw.col(feats_added).fill_null(strategy="forward").over(self.granularity)]
        )

        return self._add_weighted_prob(transformed_df=concat_df)

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
