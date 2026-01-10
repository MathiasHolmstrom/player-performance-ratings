import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge.data_structures import ColumnNames
from spforge.feature_generator._base import LagGenerator
from spforge.feature_generator._utils import (
    future_lag_transformations_wrapper,
    future_validator,
    historical_lag_transformations_wrapper,
    required_lag_column_names,
    transformation_validator,
)


class BinaryOutcomeRollingMeanTransformer(LagGenerator):

    def __init__(
        self,
        features: list[str],
        window: int,
        binary_column: str,
        granularity: list[str] = None,
        prob_column: str | None = None,
        min_periods: int = 1,
        add_opponent: bool = False,
        prefix: str = "rolling_mean_binary",
        match_id_column: str | None = None,
        update_column: str | None = None,
        column_names: ColumnNames | None = None,
    ):
        super().__init__(
            match_id_column=match_id_column,
            features=features,
            add_opponent=add_opponent,
            prefix=prefix,
            iterations=[],
            granularity=granularity,
            update_column=update_column,
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
            self._entity_features_out.append(feature1)
            self._entity_features_out.append(feature2)

            if self.add_opponent:
                self._features_out.append(f"{self.prefix}_{feature_name}{self.window}_1_opponent")
                self._features_out.append(f"{self.prefix}_{feature_name}{self.window}_0_opponent")

        if self.prob_column:
            self._weighted_features_out = {}
            self._weighted_features_out_opponent = {}
            for feature_name in self.features:
                self._weighted_features_out[feature_name] = (
                    f"{self.prefix}_{feature_name}_{self.prob_column}{self.window}"
                )
                if self.add_opponent:
                    self._weighted_features_out_opponent[feature_name] = (
                        f"{self.prefix}_{feature_name}_{self.prob_column}{self.window}_opponent"
                    )
                    self._features_out.extend(
                        [
                            self._weighted_features_out[feature_name],
                            self._weighted_features_out_opponent[feature_name],
                        ]
                    )
                else:
                    self._features_out.append(self._weighted_features_out[feature_name])

        self._estimator_features_out = self._features_out.copy()

    @nw.narwhalify
    @historical_lag_transformations_wrapper
    @required_lag_column_names
    @transformation_validator
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:

        if df.schema[self.binary_column] in [nw.Float64, nw.Float32]:
            df = df.with_columns(nw.col(self.binary_column).cast(nw.Int64))

        add_cols = (
            [self.binary_column, self.prob_column] if self.prob_column else [self.binary_column]
        )
        grouped = self._maybe_group(df, additional_cols=add_cols)
        if self.column_names:
            self._store_df(grouped_df=grouped, ori_df=df, additional_cols=add_cols)
            grouped_with_feats = self._generate_features(grouped, ori_df=df)
            df = self._merge_into_input_df(
                df=df,
                concat_df=grouped_with_feats,
                features_out=self._entity_features_out,
            )

        else:
            join_on_cols = self.group_to_granularity
            grouped_with_feats = self._generate_features(grouped, ori_df=df).sort("__row_index")
            feats = [f for f in self.features_out if f not in self._weighted_features_out.values()]
            df = df.join(
                grouped_with_feats.select([*join_on_cols, *feats]),
                on=join_on_cols,
                how="left",
            ).unique("__row_index")

        df = self._post_features_generated(df)
        return self._add_weighted_prob(df)

    @nw.narwhalify
    @future_lag_transformations_wrapper
    @future_validator
    @transformation_validator
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:

        if self.binary_column in df.columns and df.schema[self.binary_column] in [
            nw.Float64,
            nw.Float32,
        ]:
            df = df.with_columns(nw.col(self.binary_column).cast(nw.Int64))
        add_cols = (
            [self.binary_column, self.prob_column] if self.prob_column else [self.binary_column]
        )
        sort_col = self.column_names.start_date if self.column_names else "__row_index"
        grouped = self._group_to_granularity_level(
            df=df, sort_col=sort_col, additional_cols=add_cols
        )
        grouped_df_with_feats = self._generate_features(df=grouped, ori_df=df)
        df = self._merge_into_input_df(df=df, concat_df=grouped_df_with_feats)
        df = self._post_features_generated(df)
        df = self._forward_fill_future_features(df=df)
        return self._add_weighted_prob(df)

    def _get_known_future_features(self) -> list[str]:
        known_future_features = []
        if self.prob_column:
            for _idx, feature_name in enumerate(self.features):
                weighted_prob_feat_name = (
                    f"{self.prefix}_{feature_name}_{self.prob_column}{self.window}"
                )
                known_future_features.append(weighted_prob_feat_name)

        return known_future_features

    def _generate_features(self, df: IntoFrameT, ori_df: IntoFrameT) -> IntoFrameT:
        if self.column_names and self._df is not None:
            sort_col = self.column_names.start_date
            add_cols = (
                [self.binary_column, self.prob_column] if self.prob_column else [self.binary_column]
            )
            concat_df = self._concat_with_stored(
                group_df=df, ori_df=ori_df, additional_cols=add_cols
            )
        else:
            concat_df = df
            if "__row_index" not in concat_df.columns:
                concat_df = concat_df.with_row_index(name="__row_index")
            sort_col = "__row_index"
        concat_df = concat_df.sort(sort_col)
        feats_added = []

        for feature in self.features:
            concat_df = concat_df.with_columns(
                [
                    nw.when(nw.col(self.binary_column) == 1)
                    .then(nw.col(feature))
                    .alias("value_result_1"),
                    nw.when(nw.col(self.binary_column) == 0)
                    .then(nw.col(feature))
                    .alias("value_result_0"),
                ]
            )

            concat_df = concat_df.with_columns(
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

        concat_df = concat_df.with_columns(
            [nw.col(feats_added).fill_null(strategy="forward").over(self.granularity)]
        )
        return concat_df

    def _add_weighted_prob(self, transformed_df: IntoFrameT) -> IntoFrameT:

        if self.prob_column:
            for _idx, feature_name in enumerate(self.features):

                transformed_df = transformed_df.with_columns(
                    (
                        nw.col(f"{self.prefix}_{feature_name}{self.window}_1")
                        * nw.col(self.prob_column)
                        + nw.col(f"{self.prefix}_{feature_name}{self.window}_0")
                        * (1 - nw.col(self.prob_column))
                    ).alias(self._weighted_features_out[feature_name])
                )
                if self.add_opponent:
                    transformed_df = transformed_df.with_columns(
                        (
                            nw.col(f"{self.prefix}_{feature_name}{self.window}_1_opponent")
                            * (1 - nw.col(self.prob_column))
                            + nw.col(f"{self.prefix}_{feature_name}{self.window}_0_opponent")
                            * nw.col(self.prob_column)
                        ).alias(self._weighted_features_out_opponent[feature_name])
                    )

        return transformed_df
