import logging
from typing import Optional, List

import narwhals
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformers.base_transformer import BaseTransformer


class SklearnEstimatorImputer(BaseTransformer):

    def __init__(
            self,
            features: list[str],
            target_name: str,
            estimator: Optional = None,
    ):
        self.target_name = target_name
        super().__init__(features=features, features_out=features)
        self.estimator = estimator or LGBMRegressor()

    @nw.narwhalify
    def fit_transform(self, df: FrameT, column_names: Optional[ColumnNames] = None) -> IntoFrameT:
        fit_df = df.filter((nw.col(self.target_name).is_finite())&(~nw.col(self.target_name).is_null()))
        self.estimator.fit(fit_df.select(self.features), fit_df[self.target_name])
        return self.transform(df)

    @nw.narwhalify
    def transform(self, df: FrameT) -> IntoFrameT:
        df = df.with_columns(
            **{
                f"imputed_col_{self.target_name}": self.estimator.predict(
                    df.select(self.features)
                )
            }
        )
        df = df.with_columns(
            nw.when(
                (nw.col(self.target_name).is_null())
                | ~(nw.col(self.target_name).is_finite())
            )
            .then(nw.col(f"imputed_col_{self.target_name}"))
            .otherwise(
                    nw.col(self.target_name)
                    ).alias(self.target_name)

        )
        return df.drop(f"imputed_col_{self.target_name}").to_native()

    @property
    def features_out(self) -> list[str]:
        return [self.target_name]


class PartialStandardScaler(BaseTransformer):

    def __init__(
            self,
            features: list[str],
            ratio: float = 0.55,
            target_mean: float = 0.5,
            max_value: float = 2,
            prefix: str = "",
    ):
        features_out = []
        self.prefix = prefix
        for feature in self.features:
            features_out.append(self.prefix + feature)
        super().__init__(features=features, features_out=features_out)
        self.ratio = ratio
        self.target_mean = target_mean
        self.max_value = max_value

        self._features_mean = {}
        self._features_std = {}
        self._features_out = []

    @narwhals.narwhalify
    def fit_transform(self, df: FrameT, column_names: Optional[ColumnNames] = None) -> IntoFrameT:
        for feature in self.features:
            df = df.with_columns(
                nw.col(feature).filter(nw.col(feature).is_in([float('inf'), float('-inf')]))
            )

            self._features_mean[feature] = (
                df[feature].mean()
            )
            self._features_std[feature] = (
                df[feature].std()
            )

        return self.transform(df)

    @narwhals.narwhalify
    def transform(self, df: FrameT) -> IntoFrameT:
        for feature in self.features:
            new_feature = self.prefix + feature
            df = df.with_column(
                (
                        (nw.col(feature) - self._features_mean[feature])
                        / self._features_std[feature]
                        * self.ratio
                        + self.target_mean
                ).alias(new_feature)
            )

            df = df.with_column(
                nw.col(new_feature)
                .clip(-self.max_value + self.target_mean, self.max_value)
                .alias(new_feature)
            )
        return df.to_native()

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class MinMaxTransformer(BaseTransformer):

    def __init__(
            self,
            features: list[str],
            quantile: float = 0.98,
            multiply_align: bool = False,
            add_align: bool = True,
            prefix: str = "",
    ):

        features_out = [prefix + f for f in features]

        super().__init__(features=features, features_out=features_out)
        self.quantile = quantile
        self.prefix = prefix
        self._trained_mean_values = {}
        self._min_values = {}
        self._max_values = {}
        self.multiply_align = multiply_align
        self.add_align = add_align

        if self.quantile < 0 or self.quantile > 1:
            raise ValueError("quantile must be between 0 and 1")

    @nw.narwhalify
    def fit_transform(self, df: FrameT, column_names: Optional[ColumnNames]) -> FrameT:
        df = df.copy()
        for feature in self.features:
            self._min_values[feature] = \
            df.select(nw.col(feature).quantile(1 - self.quantile, interpolation='linear')).to_numpy()[0]
            self._max_values[feature] = \
            df.select(nw.col(feature).quantile(self.quantile, interpolation='linear')).to_numpy()[0]

            # Check for equal min and max values
            if self._min_values[feature] == self._max_values[feature]:
                raise ValueError(
                    f"Feature {feature} has the same min and max value. This can lead to division by zero. "
                    f"This feature is not suited for MinMaxTransformer"
                )

            # Normalize the feature
            normalized_feature = (
                    (nw.col(feature) - self._min_values[feature])
                    / (self._max_values[feature] - self._min_values[feature])
            ).clip(0, 1)

            # Assign normalized feature to a new column
            df = df.with_column(normalized_feature.alias(self.prefix + feature))

            # Calculate the mean of the normalized feature
            self._trained_mean_values[feature] = df.select(nw.col(self.prefix + feature).mean()).to_numpy()[0]

            # Apply alignment adjustments
            if self.multiply_align:
                df = df.with_column(
                    (nw.col(self.prefix + feature) * 0.5 / self._trained_mean_values[feature]).alias(
                        self.prefix + feature)
                )

            if self.add_align:
                df = df.with_column(
                    (nw.col(self.prefix + feature) + 0.5 - self._trained_mean_values[feature]).alias(
                        self.prefix + feature)
                )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature in self.features:
            df[self.prefix + feature] = (df[feature] - self._min_values[feature]) / (
                    self._max_values[feature] - self._min_values[feature]
            )
            df[self.prefix + feature].clip(0, 1, inplace=True)
            if self.multiply_align:
                df[self.prefix + feature] = (
                        df[self.prefix + feature] * 0.5 / self._trained_mean_values[feature]
                )
            if self.add_align:
                df[self.prefix + feature] = (
                        df[self.prefix + feature] + 0.5 - self._trained_mean_values[feature]
                )

        return df

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class DiminishingValueTransformer(BaseTransformer):

    def __init__(
            self,
            features: List[str],
            cutoff_value: Optional[float] = None,
            quantile_cutoff: float = 0.93,
            excessive_multiplier: float = 0.8,
            reverse: bool = False,
    ):
        super().__init__(features=features, features_out=features)
        self.cutoff_value = cutoff_value
        self.excessive_multiplier = excessive_multiplier
        self.quantile_cutoff = quantile_cutoff
        self.reverse = reverse
        self._feature_cutoff_value = {}

    @nw.narwhalify
    def fit_transform(self, df: FrameT, column_names: Optional[ColumnNames] = None) -> IntoFrameT:

        for feature_name in self.features:
            if self.cutoff_value is None:
                if self.reverse:
                    self._feature_cutoff_value[feature_name] = df[
                        feature_name
                    ].quantile(1 - self.quantile_cutoff, interpolation='linear')
                else:
                    self._feature_cutoff_value[feature_name] = df[
                        feature_name
                    ].quantile(self.quantile_cutoff, interpolation='linear')
            else:
                self._feature_cutoff_value[feature_name] = self.cutoff_value

        return self.transform(df)
    @nw.narwhalify
    def transform(self, df: FrameT) -> IntoFrameT:
        for feature_name in self.features:
            cutoff_value = self._feature_cutoff_value[feature_name]

            if self.reverse:
                df = df.with_columns(
                    nw.when(nw.col(feature_name) <= cutoff_value)
                    .then(
                        -(cutoff_value - nw.col(feature_name)) * self.excessive_multiplier + cutoff_value
                    )
                    .otherwise(nw.col(feature_name))
                    .alias(feature_name)
                )
            else:
                df = df.with_columns(
                    nw.when(nw.col(feature_name) >= cutoff_value)
                    .then(
                        (nw.col(feature_name) - cutoff_value)
                        .clip(lower_bound=0)
                        * self.excessive_multiplier
                        + cutoff_value
                    )
                    .otherwise(nw.col(feature_name))
                    .alias(feature_name)
                )

        return df.to_native()

    @property
    def features_out(self) -> list[str]:
        return self.features


class SymmetricDistributionTransformer(BaseTransformer):

    def __init__(
            self,
            features: List[str],
            granularity: Optional[list[str]] = None,
            skewness_allowed: float = 0.15,
            max_iterations: int = 50,
            min_excessive_multiplier: float = 0.04,
            min_rows: int = 10,
            prefix: str = "symmetric_",
    ):
        super().__init__(features=features, features_out=[prefix + feature for feature in features])
        self.granularity = granularity
        self.skewness_allowed = skewness_allowed
        self.max_iterations = max_iterations
        self.min_rows = min_rows
        self.min_excessive_multiplier = min_excessive_multiplier
        self.prefix = prefix

        self._diminishing_value_transformer = {}

    @nw.narwhalify
    def fit_transform(self, df: FrameT) -> IntoFrameT:

        if self.granularity:

            df = df.with_columns(
                nw.concat_str([nw.col(col) for col in self.granularity], separator="_").alias("__concat_granularity")
            )


        for feature in self.features:
            feature_min = df[feature].min()
            feature_max = df[feature].max()

            if feature_min == feature_max:
                raise ValueError(
                    f"SymmetricDistributionTransformer: {feature} has the same min and max value."
                )
            self._diminishing_value_transformer[feature] = {}
            if self.granularity:

                unique_values = df["__concat_granularity"].unique()
                for unique_value in unique_values:
                    rows = df.filter(nw.col("__concat_granularity") == unique_value)

                    self._fit(
                        rows=rows, feature=feature, granularity_value=unique_value
                    )

            else:
                self._fit(rows=df, feature=feature, granularity_value=None)

        return self.transform(df)

    def _fit(
            self, rows: FrameT, feature: str, granularity_value: Optional[str]
    ) -> None:

        skewness = rows[feature].skew()
        excessive_multiplier = 0.8
        quantile_cutoff = 0.95

        iteration = 0
        while (
                abs(skewness) > self.skewness_allowed
                and len(rows) > self.min_rows
                and iteration < self.max_iterations
        ):

            if skewness < 0:
                reverse = True
            else:
                reverse = False

            self._diminishing_value_transformer[feature][granularity_value] = (
                DiminishingValueTransformer(
                    features=[feature],
                    reverse=reverse,
                    excessive_multiplier=excessive_multiplier,
                    quantile_cutoff=quantile_cutoff,
                )
            )
            transformed_rows = self._diminishing_value_transformer[feature][
                granularity_value
            ].fit_transform(rows)
            new_excessive_multiplier = excessive_multiplier * 0.94
            if new_excessive_multiplier < self.min_excessive_multiplier:
                break
            excessive_multiplier = new_excessive_multiplier
            next_quantile_cutoff = quantile_cutoff * 0.994
            if (
                    transformed_rows[feature].quantile(next_quantile_cutoff)
                    > transformed_rows[feature].min()
            ):
                quantile_cutoff = next_quantile_cutoff
            iteration += 1
            skewness = transformed_rows[feature].skew()

    @nw.narwhalify
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.granularity:
            df = df.with_columns(
                nw.concat_str([nw.col(col) for col in self.granularity], separator="_").alias(
                    "__concat_granularity")
            )

        for feature in self.features:
            out_feature = self.prefix + feature

            if self.granularity:
                unique_values = df["__concat_granularity"].unique().to_list()

                if len(unique_values) > 100:
                    logging.warning(
                        f"SymmetricDistributionTransformer: {feature} has more than 100 unique values."
                        f" This can lead to long runtimes. Consider setting a lower granularity."
                    )

                if len(self._diminishing_value_transformer[feature]) == 0:
                    df = df.with_columns(nw.col(feature).alias(out_feature))
                else:
                    updated_rows = []
                    for unique_value in unique_values:
                        rows = df.filter(nw.col("__concat_granularity") == unique_value)

                        if unique_value in self._diminishing_value_transformer[feature]:
                            rows = self._diminishing_value_transformer[feature][unique_value].transform(rows)

                        updated_rows.append(rows)

                    df = nw.concat(updated_rows).with_columns(
                        nw.col(feature).alias(out_feature)
                    )
            else:
                if None in self._diminishing_value_transformer[feature]:
                    df = self._diminishing_value_transformer[feature][None].transform(df)

        if "__concat_granularity" in df.columns:
            df = df.drop(["__concat_granularity"])

        return df

    @property
    def features_out(self) -> list[str]:
        return [self.prefix + feature for feature in self.features]


class GroupByTransformer(BaseTransformer):

    def __init__(
            self,
            features: list[str],
            granularity: list[str],
            agg_func: str = "mean",
            prefix: str = "mean_",
    ):
        super().__init__(features=features)
        self.granularity = granularity
        self.agg_func = agg_func
        self.prefix = prefix
        self._features_out = []
        self._grouped = None
        for feature in self.features:
            self._features_out.append(self.prefix + feature)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._grouped = (
            df.groupby(self.granularity)[self.features]
            .agg(self.agg_func)
            .reset_index()
            .rename(
                columns={feature: self.prefix + feature for feature in self.features}
            )
        )

        return self.transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(self._grouped, on=self.granularity, how="left")

    @property
    def features_out(self) -> list[str]:
        return self._features_out
