import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from player_performance_ratings.predictor import Predictor
from player_performance_ratings.transformers.base_transformer import (
    BasePerformancesTransformer,
)


class SklearnEstimatorImputer(BasePerformancesTransformer):

    def __init__(
        self,
        features: list[str],
        target_name: str,
        estimator: Optional[LGBMRegressor] = None,
    ):
        super().__init__(features=features)
        self.estimator = estimator or LGBMRegressor()
        self.target_name = target_name

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.estimator.fit(df[self.features], df[self.target_name])
        df = df.assign(
            **{
                f"imputed_col_{self.target_name}": self.estimator.predict(
                    df[self.features]
                )
            }
        )
        df[self.target_name] = df[self.target_name].fillna(
            df[f"imputed_col_{self.target_name}"]
        )
        return df.drop(columns=[f"imputed_col_{self.target_name}"])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            **{
                f"imputed_col_{self.target_name}": self.estimator.predict(
                    df[self.features]
                )
            }
        )
        df[self.target_name] = df[self.target_name].fillna(
            df[f"imputed_col_{self.target_name}"]
        )
        return df.drop(columns=[f"imputed_col_{self.target_name}"])

    @property
    def features_out(self) -> list[str]:
        return [self.target_name]


class PartialStandardScaler(BasePerformancesTransformer):

    def __init__(
        self,
        features: list[str],
        ratio: float = 0.55,
        target_mean: float = 0.5,
        max_value: float = 2,
        prefix: str = "",
    ):
        super().__init__(features=features)
        self.ratio = ratio
        self.target_mean = target_mean
        self.prefix = prefix
        self.max_value = max_value

        self._features_mean = {}
        self._features_std = {}
        self._features_out = []
        for feature in self.features:
            self._features_out.append(self.prefix + feature)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            self._features_mean[feature] = (
                df[feature].replace([np.inf, -np.inf], np.nan).mean()
            )
            self._features_std[feature] = (
                df[feature].replace([np.inf, -np.inf], np.nan).std()
            )

        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            new_feature = self.prefix + feature
            df = df.assign(
                **{
                    new_feature: (
                        (df[feature] - self._features_mean[feature])
                        / self._features_std[feature]
                    )
                    * self.ratio
                    + self.target_mean
                }
            )
            df[new_feature] = df[new_feature].clip(
                -self.max_value + self.target_mean, self.max_value
            )
        return df

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class MinMaxTransformer(BasePerformancesTransformer):

    def __init__(
        self,
        features: list[str],
        quantile: float = 0.98,
        multiply_align: bool = False,
        add_align: bool = True,
        prefix: str = "",
    ):
        super().__init__(features=features)
        self.quantile = quantile
        self.prefix = prefix
        self._trained_mean_values = {}
        self._min_values = {}
        self._max_values = {}
        self.multiply_align = multiply_align
        self.add_align = add_align

        if self.quantile < 0 or self.quantile > 1:
            raise ValueError("quantile must be between 0 and 1")

        self._features_out = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature in self.features:
            self._min_values[feature] = df[feature].quantile(1 - self.quantile)
            self._max_values[feature] = df[feature].quantile(self.quantile)

            if self._min_values[feature] == self._max_values[feature]:
                raise ValueError(
                    f"Feature {feature} has the same min and max value. This can lead to division by zero. "
                    f"This feature is not suited for MinMaxTransformer"
                )

            df[self.prefix + feature] = (df[feature] - self._min_values[feature]) / (
                self._max_values[feature] - self._min_values[feature]
            )
            df[self.prefix + feature].clip(0, 1, inplace=True)
            self._trained_mean_values[feature] = df[self.prefix + feature].mean()
            if self.multiply_align:
                df[self.prefix + feature] = (
                    df[self.prefix + feature] * 0.5 / self._trained_mean_values[feature]
                )
            if self.add_align:
                df[self.prefix + feature] = (
                    df[self.prefix + feature] + 0.5 - self._trained_mean_values[feature]
                )

            self._features_out.append(self.prefix + feature)

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


class DiminishingValueTransformer(BasePerformancesTransformer):

    def __init__(
        self,
        features: List[str],
        cutoff_value: Optional[float] = None,
        quantile_cutoff: float = 0.93,
        excessive_multiplier: float = 0.8,
        reverse: bool = False,
    ):
        super().__init__(features=features)
        self.cutoff_value = cutoff_value
        self.excessive_multiplier = excessive_multiplier
        self.quantile_cutoff = quantile_cutoff
        self.reverse = reverse
        self._feature_cutoff_value = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for feature_name in self.features:
            if self.cutoff_value is None:
                if self.reverse:
                    self._feature_cutoff_value[feature_name] = df[
                        feature_name
                    ].quantile(1 - self.quantile_cutoff)
                else:
                    self._feature_cutoff_value[feature_name] = df[
                        feature_name
                    ].quantile(self.quantile_cutoff)
            else:
                self._feature_cutoff_value[feature_name] = self.cutoff_value

        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature_name in self.features:
            cutoff_value = self._feature_cutoff_value[feature_name]
            if self.reverse:
                df = df.assign(
                    **{
                        feature_name: lambda x: np.where(
                            x[feature_name] <= cutoff_value,
                            -(cutoff_value - df[feature_name])
                            * self.excessive_multiplier
                            + cutoff_value,
                            x[feature_name],
                        )
                    }
                )
            else:

                df = df.assign(
                    **{
                        feature_name: lambda x: np.where(
                            x[feature_name] >= cutoff_value,
                            (x[feature_name] - cutoff_value).clip(lower=0)
                            * self.excessive_multiplier
                            + cutoff_value,
                            x[feature_name],
                        )
                    }
                )

            df = df.assign(**{feature_name: df[feature_name].fillna(df[feature_name])})
        return df

    @property
    def features_out(self) -> list[str]:
        return self.features


class SymmetricDistributionTransformer(BasePerformancesTransformer):

    def __init__(
        self,
        features: List[str],
        granularity: Optional[list[str]] = None,
        skewness_allowed: float = 0.15,
        max_iterations: int = 50,
        min_excessive_multiplier: float = 0.04,
        prefix: str = "symmetric_",
    ):
        super().__init__(features=features)
        self.granularity = granularity
        self.skewness_allowed = skewness_allowed
        self.max_iterations = max_iterations
        self.min_excessive_multiplier = min_excessive_multiplier
        self.prefix = prefix

        self._diminishing_value_transformer = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.granularity:
            df = df.assign(
                __concat_granularity=df[self.granularity]
                .astype(str)
                .agg("_".join, axis=1)
            )

        for feature in self.features:
            if df[feature].min() == df[feature].max():
                raise ValueError(
                    f"SymmetricDistributionTransformer: {feature} has the same min and max value."
                )
            self._diminishing_value_transformer[feature] = {}
            if self.granularity:

                unique_values = df["__concat_granularity"].unique()
                for unique_value in unique_values:
                    rows = df[df["__concat_granularity"] == unique_value]

                    self._fit(
                        rows=rows, feature=feature, granularity_value=unique_value
                    )

            else:
                self._fit(rows=df, feature=feature, granularity_value=None)

        return self.transform(df)

    def _fit(
        self, rows: pd.DataFrame, feature: str, granularity_value: Optional[str]
    ) -> None:

        skewness = rows[feature].skew()
        excessive_multiplier = 0.8
        quantile_cutoff = 0.95

        iteration = 0
        while (
            abs(skewness) > self.skewness_allowed
            and len(rows) > 10
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.granularity:
            df = df.assign(
                __concat_granularity=df[self.granularity].apply(
                    lambda x: "_".join(x), axis=1
                )
            )

        for feature in self.features:
            out_feature = self.prefix + feature
            if self.granularity:

                unique_values = df["__concat_granularity"].unique()
                if len(unique_values) > 100:
                    logging.warning(
                        f"SymmetricDistributionTransformer: {feature} has more than 100 unique values."
                        f" This can lead to long runtimes. Consider setting a lower granularity"
                    )

                if len(self._diminishing_value_transformer[feature]) == 0:
                    df[out_feature] = df[feature]
                else:

                    for unique_value in unique_values:
                        rows = df[df["__concat_granularity"] == unique_value]

                        if unique_value in self._diminishing_value_transformer[feature]:
                            rows = self._diminishing_value_transformer[feature][
                                unique_value
                            ].transform(rows)

                        df.loc[
                            df["__concat_granularity"] == unique_value, out_feature
                        ] = rows[feature]

            else:
                if None in self._diminishing_value_transformer[feature]:
                    df = self._diminishing_value_transformer[feature][None].transform(
                        df
                    )

        if "__concat_granularity" in df.columns:
            df = df.drop(columns=["__concat_granularity"])
        return df

    @property
    def features_out(self) -> list[str]:
        return [self.prefix + feature for feature in self.features]


class GroupByTransformer(BasePerformancesTransformer):

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
