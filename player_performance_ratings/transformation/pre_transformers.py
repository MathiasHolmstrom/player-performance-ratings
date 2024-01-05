import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from player_performance_ratings.transformation.base_transformer import BaseTransformer


class ConvertDataFrameToCategoricalTransformer(BaseTransformer):

    def __init__(self, features: list[str]):
        super().__init__(features=features)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            df = df.assign(**{feature: df[feature].astype('category')})
        return df

    @property
    def features_out(self) -> list[str]:
        return self.features



class SklearnEstimatorImputer(BaseTransformer):

    def __init__(self, features: list[str], target_name: str, estimator: Optional[LGBMRegressor] = None):
        super().__init__(features=features)
        self.estimator = estimator or LGBMRegressor()
        self.target_name = target_name

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.estimator.fit(df[self.features], df[self.target_name])
        df = df.assign(**{
            f'imputed_col_{self.target_name}': self.estimator.predict(df[self.features])
        })
        df[self.target_name] = df[self.target_name].fillna(df[f'imputed_col_{self.target_name}'])
        return df.drop(columns=[f'imputed_col_{self.target_name}'])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{
            f'imputed_col_{self.target_name}': self.estimator.predict(df[self.features])
        })
        df[self.target_name] = df[self.target_name].fillna(df[f'imputed_col_{self.target_name}'])
        return df.drop(columns=[f'imputed_col_{self.target_name}'])

    @property
    def features_out(self) -> list[str]:
        return [self.target_name]


class SkLearnTransformerWrapper(BaseTransformer):

    def __init__(self, transformer, features: list[str]):
        self.transformer = transformer
        super().__init__(features=features)
        self._features_out = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            transformed_values = self.transformer.fit_transform(df[self.features]).toarray()
        except AttributeError:
            transformed_values = self.transformer.fit_transform(df[self.features])
            if isinstance(transformed_values, pd.DataFrame):
                transformed_values = transformed_values.to_numpy()

        self._features_out = self.transformer.get_feature_names_out().tolist()

        return df.assign(
            **{self._features_out[idx]: transformed_values[:, idx] for idx in range(len(self._features_out))})

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            transformed_values = self.transformer.transform(df[self.features]).toarray()
        except AttributeError:
            transformed_values = self.transformer.transform(df[self.features])
            if isinstance(transformed_values, pd.DataFrame):
                transformed_values = transformed_values.to_numpy()
        return df.assign(
            **{self._features_out[idx]: transformed_values[:, idx] for idx in range(len(self._features_out))})

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class MinMaxTransformer(BaseTransformer):

    def __init__(self,
                 features: list[str],
                 quantile: float = 0.99,
                 allowed_mean_diff: Optional[float] = 0.02,
                 prefix: str = ""
                 ):
        super().__init__(features=features)
        self.quantile = quantile
        self.allowed_mean_diff = allowed_mean_diff
        self.prefix = prefix
        self._original_mean_values = {}
        self._mean_aligning_iterations = 0
        self._min_values = {}
        self._max_values = {}

        if self.quantile < 0 or self.quantile > 1:
            raise ValueError("quantile must be between 0 and 1")

        self._features_out = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._mean_aligning_iterations = 0
        df = df.copy()
        for feature in self.features:
            self._min_values[feature] = df[feature].quantile(1 - self.quantile)
            self._max_values[feature] = df[feature].quantile(self.quantile)

            df[self.prefix + feature] = (df[feature] - self._min_values[feature]) / (
                    self._max_values[feature] - self._min_values[feature])
            df[self.prefix + feature].clip(0, 1, inplace=True)
            self._features_out.append(self.prefix + feature)

            if self.allowed_mean_diff:

                mean_value = df[self.prefix + feature].mean()
                self._original_mean_values[feature] = mean_value

                while abs(0.5 - mean_value) > self.allowed_mean_diff:

                    if mean_value > 0.5:
                        df[self.prefix + feature] = df[self.prefix + feature] * (1 - self.allowed_mean_diff)
                    else:
                        df[self.prefix + feature] = df[self.prefix + feature] * (1 + self.allowed_mean_diff)

                    df[self.prefix + feature].clip(0, 1, inplace=True)
                    mean_value = df[self.prefix + feature].mean()

                    self._mean_aligning_iterations += 1
                if self._mean_aligning_iterations > 100:
                    raise ValueError(
                        f"MinMaxTransformer: {feature} mean value is {mean_value} after {self._mean_aligning_iterations} repetitions."
                        f"This is above the allowed mean difference of {self.allowed_mean_diff}."
                        f" It is recommended to use DiminishingValueTransformer or SymmetricDistributionTransformer before MinMaxTransformer."
                        f"If positions are known use NetOverPredictedTransformer before DiminishingValueTransformer or SymmetricDistributionTransformer.")


        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            df[self.prefix + feature] = (df[feature] - self._min_values[feature]) / (
                    self._max_values[feature] - self._min_values[feature])
            df[self.prefix + feature].clip(0, 1, inplace=True)

            for _ in range(self._mean_aligning_iterations):

                if self._original_mean_values[feature] > 0.5:
                    df[self.prefix + feature] = df[self.prefix + feature] * (1 - self.allowed_mean_diff)
                else:
                    df[self.prefix + feature] = df[self.prefix + feature] * (1 + self.allowed_mean_diff)

                df[self.prefix + feature].clip(0, 1, inplace=True)

        return df

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class DiminishingValueTransformer(BaseTransformer):

    def __init__(self,
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
                self._feature_cutoff_value[feature_name] = df[feature_name].quantile(self.quantile_cutoff)
            else:
                self._feature_cutoff_value[feature_name] = self.cutoff_value

        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature_name in self.features:
            cutoff_value = self._feature_cutoff_value[feature_name]
            if self.reverse:
                cutoff_value = 1 - cutoff_value
                df = df.assign(**{
                    feature_name: lambda x: np.where(
                        x[feature_name] <= cutoff_value,
                        - (cutoff_value - df[feature_name]) * self.excessive_multiplier + cutoff_value,
                        x[feature_name]

                    )
                })
            else:

                df = df.assign(**{
                    feature_name: lambda x: np.where(
                        x[feature_name] >= cutoff_value,
                        (x[feature_name] - cutoff_value).clip(lower=0) * self.excessive_multiplier + cutoff_value,
                        x[feature_name]
                    )
                })

            df = df.assign(**{feature_name: df[feature_name].fillna(df[feature_name])})
        return df

    @property
    def features_out(self) -> list[str]:
        return self.features


class SymmetricDistributionTransformer(BaseTransformer):

    def __init__(self,
                 features: List[str],
                 granularity: Optional[list[str]] = None,
                 skewness_allowed: float = 0.15,
                 max_iterations: int = 20,
                 prefix: str = ""
                 ):
        super().__init__(features=features)
        self.granularity = granularity
        self.skewness_allowed = skewness_allowed
        self.max_iterations = max_iterations
        self.prefix = prefix

        self._diminishing_value_transformer = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.granularity:
            df = df.assign(__concat_granularity=df[self.granularity].apply(lambda x: "_".join(x), axis=1))

        for feature in self.features:
            self._diminishing_value_transformer[feature] = {}
            if self.granularity:

                unique_values = df["__concat_granularity"].unique()
                for unique_value in unique_values:
                    rows = df[df["__concat_granularity"] == unique_value]

                    self._fit(rows=rows, feature=feature, granularity_value=unique_value)

            else:
                self._fit(rows=df, feature=feature, granularity_value=None)

        return self.transform(df)

    def _fit(self, rows: pd.DataFrame, feature: str, granularity_value: Optional[str]) -> None:
        skewness = rows[feature].skew()
        excessive_multiplier = 0.8
        quantile_cutoff = 0.95

        iteration = 0
        while abs(skewness) > self.skewness_allowed and len(
                rows) > 10 and iteration < self.max_iterations:

            if skewness < 0:
                reverse = True
            else:
                reverse = False


            self._diminishing_value_transformer[feature][granularity_value] = DiminishingValueTransformer(
                features=[feature],
                reverse=reverse,
                excessive_multiplier=excessive_multiplier,
                quantile_cutoff=quantile_cutoff)
            transformed_rows = self._diminishing_value_transformer[feature][granularity_value].fit_transform(rows)
            excessive_multiplier *= 0.95
            quantile_cutoff *= 0.99
            iteration += 1
            skewness = transformed_rows[feature].skew()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.granularity:
            df = df.assign(__concat_granularity=df[self.granularity].apply(lambda x: "_".join(x), axis=1))

        for feature in self.features:
            out_feature = self.prefix + feature
            if self.granularity:

                unique_values = df["__concat_granularity"].unique()
                for unique_value in unique_values:
                    rows = df[df["__concat_granularity"] == unique_value]
                    if unique_value in self._diminishing_value_transformer[feature]:
                        rows = self._diminishing_value_transformer[feature][unique_value].transform(rows)
                        df.loc[df["__concat_granularity"] == unique_value, out_feature] = rows[feature]

            else:
                if None in self._diminishing_value_transformer[feature]:
                    df = self._diminishing_value_transformer[feature][None].transform(df)

        return df

    @property
    def features_out(self) -> list[str]:
        return [self.prefix + feature for feature in self.features]


class GroupByTransformer(BaseTransformer):

    def __init__(self,
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
        self._grouped = (df
                         .groupby(self.granularity)[self.features]
                         .agg(self.agg_func)
                         .reset_index()
                         .rename(columns={feature: self.prefix + feature for feature in self.features})
                         )

        return self.transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(self._grouped, on=self.granularity, how='left')

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class NetOverPredictedTransformer(BaseTransformer):

    def __init__(self,
                 features: list[str],
                 granularity: list[str],
                 predict_transformer: Optional[BaseTransformer] = None,
                 prefix: str = "",
                 ):
        self.granularity = granularity
        self.predict_transformer = predict_transformer or GroupByTransformer(features=features, granularity=granularity,
                                                                             agg_func='mean')
        self.prefix = prefix
        super().__init__(features=features)
        self._features_out = []
        for idx, predicted_feature in enumerate(self.predict_transformer.features_out):
            new_feature_name = f'{self.prefix}{self.features[idx]}'
            self._features_out.append(new_feature_name)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.assign(__id=range(1, len(df) + 1))
        _ = self.predict_transformer.fit_transform(df)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        predicted_df = self.predict_transformer.transform(df)
        df = df.merge(predicted_df[self.predict_transformer.features_out + ['__id']], on="__id").drop(
            columns=["__id"])

        return self._add_net_over_predicted(df=df)

    def _add_net_over_predicted(self, df: pd.DataFrame) -> pd.DataFrame:
        for idx, predicted_feature in enumerate(self.predict_transformer.features_out):
            new_feature_name = self._features_out[idx]
            df = df.assign(**{new_feature_name: df[self.features[idx]] - df[predicted_feature]})
            df = df.drop(columns=[predicted_feature])

        return df

    @property
    def features_out(self) -> list[str]:
        return self._features_out
