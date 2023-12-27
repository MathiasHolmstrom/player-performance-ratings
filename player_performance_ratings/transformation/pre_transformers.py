import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from player_performance_ratings.transformation.base_transformer import BaseTransformer


@dataclass
class ColumnWeight:
    name: str
    weight: float
    lower_is_better: bool = False


class ColumnsWeighter(BaseTransformer):

    def __init__(self,
                 weighted_column_name: str,
                 column_weights: list[ColumnWeight]
                 ):
        self.weighted_column_name = weighted_column_name
        self.column_weights = column_weights

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[f"__{self.weighted_column_name}"] = 0

        df['sum_cols_weights'] = 0
        for column_weight in self.column_weights:
            df[f'weight__{column_weight.name}'] = column_weight.weight
            df.loc[df[column_weight.name].isna(), f'weight__{column_weight.name}'] = 0
            df.loc[df[column_weight.name].isna(), column_weight.name] = 0
            df['sum_cols_weights'] = df['sum_cols_weights'] + df[f'weight__{column_weight.name}']

        drop_cols = ['sum_cols_weights', f"__{self.weighted_column_name}"]
        for column_weight in self.column_weights:
            df[f'weight__{column_weight.name}'] / df['sum_cols_weights']
            drop_cols.append(f'weight__{column_weight.name}')

        for column_weight in self.column_weights:

            if column_weight.lower_is_better:
                df[f"__{self.weighted_column_name}"] += df[f'weight__{column_weight.name}'] * (
                            1 - df[column_weight.name])
            else:
                df[f"__{self.weighted_column_name}"] += df[f'weight__{column_weight.name}'] * df[column_weight.name]

        df[self.weighted_column_name] = df[f"__{self.weighted_column_name}"]
        df = df.drop(columns=drop_cols)
        return df

    @property
    def features_created(self) -> list[str]:
        return [self.weighted_column_name]


class SklearnEstimatorImputer(BaseTransformer):

    def __init__(self, features: list[str], target_name: str, estimator: Optional[LGBMRegressor] = None, ):
        self.estimator = estimator or LGBMRegressor()
        self.features = features
        self.target_name = target_name

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.estimator.fit(df[self.features], df[self.target_name])
        df = df.assign(**{
            f'imputed_col_{self.target_name}': self.estimator.predict(df[self.features])
        })
        df[self.target_name] = df[self.target_name].fillna(df[f'imputed_col_{self.target_name}'])
        return df.drop(columns=[f'imputed_col_{self.target_name}'])

    @property
    def features_created(self) -> list[str]:
        return [self.target_name]


class SkLearnTransformerWrapper(BaseTransformer):

    def __init__(self, transformer, features: list[str]):
        self.transformer = transformer
        self.features = features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.features] = self.transformer.fit_transform(df[self.features])
        return df

    @property
    def features_created(self) -> list[str]:
        return self.features


class MinMaxTransformer(BaseTransformer):

    def __init__(self,
                 features: list[str],
                 quantile: float = 0.99,
                 allowed_mean_diff: Optional[float] = 0.01,
                 prefix: str = ""
                 ):
        self.features = features
        self.quantile = quantile
        self.allowed_mean_diff = allowed_mean_diff
        self.prefix = prefix
        if self.quantile < 0 or self.quantile > 1:
            raise ValueError("quantile must be between 0 and 1")

        self._feature_names_created = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature in self.features:
            max_value = df[feature].quantile(self.quantile)
            min_value = df[feature].quantile(1 - self.quantile)

            df[self.prefix + feature] = (df[feature] - min_value) / (max_value - min_value)
            df[self.prefix + feature].clip(0, 1, inplace=True)
            self._feature_names_created.append(self.prefix + feature)

            if self.allowed_mean_diff:
                reps = 0
                mean_value = df[self.prefix + feature].mean()
                while abs(0.5 - mean_value) > self.allowed_mean_diff:

                    if mean_value > 0.5:
                        df[self.prefix + feature] = df[self.prefix + feature] * (1 - self.allowed_mean_diff)
                    else:
                        df[self.prefix + feature] = df[self.prefix + feature] * (1 + self.allowed_mean_diff)

                    df[self.prefix + feature].clip(0, 1, inplace=True)
                    mean_value = df[self.prefix + feature].mean()

                    reps += 1
                    if reps > 100:
                        logging.warning(
                            f"MinMaxTransformer: {feature} mean value is {mean_value} after {reps} repetitions")
                        continue

        return df

    @property
    def features_created(self) -> list[str]:
        return self._feature_names_created


class DiminishingValueTransformer(BaseTransformer):

    def __init__(self,
                 features: List[str],
                 cutoff_value: Optional[float] = None,
                 quantile_cutoff: float = 0.93,
                 excessive_multiplier: float = 0.8,
                 reverse: bool = False,
                 ):
        self.features = features
        self.cutoff_value = cutoff_value
        self.excessive_multiplier = excessive_multiplier
        self.trained_count: int = 0
        self.quantile_cutoff = quantile_cutoff
        self.reverse = reverse

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for feature_name in self.features:
            if self.cutoff_value is None:
                cutoff_value = df[feature_name].quantile(self.quantile_cutoff)
            else:
                cutoff_value = self.cutoff_value

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

        df[self.features] = df[self.features].fillna(df[self.features])

        return df

    @property
    def features_created(self) -> list[str]:
        return self.features


class SymmetricDistributionTransformer(BaseTransformer):

    def __init__(self, features: List[str], granularity: Optional[list[str]] = None, skewness_allowed: float = 0.1,
                 max_iterations: int = 20):
        self.features = features
        self.granularity = granularity
        self.skewness_allowed = skewness_allowed
        self.max_iterations = max_iterations
        self._excessive_multiplier = 0.8
        self._quantile_cutoff = 0.95

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        for feature in self.features:
            if self.granularity:
                for column_granularity in self.granularity:
                    unique_values = df[column_granularity].unique()
                    for unique_value in unique_values:
                        rows = df[df[column_granularity] == unique_value]

                        rows = self._transform_rows(rows=rows, feature=feature)

                        condition = df[column_granularity] == unique_value
                        if isinstance(rows[feature], (list, np.ndarray, pd.Series)):
                            assert len(rows[
                                           feature]) == condition.sum(), "Length of rows[feature] must match the number of rows in condition"

                        df.loc[df[column_granularity] == unique_value, feature] = rows[feature]

            else:
                df = self._transform_rows(rows=df, feature=feature)
        return df

    def _transform_rows(self, rows: pd.DataFrame, feature: str) -> pd.DataFrame:
        skewness = rows[feature].skew()



        iteration = 0
        while abs(skewness) > self.skewness_allowed and len(
                rows) > 10 and iteration < self.max_iterations:

            if skewness < 0:
                reverse = True
            else:
                reverse = False
            diminishing_value_transformer = DiminishingValueTransformer(features=[feature],
                                                                        reverse=reverse,
                                                                        excessive_multiplier=self._excessive_multiplier,
                                                                        quantile_cutoff=self._quantile_cutoff)
            rows = diminishing_value_transformer.transform(rows)
            self._excessive_multiplier *= 0.96
            self._quantile_cutoff *= 0.993
            iteration += 1
            skewness = rows[feature].skew()

        return rows

    @property
    def features_created(self) -> list[str]:
        return self.features


class GroupByTransformer(BaseTransformer):

    def __init__(self,
                 features: list[str],
                 granularity: list[str],
                 agg_func: str = "mean",
                 prefix: str = "mean_",
                 ):
        self.features = features
        self.granularity = granularity
        self.agg_func = agg_func
        self.prefix = prefix
        self._feature_names_created = []
        for feature in self.features:
            self._feature_names_created.append(self.prefix + feature)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for idx, feature in enumerate(self.features):
            df = df.assign(
                **{self._feature_names_created[idx]: df.groupby(self.granularity)[feature].transform(self.agg_func)})
        return df

    @property
    def features_created(self) -> list[str]:
        return self._feature_names_created


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
        self.features = features
        self._feature_names_created = []
        for idx, predicted_feature in enumerate(self.predict_transformer.features_created):
            new_feature_name = f'{self.prefix}{self.features[idx]}'
            self._feature_names_created.append(new_feature_name)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.assign(__id=range(1, len(df) + 1))
        predicted_df = self.predict_transformer.transform(df)
        df = df.merge(predicted_df[self.predict_transformer.features_created + ['__id']], on="__id").drop(
            columns=["__id"])

        for idx, predicted_feature in enumerate(self.predict_transformer.features_created):
            new_feature_name = self._feature_names_created[idx]
            df = df.assign(**{new_feature_name: df[self.features[idx]] - df[predicted_feature]})
            df = df.drop(columns=[predicted_feature])

        return df

    @property
    def features_created(self) -> list[str]:
        return self._feature_names_created
