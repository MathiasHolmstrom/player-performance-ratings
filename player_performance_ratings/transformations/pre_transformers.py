import logging
import math
from dataclasses import dataclass
from typing import Optional,  List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from player_performance_ratings.transformations.base_transformer import BaseTransformer


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
        df[self.weighted_column_name] = 0

        df['sum_cols_weights'] = 0
        for column_weight in self.column_weights:
            df[f'weight__{column_weight.name}'] = column_weight.weight
            df.loc[df[column_weight.name].isna(), f'weight__{column_weight.name}'] = 0
            df.loc[df[column_weight.name].isna(), column_weight.name] = 0
            df['sum_cols_weights'] = df['sum_cols_weights'] + df[f'weight__{column_weight.name}']

        drop_cols = ['sum_cols_weights']
        for column_weight in self.column_weights:
            df[f'weight__{column_weight.name}'] / df['sum_cols_weights']
            drop_cols.append(f'weight__{column_weight.name}')

        for column_weight in self.column_weights:

            if column_weight.lower_is_better:
                df[self.weighted_column_name] += df[f'weight__{column_weight.name}'] * (1 - df[column_weight.name])
            else:
                df[self.weighted_column_name] += df[f'weight__{column_weight.name}'] * df[column_weight.name]
        df = df.drop(columns=drop_cols)
        return df


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


class SkLearnTransformerWrapper(BaseTransformer):

    def __init__(self, transformer, features: list[str]):
        self.transformer = transformer
        self.features = features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.features] = self.transformer.fit_transform(df[self.features])
        return df


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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature in self.features:
            max_value = df[feature].quantile(self.quantile)
            min_value = df[feature].quantile(1 - self.quantile)

            df[self.prefix + feature] = (df[feature] - min_value) / (max_value - min_value)
            df[self.prefix + feature].clip(0, 1, inplace=True)

            if self.allowed_mean_diff:
                reps = 0
                mean_value = df[self.prefix + feature].mean()
                while abs(0.5 - mean_value) > self.allowed_mean_diff:

                    if mean_value > 0.5:
                        df[self.prefix + feature] = df[self.prefix + feature] * (1- self.allowed_mean_diff)
                    else:
                        df[self.prefix + feature] = df[self.prefix + feature] * (1+ self.allowed_mean_diff)

                    df[self.prefix + feature].clip(0, 1, inplace=True)
                    mean_value = df[self.prefix + feature].mean()

                    reps +=1
                    if reps > 100:
                        logging.warning(f"MinMaxTransformer: {feature} mean value is {mean_value} after {reps} repetitions")
                        continue

        return df


class DiminishingValueTransformer(BaseTransformer):

    def __init__(self,
                 features: List[str],
                 cutoff_value: Optional[float] = None,
                 quantile_cutoff: float = 0.95,
                 excessive_multiplier: float = 0.8,

                 ):
        self.features = features
        self.cutoff_value = cutoff_value
        self.excessive_multiplier = excessive_multiplier
        self.trained_count: int = 0
        self.quantile_cutoff = quantile_cutoff

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for feature_name in self.features:
            if self.cutoff_value is None:
                cutoff_value = df[feature_name].quantile(self.quantile_cutoff)
            else:
                cutoff_value = self.cutoff_value

            df = df.assign(feature_name=lambda x: np.where(
                x[feature_name] >= cutoff_value,
                (x[feature_name] - cutoff_value).clip(lower=0) * self.excessive_multiplier + cutoff_value,
                np.where(
                    x[feature_name] < -cutoff_value,
                    (-x[feature_name] - cutoff_value).clip(upper=0) * self.excessive_multiplier - cutoff_value,
                    x[feature_name]
                )
            ))

        df[self.features] = df[self.features].fillna(df[self.features])

        return df
