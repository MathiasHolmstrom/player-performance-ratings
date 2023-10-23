from dataclasses import dataclass
from typing import Optional

import pandas as pd
from lightgbm import LGBMRegressor

from player_performance_ratings.transformers.base_transformer import BaseTransformer


@dataclass
class ColumnWeight:
    name: str
    weight: float
    is_negative: bool = False


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

            if column_weight.is_negative:
                df[self.weighted_column_name] += df[f'weight__{column_weight.name}'] * -df[column_weight.name]
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

    def __init__(self, features: list[str], quantile: float = 0.99, prefix: str = ""):
        self.features = features
        self.quantile = quantile
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

        return df
