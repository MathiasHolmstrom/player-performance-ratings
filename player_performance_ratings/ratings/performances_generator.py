import logging
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.transformation.pre_transformers import \
    SymmetricDistributionTransformer, MinMaxTransformer, PartialStandardScaler

PartialStandardScaler

from player_performance_ratings import ColumnNames

from player_performance_ratings.transformation.base_transformer import BaseTransformer


@dataclass
class ColumnWeight:
    name: str
    weight: float
    lower_is_better: bool = False

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Weight must be positive")
        if self.weight > 1:
            raise ValueError("Weight must be less than 1")


def auto_create_pre_performance_transformations(
        pre_transformations: list[BaseTransformer],
        column_weights: Union[list[list[ColumnWeight]], list[ColumnWeight]],
        column_names: list[ColumnNames],
) -> list[BaseTransformer]:
    """
    Creates a list of transformers that ensure the performance column is generated in a way that makes sense for the rating model.
    Ensures columns aren't too skewed, scales them to similar ranges, ensure values are between 0 and 1,
    and then weights them according to the column_weights.
    """

    if not isinstance(column_weights[0], list):
        column_weights = [column_weights]


    contains_not_position = True if any([c.position is None for c in column_names]) else False

    not_transformed_features = []
    transformed_features = []


    for idx, col_weights in enumerate(column_weights):
        not_transformed_features += [c.name for c in column_weights[idx] if c.name not in not_transformed_features]


        if column_names[idx].position is not None:
            granularity = [column_names[idx].position]
            features_in = []
            for column_weight in col_weights:
                features_in.append(column_weight.name)
                not_transformed_features.remove(column_weight.name)

            distribution_transformer = SymmetricDistributionTransformer(
                features=features_in,
                granularity=granularity, prefix="")
            transformed_features += [distribution_transformer.prefix + f for f in distribution_transformer.features]
            pre_transformations.append(distribution_transformer)


    if contains_not_position:
        distribution_transformer = SymmetricDistributionTransformer(features=not_transformed_features, prefix="")
        pre_transformations.append(distribution_transformer)

    all_features = transformed_features + not_transformed_features

    pre_transformations.append(PartialStandardScaler(features=all_features, ratio=1, max_value=9999, target_mean=0, prefix=""))
    pre_transformations.append(MinMaxTransformer(features=all_features))

    return pre_transformations


class PerformancesGenerator():

    def __init__(self,
                 column_weights: Union[list[list[ColumnWeight]], list[ColumnWeight]],
                 column_names: Union[list[ColumnNames], ColumnNames],
                 pre_transformations: Optional[list[BaseTransformer]] = None,
                 auto_transform_performance: bool = True
                 ):
        self.column_names = column_names if isinstance(column_names, list) else [column_names]
        self.column_weights = column_weights if isinstance(column_weights[0], list) else [column_weights]
        if len(self.column_names) > len(self.column_weights):
            raise ValueError("There cannot be more column names items than column weights items")
        elif len(self.column_names) < len(self.column_weights):
            for _ in range(len(self.column_weights) - len(self.column_names)):
                self.column_names.append(self.column_names[0])
        self.auto_transform_performance = auto_transform_performance

        self.pre_transformations = pre_transformations or []
        if self.auto_transform_performance:
            self.pre_transformations = auto_create_pre_performance_transformations(
                pre_transformations=self.pre_transformations, column_weights=column_weights,
                column_names=self.column_names)

    def generate(self, df):

        if self.pre_transformations:
            for pre_transformation in self.pre_transformations:
                df = pre_transformation.fit_transform(df)

        for idx, col_name in enumerate(self.column_names):
            if self.pre_transformations:
                max_idx = len(self.pre_transformations) - 1
                column_weighs_mapping = {col_weight.name: self.pre_transformations[max_idx].features_out[idx] for
                                         idx, col_weight in enumerate(self.column_weights[idx])}
            else:
                column_weighs_mapping = None

            df[col_name.performance] = self._weight_columns(df=df, col_name=col_name,
                                                            col_weights=self.column_weights[idx],
                                                            column_weighs_mapping=column_weighs_mapping)

            if df[col_name.performance].isnull().any():
                logging.error(
                    f"df[{col_name.performance}] contains nan values. Make sure all column_names used in column_weights are imputed beforehand")
                raise ValueError("performance contains nan values")

        return df

    def _weight_columns(self,
                        df: pd.DataFrame,
                        col_name: ColumnNames,
                        col_weights: list[ColumnWeight],
                        column_weighs_mapping: dict[str, str]
                        ) -> pd.DataFrame:
        df = df.copy()
        df[f"__{col_name.performance}"] = 0

        df['sum_cols_weights'] = 0
        for column_weight in col_weights:
            df[f'weight__{column_weight.name}'] = column_weight.weight
            df.loc[df[column_weight.name].isna(), f'weight__{column_weight.name}'] = 0
            df.loc[df[column_weight.name].isna(), column_weight.name] = 0
            df['sum_cols_weights'] = df['sum_cols_weights'] + df[f'weight__{column_weight.name}']

        drop_cols = ['sum_cols_weights', f"__{col_name.performance}"]
        for column_weight in col_weights:
            df[f'weight__{column_weight.name}'] / df['sum_cols_weights']
            drop_cols.append(f'weight__{column_weight.name}')

        sum_weight = sum([w.weight for w in col_weights])

        for column_weight in col_weights:

            if column_weighs_mapping:
                feature_name = column_weighs_mapping[column_weight.name]
            else:
                feature_name = column_weight.name

            if column_weight.lower_is_better:
                df[f"__{col_name.performance}"] += df[f'weight__{column_weight.name}']/sum_weight * (
                        1 - df[feature_name])
            else:
                df[f"__{col_name.performance}"] += df[f'weight__{column_weight.name}']/sum_weight * df[feature_name]

        return df[f"__{col_name.performance}"]

    @property
    def features_out(self) -> list[str]:
        return [c.performance for c in self.column_names]
