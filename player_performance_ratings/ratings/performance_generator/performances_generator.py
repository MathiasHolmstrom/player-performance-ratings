import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.ratings.performance_generator.performances_transformers import (
    SymmetricDistributionTransformer,
    MinMaxTransformer,
    PartialStandardScaler,
)


from player_performance_ratings.transformers.base_transformer import (
    BasePerformancesTransformer,
)


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


@dataclass
class Performance:
    name: str = "performance"
    weights: list[ColumnWeight] = field(default_factory=list)


def auto_create_pre_performance_transformations(
    pre_transformers: list[BasePerformancesTransformer],
    performances: list[Performance],
) -> list[BasePerformancesTransformer]:
    """
    Creates a list of transformers that ensure the performance column is generated in a way that makes sense for the rating model.
    Ensures columns aren't too skewed, scales them to similar ranges, ensure values are between 0 and 1,
    and then weights them according to the column_weights.
    """

    not_transformed_features = []
    transformed_features = []

    for performance in performances:
        not_transformed_features += [
            p.name
            for p in performance.weights
            if p.name not in not_transformed_features
        ]

    distribution_transformer = SymmetricDistributionTransformer(
        features=not_transformed_features, prefix=""
    )
    pre_transformers.append(distribution_transformer)

    all_features = transformed_features + not_transformed_features

    pre_transformers.append(
        PartialStandardScaler(
            features=all_features, ratio=1, max_value=9999, target_mean=0, prefix=""
        )
    )
    pre_transformers.append(MinMaxTransformer(features=all_features))

    return pre_transformers


class PerformancesGenerator:

    def __init__(
        self,
        performances: Union[list[Performance], Performance],
        transformers: Optional[list[BasePerformancesTransformer]] = None,
        auto_transform_performance: bool = True,
    ):

        self.performances = (
            performances if isinstance(performances, list) else [performances]
        )
        self.auto_transform_performance = auto_transform_performance
        self.original_transformers = (
            [copy.deepcopy(p) for p in transformers] if transformers else []
        )
        self.transformers = transformers or []
        if self.auto_transform_performance:
            self.transformers = auto_create_pre_performance_transformations(
                pre_transformers=self.transformers, performances=self.performances
            )

    def generate(self, df):

        if self.transformers:
            for transformer in self.transformers:
                df = transformer.fit_transform(df)

        for performance in self.performances:
            if self.transformers:
                max_idx = len(self.transformers) - 1
                column_weighs_mapping = {
                    col_weight.name: self.transformers[max_idx].features_out[idx]
                    for idx, col_weight in enumerate(performance.weights)
                }
            else:
                column_weighs_mapping = None

            df[performance.name] = self._weight_columns(
                df=df,
                performance_column_name=performance.name,
                col_weights=performance.weights,
                column_weighs_mapping=column_weighs_mapping,
            )

            if df[performance.name].isnull().any():
                logging.error(
                    f"df[{performance.name}] contains nan values. Make sure all column_names used in column_weights are imputed beforehand"
                )
                raise ValueError("performance contains nan values")

        return df

    def _weight_columns(
        self,
        df: pd.DataFrame,
        performance_column_name: str,
        col_weights: list[ColumnWeight],
        column_weighs_mapping: dict[str, str],
    ) -> pd.DataFrame:
        df = df.copy()
        df[f"__{performance_column_name}"] = 0

        df["sum_cols_weights"] = 0
        for column_weight in col_weights:
            df[f"weight__{column_weight.name}"] = column_weight.weight
            df.loc[df[column_weight.name].isna(), f"weight__{column_weight.name}"] = 0
            df.loc[df[column_weight.name].isna(), column_weight.name] = 0
            df["sum_cols_weights"] = (
                df["sum_cols_weights"] + df[f"weight__{column_weight.name}"]
            )

        drop_cols = ["sum_cols_weights", f"__{performance_column_name}"]
        for column_weight in col_weights:
            df[f"weight__{column_weight.name}"] / df["sum_cols_weights"]
            drop_cols.append(f"weight__{column_weight.name}")

        sum_weight = sum([w.weight for w in col_weights])

        for column_weight in col_weights:

            if column_weighs_mapping:
                feature_name = column_weighs_mapping[column_weight.name]
            else:
                feature_name = column_weight.name

            if column_weight.lower_is_better:
                df[f"__{performance_column_name}"] += (
                    df[f"weight__{column_weight.name}"]
                    / sum_weight
                    * (1 - df[feature_name])
                )
            else:
                df[f"__{performance_column_name}"] += (
                    df[f"weight__{column_weight.name}"] / sum_weight * df[feature_name]
                )

        return df[f"__{performance_column_name}"].clip(0, 1)

    @property
    def features_out(self) -> list[str]:
        return [c.name for c in self.performances]
