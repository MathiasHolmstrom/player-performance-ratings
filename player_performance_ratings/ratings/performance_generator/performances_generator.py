import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw
import pandas as pd
from player_performance_ratings.transformers.base_transformer import BaseTransformer
from player_performance_ratings.transformers.performances_transformers import PartialStandardScaler, MinMaxTransformer, \
    SymmetricDistributionTransformer


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
        pre_transformers: list[BaseTransformer],
        performances: list[Performance],
) -> list[BaseTransformer]:
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
            transformers: Optional[list[BaseTransformer]] = None,
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

    @nw.narwhalify
    def generate(self, df: FrameT) -> IntoFrameT:
        input_cols = df.columns
        if self.transformers:
            for transformer in self.transformers:
                df = transformer.fit_transform(df)

        df = nw.from_native(df)
        for performance in self.performances:
            if self.transformers:
                max_idx = len(self.transformers) - 1
                column_weighs_mapping = {
                    col_weight.name: self.transformers[max_idx].features_out[idx]
                    for idx, col_weight in enumerate(performance.weights)
                }
            else:
                column_weighs_mapping = None

            df = self._weight_columns(
                df=df,
                performance_column_name=performance.name,
                col_weights=performance.weights,
                column_weighs_mapping=column_weighs_mapping
            )

            if len(df.filter(nw.col(performance.name).is_null())) > 0 or len(df.filter(
                    nw.col(performance.name).is_finite())) != len(df):
                logging.error(
                    f"df[{performance.name}] contains nan values. Make sure all column_names used in column_weights are imputed beforehand"
                )
                raise ValueError("performance contains nan values")

        return df.select(*input_cols, *self.features_out).to_native()

    def _weight_columns(
            self,
            df: FrameT,
            performance_column_name: str,
            col_weights: list[ColumnWeight],
            column_weighs_mapping: dict[str, str],
    ) -> FrameT:
        df = df.with_columns(
            nw.lit(0).alias(f"__{performance_column_name}")
        )

        df = df.with_columns(
            nw.lit(0).alias("sum_cols_weights")
        )

        for column_weight in col_weights:
            weight_col = f"weight__{column_weight.name}"
            feature_col = column_weight.name

            df = df.with_columns(
                nw.when((nw.col(feature_col).is_null()) & (~nw.col(feature_col).is_finite()))
                .then(0)
                .otherwise(column_weight.weight)
                .alias(weight_col)
            )

            df = df.with_columns(
                nw.when((nw.col(feature_col).is_null()) & (~nw.col(feature_col).is_finite()))
                .then(0)
                .otherwise(nw.col(feature_col))
                .alias(feature_col)
            )

            df = df.with_columns(
                (nw.col("sum_cols_weights") + nw.col(weight_col)).alias("sum_cols_weights")
            )

        for column_weight in col_weights:
            df = df.with_columns(
                (nw.col(f"weight__{column_weight.name}") / nw.col("sum_cols_weights")).alias(
                    f"weight__{column_weight.name}")
            )

        sum_weight = sum([w.weight for w in col_weights])

        for column_weight in col_weights:
            weight_col = f"weight__{column_weight.name}"
            feature_col = column_weight.name
            feature_name = column_weighs_mapping.get(feature_col, feature_col) if column_weighs_mapping else feature_col

            if column_weight.lower_is_better:
                df = df.with_columns(
                    nw.col(f"__{performance_column_name}") + (
                        (nw.col(weight_col) / sum_weight * (1 - nw.col(feature_name))).alias(
                            f"__{performance_column_name}")
                    )
                )
            else:
                df = df.with_columns(nw.col(f"__{performance_column_name}") + (
                    (nw.col(weight_col) / sum_weight * nw.col(feature_name)).alias(f"__{performance_column_name}")
                ))

        return df.with_columns(
            nw.col(f"__{performance_column_name}").clip(0, 1).alias(performance_column_name)
        )

    @property
    def features_out(self) -> list[str]:
        return [c.name for c in self.performances]
