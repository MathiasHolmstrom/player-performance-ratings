import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw
from spforge.transformers.base_transformer import BaseTransformer
from spforge.transformers.fit_transformers import (
    PartialStandardScaler,
    MinMaxTransformer,
    SymmetricDistributionTransformer,
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


def create_performance_scalers_transformers(
        pre_transformers: list[BaseTransformer],
        features: list[str],
) -> list[BaseTransformer]:
    """
    Creates a list of transformers that ensure the performance column is generated in a way that makes sense for the rating model.
    Ensures columns aren't too skewed, scales them to similar ranges, ensure values are between 0 and 1,
    and then weights them according to the column_weights.
    """

    transformed_features = []
    not_transformed_features = features or []

    distribution_transformer = SymmetricDistributionTransformer(
        features=not_transformed_features
    )
    pre_transformers.append(distribution_transformer)

    all_features = transformed_features + not_transformed_features

    pre_transformers.append(
        PartialStandardScaler(
            features=all_features,
            ratio=1,
            max_value=9999,
            target_mean=0,
        )
    )
    pre_transformers.append(MinMaxTransformer(features=all_features))

    return pre_transformers


class PerformanceManager(BaseTransformer):

    def __init__(self,
                 features: list[str],
                 transformers: Optional[list[BaseTransformer]] = None,
                 scale_performance: bool = True,
                 prefix: str = "performance__",
                 returned_all_transformed_features: bool = False,
                 ):
        self.returned_all_transformed_features = returned_all_transformed_features
        self.prefix = prefix

        self.scale_performance = scale_performance
        self.original_transformers = (
            [copy.deepcopy(p) for p in transformers] if transformers else []
        )
        self.transformers = transformers or []
        if self.scale_performance:
            self.transformers = create_performance_scalers_transformers(
                pre_transformers=self.transformers,
                features=features,
            )

        super().__init__(features=features, features_out=self.features_out)



    @nw.narwhalify
    def fit_transform(self, df: FrameT) -> IntoFrameT:
        input_cols = df.columns

        for transformer in self.transformers:
            if self.prefix:
                for feature in transformer.features:
                    ori_feature_name = feature.removeprefix(
                        self.prefix
                    )
                    if feature not in df.columns:
                        df = df.with_columns(
                            nw.col(ori_feature_name).alias(feature)
                        )
            df = nw.from_native(transformer.fit_transform(df))

        df = df.with_columns(
            nw.col(self.transformers[-1].features_out[0]).alias(self.performance_column)
        )

        return df.select(list(set([*input_cols, *self.features_out])))

    @nw.narwhalify
    def transform(self, df: FrameT) -> IntoFrameT:
        input_cols = df.columns
        for transformer in self.transformers:
            if self.prefix:
                for feature in transformer.features:
                    ori_feature_name = feature.removeprefix(
                        self.prefix
                    )
                    if feature not in df.columns:
                        df = df.with_columns(
                            nw.col(ori_feature_name).alias(feature)
                        )
            df = nw.from_native(transformer.transform(df))

        df = df.with_columns(
            nw.col(self.transformers[-1].features_out[0]).alias(self.performance_column)
        )

        return df.select(list(set([*input_cols, *self.features_out])))


    @property
    def features_out(self) -> list[str]:
        if self.returned_all_transformed_features:
            return list(set(
                [self.performance_column] +
                list(
                    set(
                        [
                            feature
                            for t in self.transformers
                            for feature in t.features
                            if self.prefix
                        ]
                    )
                )
            )
            )

        return [self.performance_column]


    @property
    def performance_column(self) -> str:
        return self.prefix + self.transformers[-1].features_out[0]


class PerformanceWeightsManager(PerformanceManager):

    def __init__(
            self,
            weights: list[ColumnWeight],
            transformers: Optional[list[BaseTransformer]] = None,
            scale_performance: bool = True,
            prefix: str = "performance__",
            returned_all_transformed_features: bool = False,
    ):

        features = [p.name for p in weights]
        super().__init__(
            features=features,
            transformers=transformers,
            scale_performance=scale_performance,
            prefix=prefix,
            returned_all_transformed_features=returned_all_transformed_features,
        )
        self.weights = weights
        for weight in self.weights:
            if self.prefix:
                weight.name = f"{self.prefix}{weight.name}"

    @nw.narwhalify
    def fit_transform(self, df: FrameT) -> IntoFrameT:
        input_cols = df.columns

        if self.transformers:
            for transformer in self.transformers:
                if self.prefix:
                    for feature in transformer.features:
                        ori_feature_name = feature.removeprefix(
                            self.prefix
                        )
                        if feature not in df.columns:
                            df = df.with_columns(
                                nw.col(ori_feature_name).alias(feature)
                            )
                df = transformer.fit_transform(df)

        df = nw.from_native(df)

        if self.transformers:
            max_idx = len(self.transformers) - 1
            column_weighs_mapping = {
                self.performance_column: self.transformers[max_idx].features_out[idx]
                for idx, col_weight in enumerate(self.weights)
            }
        else:
            column_weighs_mapping = None

        df = self._weight_columns(
            df=df,
            performance_column_name=self.performance_column,
            col_weights=self.weights,
            column_weighs_mapping=column_weighs_mapping,
        )

        if len(df.filter(nw.col(self.performance_column).is_null())) > 0 or len(
                df.filter(nw.col(self.performance_column).is_finite())
        ) != len(df):
            logging.error(
                f"df[{self.performance_column}] contains nan values. Make sure all column_names used in column_weights are imputed beforehand"
            )
            raise ValueError("performance contains nan values")

        return df.select(list(set([*input_cols, *self.features_out])))

    def _weight_columns(
            self,
            df: FrameT,
            performance_column_name: str,
            col_weights: list[ColumnWeight],
            column_weighs_mapping: dict[str, str],
    ) -> FrameT:
        df = df.with_columns(nw.lit(0).alias(self.performance_column))

        df = df.with_columns(nw.lit(0).alias("sum_cols_weights"))

        for column_weight in col_weights:
            weight_col = f"weight__{column_weight.name}"
            feature_col = column_weight.name

            df = df.with_columns(
                nw.when(
                    (nw.col(feature_col).is_null()) & (~nw.col(feature_col).is_finite())
                )
                .then(0)
                .otherwise(column_weight.weight)
                .alias(weight_col)
            )

            df = df.with_columns(
                nw.when(
                    (nw.col(feature_col).is_null()) & (~nw.col(feature_col).is_finite())
                )
                .then(0)
                .otherwise(nw.col(feature_col))
                .alias(feature_col)
            )

            df = df.with_columns(
                (nw.col("sum_cols_weights") + nw.col(weight_col)).alias(
                    "sum_cols_weights"
                )
            )

        for column_weight in col_weights:
            df = df.with_columns(
                (
                        nw.col(f"weight__{column_weight.name}") / nw.col("sum_cols_weights")
                ).alias(f"weight__{column_weight.name}")
            )

        sum_weight = sum([w.weight for w in col_weights])

        for column_weight in col_weights:
            weight_col = f"weight__{column_weight.name}"
            feature_col = column_weight.name
            feature_name = (
                column_weighs_mapping.get(feature_col, feature_col)
                if column_weighs_mapping
                else feature_col
            )

            if column_weight.lower_is_better:
                df = df.with_columns(
                    nw.col(self.performance_column)
                    + (
                        (
                                nw.col(weight_col) / sum_weight * (1 - nw.col(feature_name))
                        ).alias(self.performance_column)
                    )
                )
            else:
                df = df.with_columns(
                    nw.col(self.performance_column)
                    + (
                        (nw.col(weight_col) / sum_weight * nw.col(feature_name)).alias(
                            self.performance_column
                        )
                    )
                )

        return df.with_columns(
            nw.col(f"__{performance_column_name}")
            .clip(0, 1)
            .alias(performance_column_name)
        )
