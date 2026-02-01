import copy
import logging
from dataclasses import dataclass
from typing import Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from sklearn.base import BaseEstimator, TransformerMixin

from spforge.performance_transformers._performances_transformers import (
    MinMaxTransformer,
    NarwhalsFeatureTransformer,
    PartialStandardScaler,
    QuantilePerformanceScaler,
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


TransformerName = Literal[
    "partial_standard_scaler", "symmetric", "min_max", "partial_standard_scaler_mean0.5"
]


def create_performance_scalers_transformers(
    transformer_names: list[TransformerName],
    pre_transformers: list[NarwhalsFeatureTransformer],
    features: list[str],
    prefix: str,
) -> list[NarwhalsFeatureTransformer]:
    if not transformer_names:
        return pre_transformers

    all_features = [prefix + f for f in features]

    for transformer_name in transformer_names:
        if transformer_name == "symmetric":
            t = SymmetricDistributionTransformer(features=all_features, prefix="")
        elif transformer_name == "partial_standard_scaler":
            t = PartialStandardScaler(
                features=all_features,
                ratio=1,
                max_value=9999,
                target_mean=0,
                prefix="",
            )
        elif transformer_name == "partial_standard_scaler_mean0.5":
            t = PartialStandardScaler(
                features=all_features,
                ratio=1,
                max_value=9999,
                target_mean=0.5,
                prefix="",
            )
        elif transformer_name == "min_max":
            t = MinMaxTransformer(features=all_features, prefix="")
        else:
            raise ValueError(f"Unknown transformer_name: {transformer_name}")

        pre_transformers.append(t)

        # Chain: next transformer consumes the previous transformer's outputs
        all_features = t.features_out

    return pre_transformers


class PerformanceManager(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features: list[str],
        performance_column: str,
        transformer_names: list[TransformerName] | None = None,
        custom_transformers: list[NarwhalsFeatureTransformer] | None = None,
        prefix: str = "performance__",
        min_value: float = -0.02,
        max_value: float = 1.02,
        zero_inflation_threshold: float = 0.15,
    ):
        self.features = features
        self.prefix = prefix
        # Store whether user explicitly disabled transformers (passed empty list)
        self._user_disabled_transformers = transformer_names is not None and len(transformer_names) == 0
        self.transformer_names = transformer_names or [
            "symmetric",
            "partial_standard_scaler",
            "min_max",
        ]
        self.custom_transformers = custom_transformers or []
        self.original_transformers = [copy.deepcopy(p) for p in self.custom_transformers]
        self.ori_performance_column = performance_column
        self.performance_column = self.prefix + performance_column
        self.min_value = min_value
        self.max_value = max_value
        self.zero_inflation_threshold = zero_inflation_threshold

        self.transformers = create_performance_scalers_transformers(
            transformer_names=self.transformer_names,
            pre_transformers=self.custom_transformers,
            features=self.features,
            prefix=self.prefix,
        )
        self._using_quantile_scaler = False

    @nw.narwhalify
    def fit(self, df: IntoFrameT, y=None):
        # Check for zero-inflated distributions and swap to quantile scaler if needed
        # Only apply when user hasn't explicitly disabled transformers (passed empty list)
        if self.zero_inflation_threshold > 0 and not self._user_disabled_transformers:
            df = self._ensure_inputs_exist(df, self.transformers[0])
            prefixed_features = [self.prefix + f for f in self.features]

            for feature in prefixed_features:
                if feature in df.columns:
                    values = df[feature].to_numpy()
                    values = values[np.isfinite(values)]

                    # Skip if binary/categorical data (few unique values)
                    # Quantile scaler is for continuous zero-inflated data, not binary outcomes
                    n_unique = len(np.unique(values))
                    if n_unique <= 3:
                        continue

                    zero_proportion = np.mean(np.abs(values) < 1e-10)

                    if zero_proportion > self.zero_inflation_threshold:
                        logging.info(
                            f"Detected zero-inflated distribution for {feature} "
                            f"({zero_proportion:.1%} zeros). Using QuantilePerformanceScaler."
                        )
                        self._using_quantile_scaler = True
                        # Use original_transformers (deepcopy made before standard transformers
                        # were appended to custom_transformers)
                        self.transformers = [
                            copy.deepcopy(t) for t in self.original_transformers
                        ] + [
                            QuantilePerformanceScaler(
                                features=prefixed_features,
                                prefix="",
                            )
                        ]
                        break

        for t in self.transformers:
            df = self._ensure_inputs_exist(df, t)
            t.fit(df)
            df = nw.from_native(t.transform(df))
        return self

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        input_cols = df.columns

        for t in self.transformers:
            df = self._ensure_inputs_exist(df, t)
            df = nw.from_native(t.transform(df))

        df = self._post_transform(df, input_cols)
        return df.to_native()

    def _ensure_inputs_exist(self, df, t: NarwhalsFeatureTransformer):
        if not self.prefix:
            return df

        missing = [c for c in t.features if c not in df.columns]
        if not missing:
            return df

        exprs = []
        for feature in missing:
            ori_feature_name = feature.removeprefix(self.prefix)
            if ori_feature_name in df.columns:
                exprs.append(nw.col(ori_feature_name).alias(feature))
        if exprs:
            df = df.with_columns(exprs)
        return df

    def _post_transform(self, df, input_cols: list[str]):
        df = df.with_columns(
            nw.col(self.transformers[-1].features_out[0]).alias(self.performance_column)
        )
        df = df.with_columns(nw.col(self.performance_column).clip(self.min_value, self.max_value))
        return df.select(list(set([*input_cols, *self.features_out])))

    @property
    def features_out(self) -> list[str]:
        return [self.performance_column]


class PerformanceWeightsManager(PerformanceManager):
    def __init__(
        self,
        weights: list[ColumnWeight],
        performance_column: str = "weighted_performance",
        custom_transformers: list[NarwhalsFeatureTransformer] | None = None,
        transformer_names: (
            list[Literal["partial_standard_scaler", "symmetric", "min_max"]] | None
        ) = None,
        max_value: float = 1.02,
        min_value: float = -0.02,
        prefix: str = "performance__",
        return_all_features: bool = False,
        zero_inflation_threshold: float = 0.15,
    ):
        self.weights = weights
        self.return_all_features = return_all_features

        super().__init__(
            features=[p.name for p in weights],
            custom_transformers=custom_transformers,
            transformer_names=transformer_names,
            prefix=prefix,
            max_value=max_value,
            min_value=min_value,
            performance_column=performance_column,
            zero_inflation_threshold=zero_inflation_threshold,
        )

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        input_cols = df.columns

        for t in self.transformers:
            df = self._ensure_inputs_exist(df, t)
            df = nw.from_native(t.transform(df))

        df = self._calculate_weights(df=df)
        df = df.with_columns(nw.col(self.performance_column).clip(self.min_value, self.max_value))
        return df.select(list(set([*input_cols, *self.features_out]))).to_native()

    def _calculate_weights(self, df: IntoFrameT) -> IntoFrameT:

        df = self._weight_columns(df=df)

        if len(df.filter(nw.col(self.performance_column).is_null())) > 0 or len(
            df.filter(nw.col(self.performance_column).is_finite())
        ) != len(df):
            logging.error(
                f"df[{self.performance_column}] contains nan values. Make sure all column_names used in column_weights are imputed beforehand"
            )
            raise ValueError("performance contains nan values")

        return df

    def _weight_columns(
        self,
        df: IntoFrameT,
    ) -> IntoFrameT:
        tmp_out_performance_colum_name = f"__{self.performance_column}"
        df = df.with_columns(
            [
                nw.lit(0).alias(tmp_out_performance_colum_name),
                nw.lit(0).alias("sum_cols_weights"),
            ]
        )

        for column_weight in self.weights:
            weight_col = f"weight__{column_weight.name}"
            feature_col = f"{self.prefix}{column_weight.name}"

            df = df.with_columns(
                nw.when((nw.col(feature_col).is_null()) | (~nw.col(feature_col).is_finite()))
                .then(0)
                .otherwise(column_weight.weight)
                .alias(weight_col)
            )

            df = df.with_columns(
                nw.when((nw.col(feature_col).is_null()) | (~nw.col(feature_col).is_finite()))
                .then(0)
                .otherwise(nw.col(feature_col))
                .alias(feature_col)
            )

            df = df.with_columns(
                (nw.col("sum_cols_weights") + nw.col(weight_col)).alias("sum_cols_weights")
            )

        for column_weight in self.weights:
            df = df.with_columns(
                (nw.col(f"weight__{column_weight.name}") / nw.col("sum_cols_weights")).alias(
                    f"weight__{column_weight.name}"
                )
            )

        for column_weight in self.weights:
            weight_col = f"weight__{column_weight.name}"
            feature_col = column_weight.name
            feature_name = f"{self.prefix}{feature_col}"

            if column_weight.lower_is_better:
                df = df.with_columns(
                    (
                        nw.col(tmp_out_performance_colum_name)
                        + (nw.col(weight_col) * (1 - nw.col(feature_name)))
                    ).alias(tmp_out_performance_colum_name)
                )
            else:
                df = df.with_columns(
                    (
                        nw.col(tmp_out_performance_colum_name)
                        + (nw.col(weight_col) * nw.col(feature_name))
                    ).alias(tmp_out_performance_colum_name)
                )

        return df.with_columns(
            nw.col(tmp_out_performance_colum_name).alias(self.performance_column)
        )

    @property
    def features_out(self) -> list[str]:
        if self.return_all_features:
            return [self.prefix + f for f in self.features] + [self.performance_column]
        return [self.performance_column]
