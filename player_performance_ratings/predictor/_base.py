import logging
from abc import abstractmethod, ABC
from typing import Optional, TypeVar

import polars as pl
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from player_performance_ratings.predictor_transformer import (
    PredictorTransformer,
    SkLearnTransformerWrapper,
    ConvertDataFrameToCategoricalTransformer,
)
from player_performance_ratings.predictor_transformer._simple_transformer import (
    SimpleTransformer,
)
DataFrameType = TypeVar("DataFrameType", pd.DataFrame, pl.DataFrame)

class BasePredictor(ABC):

    def __init__(
        self,
        estimator,
        estimator_features: list[str],
        target: str,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        pred_column: Optional[str] = None,
        filters: Optional[dict] = None,
        auto_pre_transform: bool = True,
        multiclass_output_as_struct: bool = False,
    ):
        self._estimator_features = estimator_features or []
        self.estimator = estimator
        self._target = target
        self.post_predict_transformers = post_predict_transformers or []
        self._pred_column = pred_column or f"{self._target}_prediction"
        self.pre_transformers = pre_transformers or []
        self._deepest_estimator = self.estimator
        self.filters = filters or []
        self.multiclassifier = False
        self._classes_ = None
        self.auto_pre_transform = auto_pre_transform
        self.multiclass_output_as_struct = multiclass_output_as_struct

        iterations = 0
        while hasattr(self._deepest_estimator, "estimator"):
            self._deepest_estimator = self._deepest_estimator.estimator
            iterations += 1
            if iterations > 10:
                raise ValueError("estimator is too deep")

    @property
    def estimator_type(self) -> str:
        if hasattr(self._deepest_estimator, "predict_proba"):
            return "classifier"
        return "regressor"

    @property
    def deepest_estimator(self) -> object:
        return self._deepest_estimator

    @abstractmethod
    def train(self, df: DataFrameType, estimator_features: list[str]) -> None:
        pass

    @abstractmethod
    def add_prediction(self, df: DataFrameType) -> DataFrameType:
        pass

    @property
    def pred_column(self) -> str:
        return self._pred_column

    @property
    def target(self) -> str:
        return self._target

    @property
    def columns_added(self) -> list[str]:
        if not self.multiclassifier or self.multiclass_output_as_struct:
            return [self.pred_column]
        return [self.pred_column, "classes"]

    @property
    def classes_(self) -> Optional[list[str]]:
        if self._classes_:
            return self._classes_
        if "classes_" not in dir(self.estimator):
            return None
        return self.estimator.classes_

    def _convert_multiclass_predictions_to_struct(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:

        input_cols = df.columns
        df = df.with_row_index("id")
        df_exploded = df.explode(["outcome", "probs"])

        df_pivoted = df_exploded.pivot(
            values="probs",
            index="id",
            columns="outcome"
        )

        class_columns = [c for c in df_pivoted.columns if c not in ["id"]]
        return df_pivoted.with_columns(
            pl.struct(class_columns).alias(self.pred_column)
        ).select(*input_cols)


    def set_target(self, new_target_name: str):
        self._target = new_target_name

    def _create_pre_transformers(self, df: pl.DataFrame) -> list[PredictorTransformer]:
        pre_transformers = []
        cat_feats_to_transform = []
        all_feats_in_pre_transformers = [
            f for c in self.pre_transformers for f in c.features
        ]
        for estimator_feature in self._estimator_features.copy():

            if estimator_feature not in df.columns:
                self._estimator_features.remove(estimator_feature)
                logging.warning(
                    f"Feature {estimator_feature} not in df, removing from estimator_features"
                )
                continue

            if not df[estimator_feature].dtype.is_numeric():
                if estimator_feature not in all_feats_in_pre_transformers:
                    cat_feats_to_transform.append(estimator_feature)

        if cat_feats_to_transform:
            if self._deepest_estimator.__class__.__name__ in (
                "LogisticRegression",
                "LinearRegression",
            ):
                logging.info(
                    f"Adding OneHotEncoder to pre_transformers for features: {cat_feats_to_transform}"
                )
                pre_transformers.append(
                    SkLearnTransformerWrapper(
                        transformer=OneHotEncoder(handle_unknown="ignore"),
                        features=cat_feats_to_transform,
                    )
                )

            elif self._deepest_estimator.__class__.__name__ in (
                "LGBMRegressor",
                "LGBMClassifier",
            ):
                logging.info(
                    f"Adding ConvertDataFrameToCategoricalTransformer to pre_transformers for features: {cat_feats_to_transform}"
                )
                pre_transformers.append(
                    ConvertDataFrameToCategoricalTransformer(
                        features=cat_feats_to_transform
                    )
                )

        for pre_transformer in self.pre_transformers:
            for feature in pre_transformer.features:
                if feature in self._estimator_features:
                    self._estimator_features.remove(feature)

        if self._deepest_estimator.__class__.__name__ in (
            "LogisticRegression",
            "LinearRegression",
        ):
            if "StandardScaler" not in [
                pre_transformer.transformer.__class__.__name__
                for pre_transformer in self.pre_transformers
                if hasattr(pre_transformer, "transformer")
            ]:
                logging.info(f"Adding StandardScaler to pre_transformers")
                numeric_feats = [
                    f
                    for f in self._estimator_features
                    if f not in cat_feats_to_transform
                ]
                pre_transformers.append(
                    SkLearnTransformerWrapper(
                        transformer=StandardScaler(),
                        features=numeric_feats,
                    )
                )

            if "SimpleImputer" not in [
                pre_transformer.transformer.__class__.__name__
                for pre_transformer in self.pre_transformers
                if hasattr(pre_transformer, "transformer")
            ]:
                numeric_feats = [
                    f
                    for f in self._estimator_features
                    if f not in cat_feats_to_transform
                ]
                pre_transformers.append(
                    SkLearnTransformerWrapper(
                        transformer=SimpleImputer(),
                        features=numeric_feats,
                    )
                )
        return pre_transformers

    def _fit_transform_pre_transformers(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.auto_pre_transform:
            self.pre_transformers += self._create_pre_transformers(df)

        for pre_transformer in self.pre_transformers:
            values = pre_transformer.fit_transform(df)
            features_out = pre_transformer.features_out
            df = df.with_columns(
                pl.col(v).alias(features_out[idx]) for idx, v in enumerate(values)
            )

            feats_to_remove = [
                f for f in pre_transformer.features if f in self._estimator_features
            ]
            if feats_to_remove:
                for feat in feats_to_remove:
                    self._estimator_features.remove(feat)
            self._estimator_features = list(
                set(pre_transformer.features_out + self._estimator_features)
            )

        return df

    def _transform_pre_transformers(self, df: pl.DataFrame) -> pl.DataFrame:
        for pre_transformer in self.pre_transformers:
            values = pre_transformer.transform(df)
            df = df.with_columns(**{col: values[col] for col in values.columns})
        return df

    @property
    def estimator_features(self) -> list[str]:
        return self._estimator_features
