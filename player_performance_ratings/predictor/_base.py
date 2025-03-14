import logging
from abc import abstractmethod, ABC
from typing import Optional, TypeVar, Any

import polars as pl
import pandas as pd
from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw
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
        estimator_features: list[str],
        target: str,
        estimator_features_contain: Optional[list[str]] = None,
        scale_features: bool = False,
        one_hot_encode_cat_features: bool = False,
        convert_to_cat_feats_to_cat_dtype: bool = False,
        impute_missing_values: bool = False,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        pred_column: Optional[str] = None,
        filters: Optional[dict] = None,
        auto_pre_transform: bool = True,
        multiclass_output_as_struct: bool = False,
    ):
        self._estimator_features = estimator_features or []
        self._target = target
        self._estimator_features_contain = estimator_features_contain or []
        self.post_predict_transformers = post_predict_transformers or []
        self.convert_to_cat_feats_to_cat_dtype = convert_to_cat_feats_to_cat_dtype
        self.impute_missing_values = impute_missing_values
        self._pred_column = pred_column or f"{self._target}_prediction"
        self.pre_transformers = pre_transformers or []
        self.scale_features = scale_features
        self.one_hot_encode_cat_features = one_hot_encode_cat_features
        self.filters = filters or []
        self.multiclassifier = False
        self._classes_ = None
        self.auto_pre_transform = auto_pre_transform
        self.multiclass_output_as_struct = multiclass_output_as_struct

    def _deepest_estimator(self, predictor: "BasePredictor") -> Any:
        iterations = 0

        while not hasattr(predictor, "estimator"):
            predictor = predictor.predictor
            iterations += 1
            if iterations > 10:
                raise ValueError("estimator is too deep")

        return predictor.estimator

    def reset(self) -> None:
        pass

    def _add_estimator_features_contain(self, df: FrameT) -> FrameT:
        columns = df.columns
        for contain in self._estimator_features_contain:
            estimator_feature_count = len(self._estimator_features)
            for column in columns:
                if column not in self._estimator_features and contain in column:
                    self._estimator_features.append(column)
            if len(self._estimator_features) == estimator_feature_count:
                logging.warning(f"Added no new columns containing {contain}")
        return df

    @abstractmethod
    def train(self, df: FrameT, estimator_features: Optional[list[str]] = None) -> None:
        pass

    @abstractmethod
    def predict(
        self, df: FrameT, cross_validation: Optional[bool] = None
    ) -> IntoFrameT:
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

    def _convert_multiclass_predictions_to_struct(
        self, df: FrameT, classes: list[str]
    ) -> FrameT:
        df = df.to_native()
        assert isinstance(df, pl.DataFrame)

        return nw.from_native(
            df.with_columns(
                pl.struct(
                    *[
                        pl.col(self.pred_column).list.get(i).alias(str(cls))
                        for i, cls in enumerate(classes)
                    ]
                ).alias(self.pred_column)
            )
        )

    def set_target(self, new_target_name: str):
        self._target = new_target_name

    def _create_pre_transformers(self, df: pl.DataFrame) -> list[PredictorTransformer]:
        pre_transformers = []
        cat_feats_to_transform = []

        for estimator_feature in self._estimator_features.copy():
            if not df[estimator_feature].dtype.is_numeric():
                cat_feats_to_transform.append(estimator_feature)

        for estimator_feature in self._estimator_features.copy():

            if estimator_feature not in df.columns:
                self._estimator_features.remove(estimator_feature)
                logging.warning(
                    f"Feature {estimator_feature} not in df, removing from estimator_features"
                )
                continue

        if cat_feats_to_transform:
            if self.one_hot_encode_cat_features:
                one_hot_encoder = SkLearnTransformerWrapper(
                    transformer=OneHotEncoder(handle_unknown="ignore"),
                    features=cat_feats_to_transform,
                )
                pre_transformers.append(one_hot_encoder)

            elif self.convert_to_cat_feats_to_cat_dtype:

                pre_transformers.append(
                    ConvertDataFrameToCategoricalTransformer(
                        features=cat_feats_to_transform
                    )
                )

        if self.scale_features:
            numeric_feats = [
                f for f in self._estimator_features if f not in cat_feats_to_transform
            ]
            if numeric_feats:
                pre_transformers.append(
                    SkLearnTransformerWrapper(
                        transformer=StandardScaler(),
                        features=numeric_feats,
                    )
                )

        if self.impute_missing_values:
            numeric_feats = [
                f for f in self._estimator_features if f not in cat_feats_to_transform
            ]
            if numeric_feats:

                imputer_transformer = SkLearnTransformerWrapper(
                    transformer=SimpleImputer(),
                    features=numeric_feats,
                )
                if not self._transformer_exists(imputer_transformer):
                    pre_transformers.append(imputer_transformer)

        return pre_transformers

    def _transformer_exists(self, transformer: PredictorTransformer) -> bool:
        for pre_transformer in self.pre_transformers:
            if (
                pre_transformer.__class__.__name__ == transformer.__class__.__name__
                and pre_transformer.features == transformer.features
            ):
                return True
        return False

    def _fit_transform_pre_transformers(self, df: FrameT) -> FrameT:
        if self.auto_pre_transform:
            self.pre_transformers += self._create_pre_transformers(df)

        native_namespace = nw.get_native_namespace(df)

        for pre_transformer in self.pre_transformers:
            values = nw.from_native(pre_transformer.fit_transform(df))
            features_out = pre_transformer.features_out
            df = df.with_columns(
                nw.new_series(
                    values=values[col].to_native(),
                    name=features_out[idx],
                    native_namespace=native_namespace,
                )
                for idx, col in enumerate(values.columns)
            )

            for feature in pre_transformer.features:
                if (
                    feature in self._estimator_features
                    and feature not in pre_transformer.features_out
                ):
                    self._estimator_features.remove(feature)
            for features_out in pre_transformer.features_out:
                if features_out not in self._estimator_features:
                    self._estimator_features.append(features_out)

        return df

    def _transform_pre_transformers(self, df: nw.DataFrame) -> IntoFrameT:
        for pre_transformer in self.pre_transformers:
            values = nw.from_native(pre_transformer.transform(df))
            df = df.with_columns(
                nw.new_series(
                    name=col,
                    values=values[col].to_native(),
                    native_namespace=nw.get_native_namespace(df),
                )
                for col in values.columns
            )
        return df

    @property
    def estimator_features(self) -> list[str]:
        return self._estimator_features
