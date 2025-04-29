import logging
from abc import abstractmethod, ABC
from typing import Optional, TypeVar, Any

import polars as pl
import pandas as pd
from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spforge.predictor_transformer import (
    PredictorTransformer,
    SkLearnTransformerWrapper,
    ConvertDataFrameToCategoricalTransformer,
)
from spforge.predictor_transformer._simple_transformer import (
    SimpleTransformer,
)

DataFrameType = TypeVar("DataFrameType", pd.DataFrame, pl.DataFrame)


class BasePredictor(ABC):

    def __init__(
        self,
        features: list[str],
        target: str,
        features_contain_str: Optional[list[str]] = None,
        scale_features: bool = False,
        one_hot_encode_cat_features: bool = False,
        convert_cat_features_to_cat_dtype: bool = False,
        impute_missing_values: bool = False,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        pred_column: Optional[str] = None,
        filters: Optional[dict] = None,
        auto_pre_transform: bool = True,
        multiclass_output_as_struct: bool = False,
    ):
        self._features = features or []
        self._modified_features = self._features.copy()
        self._ori_estimator_features = self._features.copy()
        self._target = target
        self.features_contain_str = features_contain_str or []
        self.post_predict_transformers = post_predict_transformers or []
        self.convert_cat_features_to_cat_dtype = convert_cat_features_to_cat_dtype
        self.impute_missing_values = impute_missing_values
        self._pred_column = pred_column or f"{self._target}_prediction"
        self._pred_columns_added = [self._pred_column]
        self.pre_transformers = pre_transformers or []
        self.scale_features = scale_features
        self.one_hot_encode_cat_features = one_hot_encode_cat_features
        self.filters = filters or []
        self.multiclassifier = False
        self._classes_ = None
        self.auto_pre_transform = auto_pre_transform
        self.multiclass_output_as_struct = multiclass_output_as_struct

    def reset(self) -> None:
        pass

    def _add_features_contain_str(self, df: FrameT) -> None:
        columns = df.columns
        already_added = []
        for contain in self.features_contain_str:
            estimator_feature_count = len(self._features)
            for column in columns:
                if column not in self._features and contain in column:
                    self._features.append(column)
                    self._modified_features.append(column)
                elif contain in column:
                    already_added.append(column)
            if (
                len(self._features) == estimator_feature_count
                and len(already_added) == 0
            ):
                raise ValueError(f"Feature Contain {contain} not found in df")

    @abstractmethod
    def train(self, df: FrameT, features: Optional[list[str]] = None) -> None:
        pass

    @abstractmethod
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:
        pass

    @property
    def pred_column(self) -> str:
        return self._pred_column

    @pred_column.setter
    def pred_column(self, new_pred_column: str):
        self._pred_column = new_pred_column

    @property
    def target(self) -> str:
        return self._target

    @target.setter
    def target(self, new_target_name: str):
        self._target = new_target_name

    @property
    def columns_added(self) -> list[str]:
        if not self.multiclassifier or self.multiclass_output_as_struct:
            return self._pred_columns_added
        return [*self._pred_columns_added, "classes"]

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

        for estimator_feature in self._features.copy():
            if not df[estimator_feature].dtype.is_numeric():
                cat_feats_to_transform.append(estimator_feature)

        for estimator_feature in self._features.copy():

            if estimator_feature not in df.columns:
                self._modified_features.remove(estimator_feature)
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

            elif self.convert_cat_features_to_cat_dtype:

                pre_transformers.append(
                    ConvertDataFrameToCategoricalTransformer(
                        features=cat_feats_to_transform
                    )
                )

        if self.scale_features:
            numeric_feats = [
                f for f in self._features if f not in cat_feats_to_transform
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
                f for f in self._modified_features if f not in cat_feats_to_transform
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
                    feature in self._modified_features
                    and feature not in pre_transformer.features_out
                ):
                    self._modified_features.remove(feature)
            for features_out in pre_transformer.features_out:
                if features_out not in self._modified_features:
                    self._modified_features.append(features_out)

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
    def features(self) -> list[str]:
        return self._features
