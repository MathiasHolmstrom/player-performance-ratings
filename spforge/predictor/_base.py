import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, TypeVar, Any

import polars as pl
import pandas as pd
from narwhals.typing import IntoFrameT, IntoFrameT
import narwhals.stable.v2 as nw
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

@dataclass
class LagFeature:
    base_name: str
    min_lag: int
    max_lag:int

class BasePredictor(ABC):

    def __init__(
        self,
        features: list[str],
        target: str,
        lag_features: list[LagFeature] | None = None,
        scale_features: bool = False,
        one_hot_encode_cat_features: bool = False,
        convert_cat_features_to_cat_dtype: bool = False,
        impute_missing_values: bool = False,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        pred_column: Optional[str] = None,
        filters: Optional[dict] = None,
        auto_pre_transform: bool = True,
    ):
        self._features = features or []
        self._modified_features = self._features.copy()
        self._ori_estimator_features = self._features.copy()
        self._target = target
        self.lag_features = lag_features or []
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

        for lag_feature in self.lag_features:
            for iter in range(lag_feature.min_lag, lag_feature.max_lag):
                self._features.append(
                    f"lag_{lag_feature.base_name}{iter}"
                )

    def reset(self) -> None:
        pass

    @abstractmethod
    def train(self, df: IntoFrameT, features: Optional[list[str]] = None) -> None:
        pass

    @abstractmethod
    def predict(
        self, df: IntoFrameT,  **kwargs
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
        if not self.multiclassifier:
            return self._pred_columns_added
        return [*self._pred_columns_added, "classes"]


    def set_target(self, new_target_name: str):
        self._target = new_target_name



    @property
    def features(self) -> list[str]:
        return self._features


class DistributionPredictor(BasePredictor):

    def __init__(
        self,
        target: str,
        point_estimate_pred_column: str,
        min_value: int,
        max_value: int,
        pred_column: Optional[str] = None,
        filters: Optional[dict] = None,
        auto_pre_transform: bool = True,
    ):

        super().__init__(
            target=target,
            features=[],
            pred_column=pred_column or f"{target}_probabilities",
            filters=filters or {},
            auto_pre_transform=auto_pre_transform,
        )
        self.point_estimate_pred_column = point_estimate_pred_column
        self.min_value = min_value
        self.max_value = max_value