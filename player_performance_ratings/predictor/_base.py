from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd
from player_performance_ratings.scorer.score import Filter

class PredictorTransformer(ABC):

    def __init__(self, features: list[str]):
        self.features = features

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def features_out(self) -> list[str]:
        pass


class BaseMLWrapper(ABC):

    def __init__(self,
                 estimator,
                 filters: Optional[list[Filter]],
                 target: str,
                 categorical_transformers: Optional[list[PredictorTransformer]] = None,
                 pred_column: Optional[str] = None,
                 ):
        self.estimator = estimator
        self.filters = filters or []
        self._target = target
        self._pred_column = pred_column or f"{self._target}_prediction"
        self.categorical_transformers = categorical_transformers
        self._estimator_categorical_features = []
        self._deepest_estimator = self.estimator
        while hasattr(self._deepest_estimator , "estimator"):
            self._deepest_estimator  = self._deepest_estimator.estimator

    @abstractmethod
    def train(self, df: pd.DataFrame, estimator_features: list[str]) -> None:
        pass

    @abstractmethod
    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def pred_column(self) -> str:
        return self._pred_column

    @property
    def target(self) -> str:
        return self._target

    @property
    def classes_(self) -> Optional[list[str]]:
        if 'classes_' not in dir(self.estimator):
            return None
        return self.estimator.classes_

    def set_target(self, new_target_name: str):
        self._target = new_target_name

    def fit_transform_categorical_transformers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.categorical_transformers:
            for pre_transformer in self.categorical_transformers:
                df = pre_transformer.fit_transform(df)
                for feature in pre_transformer.features:
                    if feature in self._estimator_features:
                        self._estimator_features.remove(feature)
                self._estimator_features = list(set(pre_transformer.features_out + self._estimator_features))
                self._estimator_categorical_features = list(
                    set(pre_transformer.features_out + self._estimator_categorical_features))
        return df

    def transform_categorical_transformers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.categorical_transformers:
            for pre_transformer in self.categorical_transformers:
                df = pre_transformer.transform(df)
        return df

    @property
    def estimator_features(self) -> list[str]:
        return self._estimator_features

    @property
    def estimator_categorical_features(self) -> list[str]:
        return self._estimator_categorical_features
