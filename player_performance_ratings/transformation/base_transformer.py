from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd
from player_performance_ratings import ColumnNames


class BaseTransformer(ABC):

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

class BasePostTransformer(ABC):

    def __init__(self, features: list[str], are_estimator_features:bool =True):
        self.features = features
        self._are_estimator_features = are_estimator_features
        self._features_out = []

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    @property
    def estimator_features_out(self) -> list[str]:
        if self._are_estimator_features:
            return self.features_out
        return []


class DifferentGranularityTransformer(ABC):

    def __init__(self, features: list[str]):
        self.features = features

    @abstractmethod
    def fit_transform(self, diff_granularity_df: pd.DataFrame, game_player_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, diff_granularity_df: pd.DataFrame, game_player_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def features_out(self) -> list[str]:
        pass


