from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd


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

    def __init__(self, features: list[str], features_to_remove:Optional[list[str]]=None):
        self.features = features
        self._features_to_remove = features_to_remove or []

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

    @property
    def features_to_remove(self) -> list[str]:
        return self._features_to_remove



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


