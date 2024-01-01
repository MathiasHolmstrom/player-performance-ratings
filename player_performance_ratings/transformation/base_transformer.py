from abc import abstractmethod, ABC

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


