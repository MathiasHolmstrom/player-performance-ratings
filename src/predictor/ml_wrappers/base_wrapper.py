from abc import abstractmethod, ABC

import pandas as pd


class BaseMLWrapper(ABC):

    def __init__(self, features: list[str], target: str, pred_column: str = "prob"):
        self.features = features
        self.target = target
        self.pred_column = pred_column

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def add_prediction(self, df: pd.DataFrame) -> None:
        pass
