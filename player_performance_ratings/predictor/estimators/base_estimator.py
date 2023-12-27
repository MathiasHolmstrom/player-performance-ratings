from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd


class BaseMLWrapper(ABC):

    def __init__(self, model, features: list[str] ,target: str, pred_column: Optional[str] = "prob"):
        self.model = model
        self.features = features
        self._target = target
        self._pred_column = pred_column

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
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
        if 'classes_' not in dir(self.model):
            return None
        return self.model.classes_

    def set_target(self, new_target_name: str):
        self._target = new_target_name



