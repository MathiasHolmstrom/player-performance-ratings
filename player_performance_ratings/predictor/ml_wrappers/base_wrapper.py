from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd


class BaseMLWrapper(ABC):

    def __init__(self, target: str, pred_column: Optional[str] = "prob", ):
        self._target = target
        self._pred_column = pred_column

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
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

    def set_target(self, new_target_name: str):
        self._target = new_target_name
