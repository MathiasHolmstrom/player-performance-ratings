from abc import abstractmethod, ABC

import pandas as pd


class CrossValidator(ABC):

    def __init__(self):
        self._scores = []

    @abstractmethod
    def cross_validate(self, df: pd.DataFrame) -> float:
        pass

    @property
    def scores(self) -> list[float]:
        return self._scores