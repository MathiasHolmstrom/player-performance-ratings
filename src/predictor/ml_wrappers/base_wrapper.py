from abc import abstractmethod, ABC

import pandas as pd


class BaseMLWrapper(ABC):

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def add_prediction(self, df: pd.DataFrame) -> None:
        pass