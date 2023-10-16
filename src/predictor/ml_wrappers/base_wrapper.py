from abc import abstractmethod, ABC

import pandas as pd


class BaseMLWrapper(ABC):


    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def add_prediction(self, df: pd.DataFrame) -> None:
        pass

    @property
    @abstractmethod
    def pred_column(self) -> str:
        pass

    @property
    @abstractmethod
    def target(self) -> str:
        pass
