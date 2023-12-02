from abc import abstractmethod, ABC

import pandas as pd


class BaseTransformer(ABC):

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass



