from abc import ABC, abstractmethod

from narwhals.stable.v1.typing import IntoFrameT

from spforge.data_structures import ColumnNames


class FeatureGenerator(ABC):

    def __init__(self, features_out: list[str]):
        self._features_out = features_out

    @abstractmethod
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        pass

    @abstractmethod
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        pass

    def transform(self, df: IntoFrameT) -> IntoFrameT:
        raise NotImplementedError

    @property
    def features_out(self) -> list[str]:
        return self._features_out
