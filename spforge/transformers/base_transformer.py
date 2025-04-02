from abc import abstractmethod, ABC
from typing import Optional

from narwhals.typing import FrameT, IntoFrameT

from spforge import ColumnNames


class BaseTransformer(ABC):

    def __init__(
        self,
        features: list[str],
        features_out: list[str],
        are_estimator_features: bool = True,
    ):
        self._features_out = features_out
        self.features = features
        self._are_estimator_features = are_estimator_features
        self.column_names = None
        self._predictor_features_out = (
            self._features_out if self._are_estimator_features else []
        )

    @abstractmethod
    def fit_transform(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        pass

    @abstractmethod
    def transform(self, df: FrameT, cross_validate: bool = False) -> IntoFrameT:
        pass

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    @property
    def predictor_features_out(self) -> list[str]:
        return self._predictor_features_out

    def reset(self) -> "BaseTransformer":
        return self
