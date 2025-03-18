from abc import abstractmethod, ABC
from narwhals.typing import FrameT, IntoFrameT

import narwhals as nw
import numpy as np


class PredictorTransformer(ABC):

    def __init__(self, features: list[str]):
        self.features = features

    @abstractmethod
    def fit_transform(self, df: FrameT) -> IntoFrameT:
        pass

    @abstractmethod
    def transform(self, df: FrameT) -> IntoFrameT:
        pass

    @property
    @abstractmethod
    def features_out(self) -> list[str]:
        pass


class ConvertDataFrameToCategoricalTransformer(PredictorTransformer):
    """
    Converts a specified list of columns to categorical dtype
    """

    def __init__(self, features: list[str]):
        super().__init__(features=features)

    @nw.narwhalify
    def fit_transform(self, df: FrameT) -> IntoFrameT:
        self._features_out = self.features
        return nw.from_native(self.transform(df)).select(self._features_out)

    @nw.narwhalify
    def transform(self, df: FrameT) -> IntoFrameT:
        df = df.with_columns(
            nw.col(feature).cast(nw.Categorical) for feature in self.features
        )
        return df.select(self._features_out)

    @property
    def features_out(self) -> list[str]:
        return self._features_out


class SkLearnTransformerWrapper(PredictorTransformer):
    """
    A wrapper around an Sklearn Transformer
    """

    def __init__(self, transformer, features: list[str]):
        self.transformer = transformer
        super().__init__(features=features)
        self._features_out = []

    @nw.narwhalify
    def fit_transform(self, df: FrameT) -> IntoFrameT:

        try:
            transformed_values = self.transformer.fit_transform(
                df.select(self.features).to_native()
            ).toarray()
        except AttributeError:
            transformed_values = self.transformer.fit_transform(
                df.select(self.features).to_native()
            )
            if not isinstance(transformed_values, np.ndarray):
                transformed_values = transformed_values.to_numpy()

        self._features_out = self.transformer.get_feature_names_out().tolist()

        return df.with_columns(
            nw.new_series(
                self._features_out[idx],
                transformed_values[:, idx],
                native_namespace=nw.get_native_namespace(df),
            )
            for idx in range(len(self._features_out))
        ).select(self._features_out)

    @nw.narwhalify
    def transform(self, df: FrameT) -> IntoFrameT:
        try:
            transformed_values = self.transformer.transform(
                df.select(self.features).to_native()
            ).toarray()
        except AttributeError:
            transformed_values = self.transformer.transform(
                df.select(self.features).to_native()
            )
            if not isinstance(transformed_values, np.ndarray):
                transformed_values = transformed_values.to_numpy()

        return df.with_columns(
            nw.new_series(
                self._features_out[idx],
                transformed_values[:, idx],
                native_namespace=nw.get_native_namespace(df),
            )
            for idx in range(len(self._features_out))
        ).select(self._features_out)

    @property
    def features_out(self) -> list[str]:
        return self._features_out
