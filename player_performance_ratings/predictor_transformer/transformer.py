from abc import abstractmethod, ABC
from narwhals.typing import FrameT

import pandas as pd
import narwhals as nw
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class PredictorTransformer(ABC):

    def __init__(self, features: list[str]):
        self.features = features

    @abstractmethod
    def fit_transform(self, df: FrameT) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: FrameT) -> pd.DataFrame:
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
    def fit_transform(self, df: FrameT) -> pd.DataFrame:
        self._features_out = self.features
        return self.transform(df.to_pandas())[self._features_out]

    @nw.narwhalify
    def transform(self, df: FrameT) -> pd.DataFrame:
        df = df.with_columns(
            nw.col(feature).cast(nw.Categorical) for feature in self.features
        )
        return df.select(self._features_out).to_pandas()

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
    def fit_transform(self, df: FrameT) -> pd.DataFrame:

        try:
            transformed_values = self.transformer.fit_transform(
                df.select(self.features).to_pandas()
            ).toarray()
        except AttributeError:
            transformed_values = self.transformer.fit_transform(df.select(self.features))
            if isinstance(transformed_values, pd.DataFrame):
                transformed_values = transformed_values.to_numpy()

        self._features_out = self.transformer.get_feature_names_out().tolist()


        return df.to_pandas().assign(
            **{
                self._features_out[idx]: transformed_values[:, idx]
                for idx in range(len(self._features_out))
            }
        )[self._features_out]



    def transform(self, df: FrameT) -> pd.DataFrame:
        try:
            transformed_values = self.transformer.transform(df.select(self.features)).toarray()
        except AttributeError:
            transformed_values = self.transformer.transform(df.select(self.features))
            if isinstance(transformed_values, pd.DataFrame):
                transformed_values = transformed_values.to_numpy()


        return df.to_pandas().assign(
            **{
                self._features_out[idx]: transformed_values[:, idx]
                for idx in range(len(self._features_out))
            }
        )[self._features_out]

    @property
    def features_out(self) -> list[str]:
        return self._features_out