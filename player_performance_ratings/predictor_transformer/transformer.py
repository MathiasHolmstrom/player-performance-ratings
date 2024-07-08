from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd
from player_performance_ratings import PredictColumnNames
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class PredictorTransformer(ABC):

    def __init__(self, features: list[str]):
        self.features = features

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._features_out = self.features
        return self.transform(df)[self._features_out]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            df = df.assign(**{feature: df[feature].astype("category")})
        return df[self._features_out]

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

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            transformed_values = self.transformer.fit_transform(
                df[self.features]
            ).toarray()
        except AttributeError:
            transformed_values = self.transformer.fit_transform(df[self.features])
            if isinstance(transformed_values, pd.DataFrame):
                transformed_values = transformed_values.to_numpy()

        self._features_out = self.transformer.get_feature_names_out().tolist()

        return df.assign(
            **{
                self._features_out[idx]: transformed_values[:, idx]
                for idx in range(len(self._features_out))
            }
        )[self._features_out]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            transformed_values = self.transformer.transform(df[self.features]).toarray()
        except AttributeError:
            transformed_values = self.transformer.transform(df[self.features])
            if isinstance(transformed_values, pd.DataFrame):
                transformed_values = transformed_values.to_numpy()
        return df.assign(
            **{
                self._features_out[idx]: transformed_values[:, idx]
                for idx in range(len(self._features_out))
            }
        )[self._features_out]

    @property
    def features_out(self) -> list[str]:
        return self._features_out
