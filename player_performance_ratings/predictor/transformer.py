from typing import Optional

import pandas as pd
from player_performance_ratings import PredictColumnNames

from player_performance_ratings.predictor._base import PredictorTransformer


class ConvertDataFrameToCategoricalTransformer(PredictorTransformer):

    def __init__(self, features: list[str]):
        super().__init__(features=features)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            df = df.assign(**{feature: df[feature].astype('category')})
        return df

    @property
    def features_out(self) -> list[str]:
        return self.features

class TargetMeanTransformer(PredictorTransformer):

    def __init__(self, features: list[str], granularity:Optional[list[str]] = None ,target: str = PredictColumnNames.TARGET):
        self.target = target
        self.granularity = granularity
        super().__init__(features=features)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("TargetMeanTransformer is not implemented yet")


    @property
    def features_out(self) -> list[str]:
        return self.features


class SkLearnTransformerWrapper(PredictorTransformer):

    def __init__(self, transformer, features: list[str]):
        self.transformer = transformer
        super().__init__(features=features)
        self._features_out = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            transformed_values = self.transformer.fit_transform(df[self.features]).toarray()
        except AttributeError:
            transformed_values = self.transformer.fit_transform(df[self.features])
            if isinstance(transformed_values, pd.DataFrame):
                transformed_values = transformed_values.to_numpy()

        self._features_out = self.transformer.get_feature_names_out().tolist()

        return df.assign(
            **{self._features_out[idx]: transformed_values[:, idx] for idx in range(len(self._features_out))})

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            transformed_values = self.transformer.transform(df[self.features]).toarray()
        except AttributeError:
            transformed_values = self.transformer.transform(df[self.features])
            if isinstance(transformed_values, pd.DataFrame):
                transformed_values = transformed_values.to_numpy()
        return df.assign(
            **{self._features_out[idx]: transformed_values[:, idx] for idx in range(len(self._features_out))})

    @property
    def features_out(self) -> list[str]:
        return self._features_out
