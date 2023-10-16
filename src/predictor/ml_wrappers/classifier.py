from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.predictor.ml_wrappers.base_wrapper import BaseMLWrapper
from src.ratings.data_structures import ColumnNames


class SKLearnClassifierWrapper(BaseMLWrapper):

    def __init__(self,
                 features: list[str],
                 target: str,
                 model: Optional = None,
                 pred_column: Optional[str] = "prob",
                 column_names: Optional[ColumnNames] = None
                 ):
        self.features = features
        self._target = target
        self.model = model or LogisticRegression()
        self.column_names = column_names
        self._pred_column = pred_column

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df[self.features], df[self._target])

    def add_prediction(self, df: pd.DataFrame) -> None:
        df[self._pred_column] = self.model.predict_proba(df[self.features])[:, 1]

    @property
    def pred_column(self) -> str:
        return self._pred_column

    @property
    def target(self) -> str:
        return self._target

