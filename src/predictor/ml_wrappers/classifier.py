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
                 granularity: Optional[list[str]] = None,
                 pred_column: Optional[str] = "prob",
                 column_names: Optional[ColumnNames] = None
                 ):
        self.features = features
        self._target = target
        self.model = model or LogisticRegression()
        self.granularity = granularity
        self.column_names = column_names
        self._pred_column = pred_column

    def fit(self, df: pd.DataFrame) -> None:
        if self.granularity:
            grouped = df.groupby(self.granularity)[self.features + [self._target]].mean().reset_index()
            grouped[self._target] = grouped[self._target].astype('int')
            self.model.fit(grouped[self.features], grouped[self._target])
        else:
            self.model.fit(df[self.features], df[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.granularity:
            grouped = df.groupby(self.granularity)[self.features + [self._target]].mean().reset_index()
            grouped[self._target] = grouped[self._target].astype('int')
            grouped[self._pred_column] = self.model.predict_proba(grouped[self.features])[:, 1]
            if self.pred_column in df.columns:
                df = df.drop(columns=[self.pred_column])
            df = df.merge(grouped[self.granularity + [self._pred_column]], on=self.granularity)
        else:
            df[self._pred_column] = self.model.predict_proba(df[self.features])[:, 1]
        return df

    @property
    def pred_column(self) -> str:
        return self._pred_column

    @property
    def target(self) -> str:
        return self._target
