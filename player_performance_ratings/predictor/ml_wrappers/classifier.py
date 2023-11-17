import logging
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor.ml_wrappers.base_wrapper import BaseMLWrapper
from player_performance_ratings.data_structures import ColumnNames


class SKLearnClassifierWrapper(BaseMLWrapper):

    def __init__(self,
                 features: list[str],
                 target: Optional[str] = PredictColumnNames.TARGET,
                 model: Optional = None,
                 multiclassifier: bool = False,
                 granularity: Optional[list[str]] = None,
                 pred_column: Optional[str] = "prob",
                 column_names: Optional[ColumnNames] = None
                 ):
        self.features = features
        self._target = target
        self.multiclassifier = multiclassifier
        self.granularity = granularity
        self.column_names = column_names

        super().__init__(target=self._target, pred_column=pred_column, model=model or LogisticRegression())

    def fit(self, df: pd.DataFrame) -> None:
        if self.granularity:
            if df[self._target].dtype =='object':
                df[self._target] = df[self._target].astype('int')
                logging.warning(f"target {self._target} was converted to int from object")
            grouped = df.groupby(self.granularity)[self.features + [self._target]].mean().reset_index()
            grouped[self._target] = grouped[self._target].astype('int')
            self.model.fit(grouped[self.features], grouped[self._target])
        else:
            self.model.fit(df[self.features], df[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.granularity:
            if df[self._target].dtype == 'object':
                df[self._target] = df[self._target].astype('int')
            grouped = df.groupby(self.granularity)[self.features + [self._target]].mean().reset_index()
            grouped[self._target] = grouped[self._target].astype('int')
            if self.multiclassifier:
                grouped[self._pred_column] = self.model.predict_proba(grouped[self.features]).tolist()
            else:
                grouped[self._pred_column] = self.model.predict_proba(grouped[self.features])[:, 1]
            if self.pred_column in df.columns:
                df = df.drop(columns=[self.pred_column])
            df = df.merge(grouped[self.granularity + [self._pred_column]], on=self.granularity)
        else:
            if self.multiclassifier:
                df[self._pred_column] = self.model.predict_proba(df[self.features]).tolist()
            else:
                df[self._pred_column] = self.model.predict_proba(df[self.features])[:, 1]
        return df
