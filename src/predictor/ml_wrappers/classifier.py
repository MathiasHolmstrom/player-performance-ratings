from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.predictor.ml_wrappers.base_wrapper import BaseMLWrapper


class SKLearnClassifierWrapper(BaseMLWrapper):

    def __init__(self,
                 features: list[str],
                 target: str,
                 model: Optional = None,
                 pred_column: Optional[str] = "prob"
                 ):
        super().__init__(features=features, target=target, pred_column=pred_column)
        self.model = model or LogisticRegression()

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df[self.features], df[self.target])

    def add_prediction(self, df: pd.DataFrame) -> None:
        df[self.pred_column] = self.model.predict_proba(df[self.features])[:, 1]
