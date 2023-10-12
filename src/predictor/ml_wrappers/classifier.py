from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.predictor.ml_wrappers.base_wrapper import BaseMLWrapper


class SKLearnClassifierWrapper(BaseMLWrapper):

    def __init__(self,
                 target: str,
                 features: list[str],
                 model: Optional = None,
                 prob_column_name: Optional[str] = "prob"
                 ):
        self.model = model or LogisticRegression()
        self.target = target
        self.features = features
        self.prob_column_name = prob_column_name

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df[self.features], df[self.target])

    def add_prediction(self, df: pd.DataFrame) -> None:
        df[self.prob_column_name] = self.model.predict_proba(df[self.features])[:, 1]
