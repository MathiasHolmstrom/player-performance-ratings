from typing import Optional

import pandas as pd
from player_performance_ratings.scorer import BaseScorer

from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper


class CrossValidator():

    def __init__(self, predictor: BaseMLWrapper, n_splits: int = 5, min_train_date: Optional[str] = None):
        self.predictor = predictor
        self.n_splits = n_splits
        self.min_train_date = min_train_date


    def cross_validate(self, df: pd.DataFrame, scorer: BaseScorer) -> float:
        pass