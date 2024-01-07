from abc import abstractmethod, ABC

import pandas as pd
from player_performance_ratings.predictor import BaseMLWrapper

from player_performance_ratings.scorer import BaseScorer


class CrossValidator(ABC):

    def __init__(self, scorer: BaseScorer, predictor: BaseMLWrapper):
        self.scorer = scorer
        self.predictor = predictor
        self._predictors = []
        self._scores = []

    @abstractmethod
    def cross_validate_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


    def cross_validation_score(self, validation_df: pd.DataFrame) -> float:
        classes_ = []
        for p in self._predictors:
            classes_ += [c for c in p.classes_.tolist() if c not in classes_]
        classes_.sort()
        return self.scorer.score(df=validation_df, classes_=classes_.sort())
