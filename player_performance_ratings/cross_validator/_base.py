from abc import abstractmethod, ABC

import pandas as pd
from player_performance_ratings.predictor import BaseMLWrapper

from player_performance_ratings.scorer import BaseScorer


class CrossValidator(ABC):

    def __init__(self, scorer: BaseScorer):
        self.scorer = scorer


    @abstractmethod
    def generate_validation_df(self, df: pd.DataFrame, predictor: BaseMLWrapper) -> pd.DataFrame:
        pass


    def cross_validation_score(self, validation_df: pd.DataFrame) -> float:
        return self.scorer.score(df=validation_df)

