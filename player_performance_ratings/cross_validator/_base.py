from abc import abstractmethod, ABC

import pandas as pd

from player_performance_ratings.scorer.score import BaseScorer
from player_performance_ratings.transformation.base_transformer import BaseTransformer

from player_performance_ratings.predictor import BasePredictor



class CrossValidator(ABC):

    def __init__(self, scorer: BaseScorer):
        self.scorer = scorer


    @abstractmethod
    def generate_validation_df(self,
                               df: pd.DataFrame,
                               post_transformers: list[BaseTransformer],
                               predictor: BasePredictor,
                               estimator_features: list[str],
                               ) -> pd.DataFrame:
        pass


    def cross_validation_score(self, validation_df: pd.DataFrame) -> float:
        return self.scorer.score(df=validation_df)

