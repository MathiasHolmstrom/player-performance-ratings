from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd
from player_performance_ratings import ColumnNames

from player_performance_ratings.predictor._base import BasePredictor
from player_performance_ratings.scorer.score import BaseScorer
from player_performance_ratings.transformation.base_transformer import BaseTransformer



class CrossValidator(ABC):

    def __init__(self, scorer: Optional[BaseScorer]):
        self.scorer = scorer


    @abstractmethod
    def generate_validation_df(self,
                               df: pd.DataFrame,
                               post_transformers: list[BaseTransformer],
                               column_names: ColumnNames,
                               predictor: BasePredictor,
                               estimator_features: list[str],
                               return_features: bool,
                               add_train_prediction: bool = False
                               ) -> pd.DataFrame:
        pass


    def cross_validation_score(self, validation_df: pd.DataFrame, scorer: Optional[BaseScorer] = None ) -> float:
        if not scorer and not self.scorer:
            raise ValueError("scorer is not defined. Either pass into constructor or as argument to method")

        scorer = scorer or self.scorer
        return scorer.score(df=validation_df)

