from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd
from player_performance_ratings import ColumnNames

from player_performance_ratings.predictor._base import BasePredictor
from player_performance_ratings.scorer.score import BaseScorer
from player_performance_ratings.transformers.base_transformer import (
    BaseLagGenerator,
    BaseTransformer,
)


class CrossValidator(ABC):

    def __init__(self, scorer: Optional[BaseScorer], min_validation_date:str):
        self.scorer = scorer
        self.min_validation_date = min_validation_date


    @property
    def validation_column_name(self) -> str:
        return "is_validation"

    @abstractmethod
    def generate_validation_df(
        self,
        df: pd.DataFrame,
        column_names: ColumnNames,
        predictor: BasePredictor,
        estimator_features: list[str],
        return_features: bool,
        pre_lag_transformers: Optional[list[BaseTransformer]] = None,
        lag_generators: Optional[list[BaseLagGenerator]] = None,
        post_lag_transformers: Optional[list[BaseTransformer]] = None,
        add_train_prediction: bool = False,
    ) -> pd.DataFrame:
        pass

    def cross_validation_score(
        self, validation_df: pd.DataFrame, scorer: Optional[BaseScorer] = None
    ) -> float:
        if not scorer and not self.scorer:
            raise ValueError(
                "scorer is not defined. Either pass into constructor or as argument to method"
            )

        scorer = scorer or self.scorer
        return scorer.score(df=validation_df)
