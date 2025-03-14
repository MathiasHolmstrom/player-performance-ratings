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
from narwhals.typing import FrameT, IntoFrameT


class CrossValidator(ABC):

    def __init__(
        self,
        scorer: Optional[BaseScorer],
        min_validation_date: str,
        predictor: BasePredictor,
    ):
        self.scorer = scorer
        self.min_validation_date = min_validation_date
        self.predictor = predictor

    @property
    def validation_column_name(self) -> str:
        return "is_validation"

    @abstractmethod
    def generate_validation_df(
        self,
        df: FrameT,
        column_names: ColumnNames,
        return_features: bool = False,
        add_train_prediction: bool = False,
    ) -> IntoFrameT:
        pass

    def cross_validation_score(
        self, validation_df: FrameT, scorer: Optional[BaseScorer] = None
    ) -> float:
        if not scorer and not self.scorer:
            raise ValueError(
                "scorer is not defined. Either pass into constructor or as argument to method"
            )

        scorer = scorer or self.scorer
        return scorer.score(df=validation_df)
