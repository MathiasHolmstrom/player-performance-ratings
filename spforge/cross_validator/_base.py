from abc import abstractmethod, ABC
from typing import Optional

from spforge import ColumnNames

from spforge.predictor._base import BasePredictor
from spforge.scorer import BaseScorer

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
