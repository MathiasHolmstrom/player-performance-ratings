from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spforge.autopipeline import AutoPipeline

from narwhals.typing import IntoFrameT

from spforge.scorer import BaseScorer


class CrossValidator(ABC):

    def __init__(
        self,
        scorer: BaseScorer | None,
        min_validation_date: str,
        estimator: "AutoPipeline",
    ):
        self.scorer = scorer
        self.min_validation_date = min_validation_date
        self.estimator = estimator

    @property
    def validation_column_name(self) -> str:
        return "is_validation"

    @abstractmethod
    def generate_validation_df(
        self,
        df: IntoFrameT,
        return_features: bool = False,
        add_train_prediction: bool = False,
    ) -> IntoFrameT:
        pass

    def cross_validation_score(
        self, validation_df: IntoFrameT, scorer: BaseScorer | None = None
    ) -> float:
        if not scorer and not self.scorer:
            raise ValueError(
                "scorer is not defined. Either pass into constructor or as argument to method"
            )

        scorer = scorer or self.scorer
        return scorer.score(df=validation_df)
