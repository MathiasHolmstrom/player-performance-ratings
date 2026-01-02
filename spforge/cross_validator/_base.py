from abc import abstractmethod, ABC
from typing import Optional

from spforge import ColumnNames

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spforge.pipeline import Pipeline

from spforge.scorer import BaseScorer

from narwhals.typing import IntoFrameT, IntoFrameT


class CrossValidator(ABC):

    def __init__(
        self,
        scorer: Optional[BaseScorer],
        min_validation_date: str,
        estimator: "Pipeline",
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
        self, validation_df: IntoFrameT, scorer: Optional[BaseScorer] = None
    ) -> float:
        if not scorer and not self.scorer:
            raise ValueError(
                "scorer is not defined. Either pass into constructor or as argument to method"
            )

        scorer = scorer or self.scorer
        return scorer.score(df=validation_df)
