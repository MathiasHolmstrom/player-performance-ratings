from typing import Optional

from spforge.predictor import BasePredictor

from spforge.transformers.base_transformer import (
    BaseTransformer,
)
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT


class PredictorTransformer(BaseTransformer):
    """
    Transformer that uses a predictor to generate predictions on the dataset
    This is useful if you want to use the output of a feature as input for another model
    """

    def __init__(self, predictor: BasePredictor, features: list[str] = None):
        """
        :param predictor: The predictor to use to add add new prediction-columns to the dataset
        :param features: The features to use for the predictor
        """
        self.predictor = predictor
        super().__init__(
            features=features, features_out=[f"{self.predictor.pred_column}"]
        )

    @nw.narwhalify
    def fit_transform(
        self, df: FrameT, column_names: Optional[None] = None
    ) -> IntoFrameT:
        self.predictor.train(df=df, features=self.features)
        return self.transform(df)

    @nw.narwhalify
    def transform(self, df: FrameT, cross_validate: bool = False) -> IntoFrameT:
        return self.predictor.predict(df=df)
