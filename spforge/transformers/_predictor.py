from typing import TYPE_CHECKING

from sklearn.base import TransformerMixin

if TYPE_CHECKING:
    from spforge.pipeline import Pipeline

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT


class PredictorTransformer(TransformerMixin):
    """
    Transformer that uses an estimator to generate predictions on the dataset
    This is useful if you want to use the output of a feature as input for another model
    """

    def __init__(self, estimator: "Pipeline", features: list[str] = None):
        """
        :param estimator: The estimator (sklearn-compatible) to use to add new prediction-columns to the dataset
        :param features: The features to track (for BaseTransformer), not used by estimator
        """
        self.estimator = estimator
        super().__init__(features=features, features_out=[f"{self.estimator.pred_column}"])

    @nw.narwhalify
    def fit(self, df: IntoFrameT) :
        self.estimator.fit(X=df)
        return self

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        return self.estimator.predict(X=df, return_features=True)
