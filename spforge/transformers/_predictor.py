from typing import Any

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT


class EstimatorTransformer(BaseEstimator,TransformerMixin):
    """
    Transformer that uses an estimator to generate predictions on the dataset
    This is useful if you want to use the output of a feature as input for another model
    """

    def __init__(self, estimator: Any,prediction_column_name: str, features: list[str] | None = None):
        """
        :param estimator: The estimator (sklearn-compatible) to use to add new prediction-columns to the dataset
        :param features: The features to track (for BaseTransformer)
        """
        self.estimator = estimator
        self.features = features
        self.prediction_column_name = prediction_column_name

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y) :
        self.features = self.features or X.columns
        self.estimator.fit(X=X.select(self.features),y=y)
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        prediction =  self.estimator.predict(X=X.select(self.features))
        return X.with_columns(
            nw.new_series(
                name=self.prediction_column_name, values=prediction, backend=nw.get_native_namespace(X)
            )
        )

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            return np.array([self.prediction_column_name], dtype=object)

        return np.array(list(input_features) + [self.prediction_column_name], dtype=object)

    def set_output(self, *, transform=None):
        if hasattr(self.estimator, "set_output"):
            self.estimator.set_output(transform=transform)
        return self


