from typing import Any

import narwhals.stable.v2 as nw
from sklearn.base import BaseEstimator, TransformerMixin, clone


class EstimatorTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that fits an estimator and appends its predictions as a new column.
    """

    def __init__(
        self, estimator: Any, prediction_column_name: str, features: list[str] | None = None
    ):
        self.estimator = estimator
        self.prediction_column_name = prediction_column_name
        self.features = features

        self.estimator_ = None
        self.features_ = None

    @nw.narwhalify
    def fit(self, X, y):
        feats = self.features if self.features is not None else list(X.columns)
        self.features_ = list(feats)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X=X.select(self.features_), y=y)
        return self

    @nw.narwhalify
    def transform(self, X):
        if self.estimator_ is None or self.features_ is None:
            raise RuntimeError("EstimatorTransformer is not fitted")

        prediction = self.estimator_.predict(X=X.select(self.features_))
        return X.with_columns(
            nw.new_series(
                name=self.prediction_column_name,
                values=prediction,
                backend=nw.get_native_namespace(X),
            )
        ).select(self.get_feature_names_out())

    def get_feature_names_out(self, input_features=None):
        return [self.prediction_column_name]

    def set_output(self, *, transform=None):
        # route set_output to the fitted estimator if present; else to the template
        target = self.estimator_ if self.estimator_ is not None else self.estimator
        if hasattr(target, "set_output"):
            target.set_output(transform=transform)
        return self
