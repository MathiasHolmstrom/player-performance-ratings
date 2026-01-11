from typing import Any

import narwhals.stable.v2 as nw
from sklearn.base import clone

from spforge.transformers._base import PredictorTransformer


class EstimatorTransformer(PredictorTransformer):
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
        self.estimator_.fit(X=X.select(self.features_).to_pandas(), y=y)
        return self

    @nw.narwhalify
    def transform(self, X):
        if self.estimator_ is None or self.features_ is None:
            raise RuntimeError("EstimatorTransformer is not fitted")

        prediction = self.estimator_.predict(X=X.select(self.features_).to_pandas())
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

    @property
    def context_features(self) -> list[str]:
        """Returns date_column if estimator is SkLearnEnhancerEstimator."""
        context = []

        # Check wrapped estimator (may be nested)
        est = self.estimator
        while hasattr(est, 'estimator'):
            if hasattr(est, 'date_column') and est.date_column:
                context.append(est.date_column)
                return context
            est = est.estimator

        # Check top-level
        if hasattr(self.estimator, 'date_column') and self.estimator.date_column:
            context.append(self.estimator.date_column)

        return context
