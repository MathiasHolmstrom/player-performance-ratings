from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from sklearn.base import BaseEstimator, is_regressor

from spforge.transformers._base import PredictorTransformer


class NetOverPredictedTransformer(PredictorTransformer):
    def __init__(
        self,
        estimator: BaseEstimator | Any,
        features: list[str],
        target_name: str,
        net_over_predicted_col: str,
        pred_column: str | None = None,
    ):
        self.features_out = (
            [net_over_predicted_col, pred_column] if pred_column else [net_over_predicted_col]
        )
        self.features_out = features
        self.target_name = target_name
        self.estimator = estimator
        self.pred_column = pred_column or "__pred"
        self.net_over_predicted_col = net_over_predicted_col

        def get_deepest_estimator(estimator):
            """
            Walk down `.estimator` attributes until the deepest estimator is found.
            """
            while hasattr(estimator, "estimator"):
                estimator = estimator.estimator
            return estimator

        deepest = get_deepest_estimator(estimator)
        assert is_regressor(deepest)

    @nw.narwhalify
    def fit(
        self,
        X: IntoFrameT,
        y,
    ):
        y = y.to_numpy() if not isinstance(y, np.ndarray) else y
        self.estimator.fit(X.to_pandas(), y)
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        predictions = self.estimator.predict(X.to_pandas())
        X = X.with_columns(
            nw.new_series(
                name=self.pred_column, values=predictions, backend=nw.get_native_namespace(X)
            )
        )
        X = X.with_columns(
            (nw.col(self.target_name) - nw.col(self.pred_column)).alias(self.net_over_predicted_col)
        )
        return X.select(self.get_feature_names_out())

    def set_output(self, *, transform=None):
        pass

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [self.net_over_predicted_col]

    @property
    def context_features(self) -> list[str]:
        """Returns target column needed for residual computation."""
        return [self.target_name] if self.target_name else []
