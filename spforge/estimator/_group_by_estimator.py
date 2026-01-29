from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from sklearn import clone
from sklearn.base import BaseEstimator

from spforge.transformers._other_transformer import GroupByReducer


class GroupByEstimator(BaseEstimator):
    def __init__(
        self,
        estimator: Any,
        granularity: list[str] | None = None,
        aggregation_weight: str | None = None,
    ):
        self.estimator = estimator
        self.granularity = granularity or []
        self.aggregation_weight = aggregation_weight
        self._reducer = GroupByReducer(self.granularity, aggregation_weight=aggregation_weight)
        self._est = None

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted_", False)

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any, sample_weight: np.ndarray | None = None):
        X = X.to_pandas()
        # Backwards compatibility: old pickled objects may not have aggregation_weight
        agg_weight = getattr(self, "aggregation_weight", None)
        self._reducer = GroupByReducer(self.granularity, aggregation_weight=agg_weight)
        X_red = nw.from_native(self._reducer.fit_transform(X))
        y_red, sw_red = self._reducer.reduce_y(X, y, sample_weight=sample_weight)

        self._est = clone(self.estimator)
        if sw_red is not None:
            self._est.fit(X_red.drop(self.granularity).to_pandas(), y_red, sample_weight=sw_red)
        else:
            self._est.fit(X_red.drop(self.granularity).to_pandas(), y_red)

        self.estimator_ = self._est
        self._is_fitted_ = True

        if hasattr(self._est, "classes_"):
            self.classes_ = self._est.classes_

        return self

    @nw.narwhalify
    def predict(self, X: IntoFrameT):
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("GroupByEstimator not fitted. Call fit() first.")
        X_red = nw.from_native(self._reducer.transform(X))
        predicted = self._est.predict(X_red.drop(self.granularity).to_pandas())
        return self._return_predicted(X=X, X_red=X_red, predicted=predicted)

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("GroupByEstimator not fitted. Call fit() first.")
        X_red = nw.from_native(self._reducer.transform(X))
        predicted = self._est.predict_proba(X_red.drop(self.granularity).to_pandas())
        return self._return_predicted(X=X, X_red=X_red, predicted=predicted)

    def _return_predicted(
        self, X: IntoFrameT, X_red: IntoFrameT, predicted: np.ndarray
    ) -> np.ndarray:
        X_red = X_red.with_columns(
            nw.new_series(
                values=predicted.tolist(),
                name="__predicted",
                backend=nw.get_native_namespace(X_red),
            )
        )
        joined = X.join(
            X_red.select([*self.granularity, "__predicted"]),
            on=self.granularity,
            how="left",
        )

        return np.vstack(joined["__predicted"].to_list())
