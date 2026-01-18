from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from sklearn import clone
from sklearn.base import BaseEstimator


class GranularityEstimator(BaseEstimator):
    def __init__(
        self,
        estimator: Any,
        granularity_column_name: str,
    ):
        self.estimator = estimator
        self.granularity_column_name = granularity_column_name
        self._granularity_estimators = {}
        self.classes_ = {}
        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any, sample_weight: np.ndarray | None = None):
        X_pd = X.to_pandas()
        y_pd = y.to_numpy() if not isinstance(y, np.ndarray) else y
        if self.granularity_column_name not in X_pd.columns:
            raise ValueError(f"granularity_column_name '{self.granularity_column_name}' not found.")

        granularity_values = X_pd[self.granularity_column_name].unique()
        self._granularity_estimators = {}
        self.classes_ = {}

        for val in granularity_values:
            mask = X_pd[self.granularity_column_name] == val
            X_group = X_pd[mask].drop(columns=[self.granularity_column_name])
            y_group = y_pd[mask]
            sw_group = sample_weight[mask] if sample_weight is not None else None

            cloned_est = clone(self.estimator)
            if sw_group is not None:
                cloned_est.fit(X_group, y_group, sample_weight=sw_group)
            else:
                cloned_est.fit(X_group, y_group)

            self._granularity_estimators[val] = cloned_est
            if hasattr(cloned_est, "classes_"):
                self.classes_[val] = cloned_est.classes_
        return self

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        if len(self._granularity_estimators) == 0:
            raise ValueError("not been fitted")
        X_pd = X.to_pandas()
        predictions = np.empty(len(X_pd), dtype=object)

        for val, est in self._granularity_estimators.items():
            mask = X_pd[self.granularity_column_name] == val
            if mask.any():
                X_group = X_pd[mask].drop(columns=[self.granularity_column_name])
                predictions[mask] = est.predict(X_group)

        try:
            return np.array(predictions)
        except (ValueError, TypeError):
            return predictions

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        X_pd = X.to_pandas()
        first_est = next(iter(self._granularity_estimators.values()))
        n_classes = first_est.predict_proba(
            X_pd[:1].drop(columns=[self.granularity_column_name])
        ).shape[1]

        probabilities = np.zeros((len(X_pd), n_classes), dtype=float)

        for val, est in self._granularity_estimators.items():
            mask = X_pd[self.granularity_column_name] == val
            if mask.any():
                X_group = X_pd[mask].drop(columns=[self.granularity_column_name])
                probabilities[mask] = est.predict_proba(X_group)
        return probabilities

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "estimator": self.estimator,
            "granularity_column_name": self.granularity_column_name,
        }
        if deep and hasattr(self.estimator, "get_params"):
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f"estimator__ {k}": v for k, v in estimator_params.items()})
        return params

    def set_params(self, **params) -> "GranularityEstimator":
        for key, value in params.items():
            if key.startswith("estimator__"):
                self.estimator.set_params(**{key[11:]: value})
            else:
                setattr(self, key, value)
        return self
