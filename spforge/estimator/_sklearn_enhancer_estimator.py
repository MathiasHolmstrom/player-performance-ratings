from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoFrameT
from sklearn import clone
from sklearn.base import BaseEstimator, is_regressor


class SkLearnEnhancerEstimator(BaseEstimator):
    def __init__(
        self,
        estimator: Any,
        date_column: str | None = None,
        day_weight_epsilon: float | None = None,
        min_prediction: float | None = None,
        max_prediction: float | None = None,
    ):
        self.estimator = estimator
        self.date_column = date_column
        self.day_weight_epsilon = day_weight_epsilon
        self.min_prediction = min_prediction
        self.max_prediction = max_prediction
        self.classes_ = []
        self.estimator_ = None

    @property
    def context_features(self) -> list[str]:
        """Returns columns needed for fitting but not for the wrapped estimator.

        Returns date_column if configured for temporal weighting.
        """
        return [self.date_column] if self.date_column else []

    def _resolve_date_column(self, cols: list[str]) -> str | None:
        if not self.date_column:
            return None

        if self.date_column in cols:
            return self.date_column

        suffix = f"__{self.date_column}"
        matches = [c for c in cols if c.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"date_column '{self.date_column}' is ambiguous after preprocessing. "
                f"Matches: {matches}"
            )
        raise ValueError(f"Could not find {self.date_column}. Available columns {cols}")

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any, sample_weight: np.ndarray | None = None):
        y = (
            y.to_numpy()
            if hasattr(y, "to_numpy")
            else (np.asarray(y) if not isinstance(y, np.ndarray) else y)
        )

        cols = list(X.columns)
        resolved_date_col = self._resolve_date_column(cols)

        combined_weights = sample_weight

        if resolved_date_col and self.day_weight_epsilon is not None:
            date_series = pd.to_datetime(X.to_pandas()[resolved_date_col])
            max_date = date_series.max()
            days_diff = (date_series - max_date).dt.total_seconds() / (24 * 60 * 60)
            min_diff = days_diff.min()

            denom = min_diff * -2 + self.day_weight_epsilon
            date_weights = (days_diff - min_diff + self.day_weight_epsilon) / denom
            date_weights = date_weights.to_numpy()

            if combined_weights is not None:
                combined_weights = date_weights * combined_weights
            else:
                combined_weights = date_weights

        X_features = X.drop([resolved_date_col]) if resolved_date_col else X

        X_features = X_features.to_pandas()

        self.estimator_ = clone(self.estimator)
        if combined_weights is not None:
            self.estimator_.fit(X_features, y, sample_weight=combined_weights)
        else:
            self.estimator_.fit(X_features, y)

        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    @nw.narwhalify
    def predict(self, X: Any) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("SkLearnEnhancerEstimator is not fitted")

        resolved_date_col = self._resolve_date_column(list(X.columns))
        X_features = X.drop([resolved_date_col]) if resolved_date_col else X

        X_features = X_features.to_pandas()

        preds = self.estimator_.predict(X_features)
        return self._clip_predictions(preds)

    def _clip_predictions(self, preds: np.ndarray) -> np.ndarray:
        if self.estimator_ is None:
            return preds
        if not is_regressor(self.estimator_):
            return preds

        if self.min_prediction is None and self.max_prediction is None:
            return preds
        lower = -np.inf if self.min_prediction is None else self.min_prediction
        upper = np.inf if self.max_prediction is None else self.max_prediction
        return np.clip(preds, lower, upper)

    @nw.narwhalify
    def predict_proba(self, X: Any) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("SkLearnEnhancerEstimator is not fitted")

        resolved_date_col = self._resolve_date_column(list(X.columns))
        X_features = X.drop([resolved_date_col]) if resolved_date_col else X

        X_features = X_features.to_pandas()

        return self.estimator_.predict_proba(X_features)
