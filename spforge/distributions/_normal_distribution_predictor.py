import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from scipy.stats import norm
from sklearn.base import BaseEstimator


class NormalDistributionPredictor(BaseEstimator):

    def __init__(
        self,
        point_estimate_pred_column: str,
        max_value: int,
        min_value: int,
        target: str,
        sigma: float = 10.0,
    ):
        self.point_estimate_pred_column = point_estimate_pred_column
        self.max_value = max_value
        self.min_value = min_value
        self.target = target
        self.sigma = sigma
        self._classes = None
        self.classes_ = np.arange(min_value, max_value + 1)
        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y, sample_weight: np.ndarray | None = None):
        """
        Fit the normal distribution predictor.

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :param y: Target Series (unused, kept for sklearn interface)
        :param sample_weight: Optional sample weights (unused)
        """
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        if self.point_estimate_pred_column not in X.columns:
            raise ValueError(
                f"point_estimate_pred_column '{self.point_estimate_pred_column}' not found in X.columns: {X.columns}"
            )
        self._classes = np.arange(self.min_value, self.max_value + 1)
        return self

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        """
        Predict probability distributions.

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :return: Array of probability distributions (n_samples, n_classes)
        """
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        if self.point_estimate_pred_column not in X.columns:
            raise ValueError(
                f"point_estimate_pred_column '{self.point_estimate_pred_column}' not found in X.columns: {X.columns}"
            )
        if self._classes is None:
            raise ValueError(
                "NormalDistributionPredictor has not been fitted yet. Call fit() first."
            )

        lower_bounds = self._classes - 0.5
        upper_bounds = self._classes + 0.5

        mu_values = X[self.point_estimate_pred_column].to_list()
        probabilities = []
        for mu in mu_values:
            cdf_upper = norm.cdf(upper_bounds, loc=mu, scale=self.sigma)
            cdf_lower = norm.cdf(lower_bounds, loc=mu, scale=self.sigma)
            probs = cdf_upper - cdf_lower
            probabilities.append(probs)

        return np.array(probabilities)

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        """
        Predict point estimates (mode of distribution).

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :return: Array of predicted values
        """
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1) + self.min_value
