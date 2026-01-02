from typing import Optional, Any

import numpy as np
import pandas as pd
import polars as pl
import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT
from sklearn.base import BaseEstimator
from sklearn import clone
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression


class SkLearnWrapper(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        estimator: Any,
    ):
        self.estimator = estimator
        self.classes_ = []
        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y):
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        y_pd = y.to_pandas() if hasattr(y, 'to_pandas') else (y.to_native() if hasattr(y, 'to_native') else y)
        self.classes_ = np.sort(np.unique(y_pd))
        self.estimator.fit(X_pd, y_pd)

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        return self.estimator.predict_proba(X_pd)

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        return self.estimator.predict(X_pd)


class LGBMWrapper(BaseEstimator):
    """
    Wrapper for sklearn estimators that adds date-based sample weighting.
    Implements sklearn's BaseEstimator interface so it can be used seamlessly with Pipeline.
    """

    def __init__(
        self,
        estimator: Any,
        date_column: str,
        day_weight_epsilon: float = 400,
    ):
        """
        :param estimator: sklearn estimator to wrap
        :param date_column: Name of the date column in X (pandas DataFrame)
        :param day_weight_epsilon: Epsilon parameter for date weighting formula
        """
        self.estimator = estimator
        self.date_column = date_column
        self.day_weight_epsilon = day_weight_epsilon
        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y, sample_weight: Optional[np.ndarray] = None):
        """
        Fit the estimator with date-based sample weighting.
        
        :param X: Features DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :param y: Target Series
        :param sample_weight: Optional sample weights (will be combined with date weights)
        """
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        y_pd = y.to_pandas() if hasattr(y, 'to_pandas') else (y.to_native() if hasattr(y, 'to_native') else y)
        
        if self.date_column not in X_pd.columns:
            raise ValueError(f"date_column '{self.date_column}' not found in X.columns: {X_pd.columns.tolist()}")
        
        date_series = pd.to_datetime(X_pd[self.date_column])
        
        max_date = date_series.max()
        days_diff = (date_series - max_date).dt.total_seconds() / (24 * 60 * 60)
        
        # Calculate weights using formula from SklearnPredictor
        min_days_diff = days_diff.min()
        weights = (days_diff - min_days_diff + self.day_weight_epsilon) / (
            min_days_diff * -2 + self.day_weight_epsilon
        )
        
        # Combine with provided sample_weight if any
        if sample_weight is not None:
            combined_weights = weights.values * sample_weight
        else:
            combined_weights = weights.values
        
        # Drop date column before passing to sklearn (sklearn can't handle datetime columns)
        X_features = X_pd.drop(columns=[self.date_column])
        
        # Fit the estimator with combined weights
        self.estimator.fit(X_features, y_pd, sample_weight=combined_weights)
        
        # Store classes_ if estimator has predict_proba
        if hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_
        
        return self

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        """Predict using the wrapped estimator."""
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        # Convert to pandas
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        # Drop date column before passing to sklearn
        X_features = X_pd.drop(columns=[self.date_column]) if self.date_column in X_pd.columns else X_pd
        return self.estimator.predict(X_features)

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        """Predict probabilities using the wrapped estimator."""
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        if not hasattr(self.estimator, 'predict_proba'):
            raise AttributeError(f"{type(self.estimator).__name__} does not have predict_proba method")
        # Convert to pandas
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        # Drop date column before passing to sklearn
        X_features = X_pd.drop(columns=[self.date_column]) if self.date_column in X_pd.columns else X_pd
        return self.estimator.predict_proba(X_features)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        params = {
            'estimator': self.estimator,
            'date_column': self.date_column,
            'day_weight_epsilon': self.day_weight_epsilon,
        }
        if deep:
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f'estimator__{k}': v for k, v in estimator_params.items()})
        return params

    def set_params(self, **params) -> 'LGBMWrapper':
        """Set parameters for this estimator."""
        estimator_params = {}
        for key, value in params.items():
            if key.startswith('estimator__'):
                estimator_params[key[11:]] = value
            elif key == 'estimator':
                self.estimator = value
            elif key == 'date_column':
                self.date_column = value
            elif key == 'day_weight_epsilon':
                self.day_weight_epsilon = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        if estimator_params:
            self.estimator.set_params(**estimator_params)
        
        return self


class OrdinalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        estimator: Optional = None,
    ):
        self.estimator = estimator or LogisticRegression()
        self.clfs = {}
        self.classes_ = []
        self.coef_ = []
        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y):
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        y_pd = y.to_pandas() if hasattr(y, 'to_pandas') else (y.to_native() if hasattr(y, 'to_native') else y)
        self.classes_ = np.sort(np.unique(y_pd))
        if self.classes_.shape[0] <= 2:
            raise ValueError("OrdinalClassifier needs at least 3 classes")

        for i in range(self.classes_.shape[0] - 1):
            binary_y = y_pd > self.classes_[i]
            clf = clone(self.estimator)
            clf.fit(X_pd, binary_y)
            self.clfs[i] = clf
            try:
                self.coef_.append(clf.coef_)
            except AttributeError:
                pass

        return self

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT):
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        clfs_probs = {k: self.clfs[k].predict_proba(X_pd) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.classes_):
            if i == 0:

                predicted.append(1 - clfs_probs[i][:, 1])
            elif i in clfs_probs:

                predicted.append(clfs_probs[i - 1][:, 1] - clfs_probs[i][:, 1])
            else:

                predicted.append(clfs_probs[i - 1][:, 1])
        return np.vstack(predicted).T

    @nw.narwhalify
    def predict(self, X: IntoFrameT):
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        return np.argmax(self.predict_proba(X), axis=1)


class GranularityEstimator(BaseEstimator):
    """
    Wrapper for sklearn estimators that trains separate estimators per granularity level.
    Implements sklearn's BaseEstimator interface so it can be used seamlessly with Pipeline.
    """

    def __init__(
            self,
            estimator,
            granularity_column_name: str,
    ):
        """
        :param estimator: sklearn estimator to wrap
        :param granularity_column_name: Name of the column in X (pandas DataFrame) to group by
        """
        self.estimator = estimator
        self.granularity_column_name = granularity_column_name
        self._granularity_estimators = {}
        self.classes_ = {}
        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y, sample_weight: Optional[np.ndarray] = None):
        """
        Fit separate estimators for each granularity level.

        :param X: Features DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :param y: Target Series
        :param sample_weight: Optional sample weights
        """
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        
        # Convert to pandas for operations
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        y_pd = y.to_pandas() if hasattr(y, 'to_pandas') else (y.to_native() if hasattr(y, 'to_native') else y)
        
        if self.granularity_column_name not in X_pd.columns:
            raise ValueError(
                f"granularity_column_name '{self.granularity_column_name}' not found in X.columns: {X_pd.columns.tolist()}"
            )

        # Group by granularity
        granularity_values = X_pd[self.granularity_column_name].unique()

        self._granularity_estimators = {}
        self.classes_ = {}

        for granularity_value in granularity_values:
            # Get subset of data for this granularity
            mask = X_pd[self.granularity_column_name] == granularity_value
            X_group = X_pd[mask].drop(columns=[self.granularity_column_name])
            y_group = y_pd[mask]
            sample_weight_group = sample_weight[mask] if sample_weight is not None else None

            # Clone estimator for this granularity
            cloned_estimator = clone(self.estimator)

            # Fit on group data
            if sample_weight_group is not None:
                cloned_estimator.fit(X_group, y_group, sample_weight=sample_weight_group)
            else:
                cloned_estimator.fit(X_group, y_group)

            # Store estimator
            self._granularity_estimators[granularity_value] = cloned_estimator

            # Store classes if available
            if hasattr(cloned_estimator, 'classes_'):
                self.classes_[granularity_value] = cloned_estimator.classes_

        return self

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        """
        Predict using the appropriate estimator for each granularity level.

        :param X: Features DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :return: Predictions array
        """
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        
        # Convert to pandas
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        
        if self.granularity_column_name not in X_pd.columns:
            raise ValueError(
                f"granularity_column_name '{self.granularity_column_name}' not found in X.columns: {X_pd.columns.tolist()}"
            )

        if not self._granularity_estimators:
            raise ValueError("GranularityEstimator has not been fitted yet. Call fit() first.")

        # Initialize predictions array
        predictions = np.empty(len(X_pd), dtype=object)

        # Group predictions by granularity
        for granularity_value, estimator in self._granularity_estimators.items():
            mask = X_pd[self.granularity_column_name] == granularity_value
            X_group = X_pd[mask].drop(columns=[self.granularity_column_name])

            if len(X_group) > 0:
                group_predictions = estimator.predict(X_group)
                # Store predictions at correct positions (preserve order)
                mask_indices = np.where(mask)[0]
                for i, idx in enumerate(mask_indices):
                    predictions[idx] = group_predictions[i]

        # Convert to proper numpy array (handle mixed types if needed)
        try:
            return np.array(predictions)
        except (ValueError, TypeError):
            # If mixed types, return as object array
            return predictions

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        """
        Predict probabilities using the appropriate estimator for each granularity level.

        :param X: Features DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :return: Probabilities array
        """
        if isinstance(X.to_native() if hasattr(X, 'to_native') else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        
        # Convert to pandas
        X_pd = X.to_pandas() if isinstance(X.to_native(), pl.DataFrame) else X.to_native()
        
        if self.granularity_column_name not in X_pd.columns:
            raise ValueError(
                f"granularity_column_name '{self.granularity_column_name}' not found in X.columns: {X_pd.columns.tolist()}"
            )

        if not self._granularity_estimators:
            raise ValueError("GranularityEstimator has not been fitted yet. Call fit() first.")

        # Determine number of classes from first estimator
        first_estimator = next(iter(self._granularity_estimators.values()))
        if not hasattr(first_estimator, 'predict_proba'):
            raise AttributeError("Estimator does not have predict_proba method")

        # Get shape from first group to determine n_classes
        first_granularity = next(iter(self._granularity_estimators.keys()))
        first_mask = X_pd[self.granularity_column_name] == first_granularity
        if first_mask.sum() > 0:
            first_X_group = X_pd[first_mask].drop(columns=[self.granularity_column_name])
            first_probs = first_estimator.predict_proba(first_X_group)
            n_classes = first_probs.shape[1]
        else:
            raise ValueError("No data found for any granularity level")

        # Initialize probabilities array
        probabilities = np.zeros((len(X_pd), n_classes), dtype=float)

        # Group predictions by granularity
        for granularity_value, estimator in self._granularity_estimators.items():
            if not hasattr(estimator, 'predict_proba'):
                raise AttributeError(
                    f"Estimator for granularity '{granularity_value}' does not have predict_proba method"
                )

            mask = X_pd[self.granularity_column_name] == granularity_value
            X_group = X_pd[mask].drop(columns=[self.granularity_column_name])

            if len(X_group) > 0:
                group_probs = estimator.predict_proba(X_group)
                # Store probabilities at correct positions (preserve order)
                mask_indices = np.where(mask)[0]
                for i, idx in enumerate(mask_indices):
                    probabilities[idx] = group_probs[i]

        return probabilities

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        params = {
            'estimator': self.estimator,
            'granularity_column_name': self.granularity_column_name,
        }
        if deep:
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f'estimator__{k}': v for k, v in estimator_params.items()})
        return params

    def set_params(self, **params) -> 'GranularityEstimator':
        """Set parameters for this estimator."""
        estimator_params = {}
        for key, value in params.items():
            if key.startswith('estimator__'):
                estimator_params[key[11:]] = value
            elif key == 'estimator':
                self.estimator = value
            elif key == 'granularity_column_name':
                self.granularity_column_name = value
            else:
                raise ValueError(f"Unknown parameter: {key}")

        if estimator_params:
            self.estimator.set_params(**estimator_params)

        return self

