from typing import Any, Optional

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoFrameT
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from spforge.transformers._other_transformer import GroupByReducer


class GroupByEstimator(BaseEstimator):
    def __init__(self, estimator: Any, granularity: list[str] | None = None):
        self.estimator = estimator
        self.granularity = granularity or []
        self._reducer = GroupByReducer(self.granularity)
        self._est = None

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted_", False)

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any, sample_weight: np.ndarray | None = None):
        X = X.to_pandas()
        self._reducer = GroupByReducer(self.granularity)
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
        predicted =  self._est.predict(X_red.drop(self.granularity).to_pandas())
        return self._return_predicted(X=X,X_red=X_red, predicted=predicted)


    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("GroupByEstimator not fitted. Call fit() first.")
        X_red = nw.from_native(self._reducer.transform(X))
        predicted = self._est.predict_proba(X_red.drop(self.granularity).to_pandas())
        return self._return_predicted(X=X,X_red=X_red, predicted=predicted)

    def _return_predicted(self,X: IntoFrameT, X_red: IntoFrameT, predicted: np.ndarray) -> np.ndarray:
        X_red = X_red.with_columns(
            nw.new_series(
                values=predicted.tolist(),
                name='__predicted',
                backend=nw.get_native_namespace(X_red)
            )
        )
        joined = X.join(
            X_red.select([*self.granularity, "__predicted"]),
            on=self.granularity,
            how="left",
        )

        return np.vstack(joined["__predicted"].to_list())

class SkLearnEnhancerEstimator(BaseEstimator):
    def __init__(
        self,
        estimator: Any,
        date_column: Optional[str] = None,
        day_weight_epsilon: Optional[float] = None,
    ):
        self.estimator = estimator
        self.date_column = date_column
        self.day_weight_epsilon = day_weight_epsilon
        self.classes_ = []
        self.estimator_ = None  # fitted clone

    def _resolve_date_column(self, cols: list[str]) -> Optional[str]:
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
    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None):
        y = y.to_numpy() if hasattr(y, "to_numpy") else (np.asarray(y) if not isinstance(y, np.ndarray) else y)

        cols = list(X.columns)
        resolved_date_col = self._resolve_date_column(cols)

        combined_weights = sample_weight

        if resolved_date_col and self.day_weight_epsilon is not None:
            # easiest/most reliable for now: go through pandas for datetime math
            date_series = pd.to_datetime(X.to_pandas()[resolved_date_col])
            max_date = date_series.max()
            days_diff = (date_series - max_date).dt.total_seconds() / (24 * 60 * 60)
            min_diff = days_diff.min()

            denom = (min_diff * -2 + self.day_weight_epsilon)
            date_weights = (days_diff - min_diff + self.day_weight_epsilon) / denom
            date_weights = date_weights.to_numpy()

            if combined_weights is not None:
                combined_weights = date_weights * combined_weights
            else:
                combined_weights = date_weights

        X_features = X.drop([resolved_date_col]) if resolved_date_col else X

        # if categorical present, convert to pandas (your current rule)
        cat_cols = [name for name, dtype in X_features.schema.items() if dtype == nw.Categorical]
        if cat_cols:
            X_features = X_features.to_pandas()

        # IMPORTANT: fit a clone, store as estimator_
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

        cat_cols = [name for name, dtype in X_features.schema.items() if dtype == nw.Categorical]
        if cat_cols:
            X_features = X_features.to_pandas()

        return self.estimator_.predict(X_features)

    @nw.narwhalify
    def predict_proba(self, X: Any) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("SkLearnEnhancerEstimator is not fitted")

        resolved_date_col = self._resolve_date_column(list(X.columns))
        X_features = X.drop([resolved_date_col]) if resolved_date_col else X

        cat_cols = [name for name, dtype in X_features.schema.items() if dtype == nw.Categorical]
        if cat_cols:
            X_features = X_features.to_pandas()

        return self.estimator_.predict_proba(X_features)



class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            estimator: Optional[Any] = None,
    ):
        self.estimator = estimator or LogisticRegression()
        self.clfs = {}
        self.classes_ = []
        self.coef_ = []
        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any):
        X_pd = X.to_pandas()
        y_pd = (
            y.to_pandas()
            if hasattr(y, "to_pandas")
            else (y.to_native() if hasattr(y, "to_native") else y)
        )
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
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        X_pd = X.to_pandas()
        clfs_probs = {k: self.clfs[k].predict_proba(X_pd) for k in self.clfs}
        predicted = []
        for i, _ in enumerate(self.classes_):
            if i == 0:
                predicted.append(1 - clfs_probs[i][:, 1])
            elif i in clfs_probs:
                predicted.append(clfs_probs[i - 1][:, 1] - clfs_probs[i][:, 1])
            else:
                predicted.append(clfs_probs[i - 1][:, 1])
        return np.vstack(predicted).T

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


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
        if len(self._granularity_estimators) ==0:
            raise ValueError('not been fitted')
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
        n_classes = first_est.predict_proba(X_pd[:1].drop(columns=[self.granularity_column_name])).shape[1]

        probabilities = np.zeros((len(X_pd), n_classes), dtype=float)

        for val, est in self._granularity_estimators.items():
            mask = X_pd[self.granularity_column_name] == val
            if mask.any():
                X_group = X_pd[mask].drop(columns=[self.granularity_column_name])
                probabilities[mask] = est.predict_proba(X_group)
        return probabilities

    def get_params(self, deep: bool = True) -> dict:
        params = {"estimator": self.estimator, "granularity_column_name": self.granularity_column_name}
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
