import contextlib
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoFrameT
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, is_regressor
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
        self.estimator_ = None  # fitted clone

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
            # easiest/most reliable for now: go through pandas for datetime math
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

        # Always convert to pandas to preserve feature names for sklearn models
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

        # Always convert to pandas to preserve feature names for sklearn models
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

        # Always convert to pandas to preserve feature names for sklearn models
        X_features = X_features.to_pandas()

        return self.estimator_.predict_proba(X_features)


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimator: Any | None = None,
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
            with contextlib.suppress(AttributeError):
                self.coef_.append(clf.coef_)
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


class ConditionalEstimator(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        gate_estimator: Any,
        gate_distance_col: str,
        outcome_0_value: str | int,
        outcome_1_value: str | int,
        outcome_0_estimator: Any,
        outcome_1_estimator: Any,
        gate_distance_col_is_feature: bool = True,
    ):
        self.gate_estimator = gate_estimator
        self.gate_distance_col = gate_distance_col
        self.outcome_0_estimator = outcome_0_estimator
        self.outcome_1_estimator = outcome_1_estimator
        self.outcome_0_value = outcome_0_value
        self.outcome_1_value = outcome_1_value
        self.gate_distance_col_is_feature = gate_distance_col_is_feature

    @nw.narwhalify
    def fit(
        self, X: IntoFrameT, y: list[int] | np.ndarray, sample_weight: np.ndarray | None = None
    ):
        self.fitted_feats = (
            X.columns
            if self.gate_distance_col_is_feature
            else X.drop(self.gate_distance_col).columns
        )

        df = X.with_columns(
            nw.new_series(name="__target", values=y, backend=nw.get_native_namespace(X))
        )

        df = df.with_columns(
            nw.when(nw.col(self.gate_distance_col) > nw.col("__target"))
            .then(nw.lit(1))
            .otherwise(nw.lit(0))
            .alias("__gate_target")
        )

        y_gate = df["__gate_target"].to_numpy()
        self.gate_estimator.fit(df.select(self.fitted_feats).to_pandas(), y_gate)

        # Classes are only the unique training targets (sklearn contract)
        y_array = y if isinstance(y, np.ndarray) else np.array(y)
        self.classes_ = sorted(list(set(y_array)))

        # Compute diffs for outcome estimators
        df = df.with_columns(
            (nw.col("__target") - nw.col(self.gate_distance_col)).alias("__diff_gate")
        )

        df0_rows = df.filter(nw.col("__gate_target") == self.outcome_0_value)
        df1_rows = df.filter(nw.col("__gate_target") == self.outcome_1_value)

        X0 = df0_rows.select(self.fitted_feats).to_pandas()
        X1 = df1_rows.select(self.fitted_feats).to_pandas()

        # Train outcome estimators on DIFFS (target - gate_distance)
        y0 = df0_rows["__diff_gate"].to_numpy()
        y1 = df1_rows["__diff_gate"].to_numpy()

        self.outcome_0_estimator.fit(X0, y0)
        self.outcome_1_estimator.fit(X1, y1)

        self.outcome_0_classes_ = np.asarray(self.outcome_0_estimator.classes_, dtype=int)
        self.outcome_1_classes_ = np.asarray(self.outcome_1_estimator.classes_, dtype=int)

        return self

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        X_feats_pd = X.select(self.fitted_feats).to_pandas()
        gate_distance = X[self.gate_distance_col].to_numpy()

        n = len(gate_distance)
        classes = np.asarray(self.classes_, dtype=int)
        C = len(classes)

        # Get gate probabilities
        gate_proba = self.gate_estimator.predict_proba(X_feats_pd)
        gate_class_to_idx = {int(c): i for i, c in enumerate(self.gate_estimator.classes_)}

        p0 = gate_proba[:, gate_class_to_idx[int(self.outcome_0_value)]]
        p1 = gate_proba[:, gate_class_to_idx[int(self.outcome_1_value)]]

        # Get diff predictions from outcome estimators
        proba_diff0 = self.outcome_0_estimator.predict_proba(X_feats_pd)
        proba_diff1 = self.outcome_1_estimator.predict_proba(X_feats_pd)

        diff0_to_idx = {int(d): j for j, d in enumerate(self.outcome_0_classes_)}
        diff1_to_idx = {int(d): j for j, d in enumerate(self.outcome_1_classes_)}

        # Map diffs to target classes: target = gate_distance + diff
        # Assign out-of-bounds probability to nearest boundary to preserve gate probabilities
        def map_diff_to_y(proba_diff: np.ndarray, diff_to_idx: dict[int, int]) -> np.ndarray:
            out = np.zeros((n, C), dtype=float)
            class_to_idx = {int(c): j for j, c in enumerate(classes)}
            min_class = int(classes[0])
            max_class = int(classes[-1])

            for i in range(n):
                gd = int(gate_distance[i])
                for diff, k in diff_to_idx.items():
                    target = gd + diff
                    prob = proba_diff[i, k]

                    if target in class_to_idx:
                        # Target is in training classes
                        j = class_to_idx[target]
                        out[i, j] += prob
                    elif target < min_class:
                        # Below range: assign to minimum class
                        out[i, 0] += prob
                    else:
                        # Above range: assign to maximum class
                        out[i, C - 1] += prob

            return out

        y_proba0 = map_diff_to_y(proba_diff0, diff0_to_idx)
        y_proba1 = map_diff_to_y(proba_diff1, diff1_to_idx)

        # Mixture: P(y) = p0 * P(y|outcome_0) + p1 * P(y|outcome_1)
        out = (p0[:, None] * y_proba0) + (p1[:, None] * y_proba1)

        # Normalize the final mixture to sum to 1 (required by sklearn)
        row_sums = out.sum(axis=1, keepdims=True)
        out = np.where(row_sums > 0, out / row_sums, out)

        return out

    @nw.narwhalify
    def predict(self, X: Any) -> np.ndarray:
        """Predict most likely global expert label."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array(self.classes_)[idx]


class FrequencyBucketingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator: Any, min_prob: float = 0.003):
        if min_prob <= 0 or min_prob > 1.0:
            raise ValueError(f"min_prob must be in (0, 1], got {min_prob}")
        self.estimator = estimator
        self.min_prob = min_prob
        self.classes_ = None
        self.estimator_ = None
        self._class_to_bucket = None
        self._bucket_to_classes = None

    def _find_nearest_left(self, idx: int, bucketed: list[bool]) -> int | None:
        for i in range(idx - 1, -1, -1):
            if not bucketed[i]:
                return i
        return None

    def _find_nearest_right(self, idx: int, bucketed: list[bool], n: int) -> int | None:
        for i in range(idx + 1, n):
            if not bucketed[i]:
                return i
        return None

    def _create_buckets(
        self, classes: np.ndarray, class_freqs: dict[float, float]
    ) -> tuple[dict[float, int], dict[int, list[float]]]:
        n = len(classes)
        bucketed = [False] * n
        class_to_bucket = {}
        bucket_to_classes = {}
        bucket_id = 0

        for idx in range(n):
            if bucketed[idx]:
                continue

            cls = classes[idx]
            freq = class_freqs[cls]

            if freq >= self.min_prob:
                class_to_bucket[cls] = bucket_id
                bucket_to_classes[bucket_id] = [cls]
                bucketed[idx] = True
                bucket_id += 1
            else:
                bucket = [idx]
                bucket_freq = freq
                bucketed[idx] = True

                while bucket_freq < self.min_prob:
                    min_bucket_idx = min(bucket)
                    max_bucket_idx = max(bucket)

                    left_idx = self._find_nearest_left(min_bucket_idx, bucketed)
                    right_idx = self._find_nearest_right(max_bucket_idx, bucketed, n)

                    if left_idx is None and right_idx is None:
                        break

                    next_idx = None
                    if left_idx is not None and class_freqs[classes[left_idx]] < self.min_prob:
                        if right_idx is not None and class_freqs[classes[right_idx]] < self.min_prob:
                            dist_left = min_bucket_idx - left_idx
                            dist_right = right_idx - max_bucket_idx
                            next_idx = left_idx if dist_left <= dist_right else right_idx
                        else:
                            next_idx = left_idx
                    elif right_idx is not None and class_freqs[classes[right_idx]] < self.min_prob:
                        next_idx = right_idx

                    if next_idx is None:
                        break

                    bucket.append(next_idx)
                    bucket_freq += class_freqs[classes[next_idx]]
                    bucketed[next_idx] = True

                bucket_classes = [classes[i] for i in bucket]
                for cls_in_bucket in bucket_classes:
                    class_to_bucket[cls_in_bucket] = bucket_id
                bucket_to_classes[bucket_id] = sorted(bucket_classes)

                bucket_id += 1

        return class_to_bucket, bucket_to_classes

    @nw.narwhalify
    def fit(
        self, X: IntoFrameT, y: list[int] | np.ndarray, sample_weight: np.ndarray | None = None
    ):
        y_array = y if isinstance(y, np.ndarray) else np.array(y)

        try:
            y_array = y_array.astype(float)
        except (ValueError, TypeError) as e:
            raise ValueError("FrequencyBucketingClassifier requires numeric classes") from e

        unique_classes, counts = np.unique(y_array, return_counts=True)

        if len(unique_classes) == 1:
            raise ValueError("FrequencyBucketingClassifier requires at least 2 classes")

        self.classes_ = np.sort(unique_classes)

        n_samples = len(y_array)
        class_freqs = {cls: count / n_samples for cls, count in zip(unique_classes, counts)}

        self._class_to_bucket, self._bucket_to_classes = self._create_buckets(
            self.classes_, class_freqs
        )

        y_bucketed = np.array([self._class_to_bucket[cls] for cls in y_array])

        X_pd = X.to_pandas()
        self.estimator_ = clone(self.estimator)

        if sample_weight is not None:
            self.estimator_.fit(X_pd, y_bucketed, sample_weight=sample_weight)
        else:
            self.estimator_.fit(X_pd, y_bucketed)

        return self

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("FrequencyBucketingClassifier not fitted. Call fit() first.")

        X_pd = X.to_pandas()
        bucket_proba = self.estimator_.predict_proba(X_pd)
        bucket_classes = self.estimator_.classes_

        n_samples = len(X_pd)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=float)

        for bucket_idx, bucket_id in enumerate(bucket_classes):
            bucket_id = int(bucket_id)
            if bucket_id not in self._bucket_to_classes:
                continue

            classes_in_bucket = self._bucket_to_classes[bucket_id]
            n_classes_in_bucket = len(classes_in_bucket)

            for cls in classes_in_bucket:
                class_idx = np.where(self.classes_ == cls)[0][0]
                proba[:, class_idx] = bucket_proba[:, bucket_idx] / n_classes_in_bucket

        row_sums = proba.sum(axis=1, keepdims=True)
        proba = np.where(row_sums > 0, proba / row_sums, proba)

        return proba

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
