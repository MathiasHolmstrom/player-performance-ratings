from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin


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
