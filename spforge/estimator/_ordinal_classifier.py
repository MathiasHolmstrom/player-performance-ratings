import contextlib
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


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
