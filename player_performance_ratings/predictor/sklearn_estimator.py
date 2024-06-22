from typing import Optional, Any

import numpy as np
import pandas as pd
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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes_ = np.sort(np.unique(y))
        self.estimator.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)


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

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] <= 2:
            raise ValueError("OrdinalClassifier needs at least 3 classes")

        for i in range(self.classes_.shape[0] - 1):
            binary_y = (y > self.classes_[i]).astype(np.uint8)
            clf = clone(self.estimator)
            clf.fit(X, binary_y)
            self.clfs[i] = clf
            try:
                self.coef_.append(clf.coef_)
            except AttributeError:
                pass

        return self

    def predict_proba(self, X):
        clfs_probs = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.classes_):
            if i == 0:

                predicted.append(1 - clfs_probs[i][:, 1])
            elif i in clfs_probs:

                predicted.append(clfs_probs[i - 1][:, 1] - clfs_probs[i][:, 1])
            else:

                predicted.append(clfs_probs[i - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
