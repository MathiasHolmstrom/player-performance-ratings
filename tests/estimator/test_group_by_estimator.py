import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator

from spforge.estimator import GroupByEstimator


class _IdentityFeatureEstimator(BaseEstimator):
    """Predicts with the first feature column from the reduced frame."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, 0].to_numpy(dtype=float)
        return np.asarray(X)[:, 0].astype(float)


class _SimpleThresholdClassifier(BaseEstimator):
    """Binary classifier on first feature with stable class order [0, 1]."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        values = X.iloc[:, 0].to_numpy(dtype=float) if hasattr(X, "iloc") else np.asarray(X)[:, 0]
        return (values >= 0.5).astype(int)

    def predict_proba(self, X):
        values = X.iloc[:, 0].to_numpy(dtype=float) if hasattr(X, "iloc") else np.asarray(X)[:, 0]
        p1 = np.clip(values, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_runtime_extra_granularity_splits_batched_scenario_predictions(frame):
    """scenario_id should split groups when provided as runtime extra granularity."""
    train_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g1", "g1"],
            "teamid": ["A", "A", "B", "B"],
            "feature": [1.0, 1.0, 2.0, 2.0],
        }
    )
    y_train = np.array([1.0, 1.0, 2.0, 2.0])

    estimator = GroupByEstimator(
        estimator=_IdentityFeatureEstimator(),
        granularity=["gameid", "teamid"],
    )
    train = train_pd if frame == "pd" else pl.from_pandas(train_pd)
    estimator.fit(train, y_train)

    pred_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g1", "g1"],
            "teamid": ["A", "A", "B", "B"],
            "scenario_id": ["s1", "s2", "s1", "s2"],
            "feature": [10.0, 100.0, 20.0, 200.0],
        }
    )
    pred = pred_pd if frame == "pd" else pl.from_pandas(pred_pd)

    estimator._runtime_extra_granularity = ["scenario_id"]
    out = estimator.predict(pred)

    # Expected when scenario_id is part of grouping:
    # A/s1 -> 10, A/s2 -> 100, B/s1 -> 20, B/s2 -> 200.
    assert np.allclose(
        np.asarray(out, dtype=float).reshape(-1),
        np.array([10.0, 100.0, 20.0, 200.0]),
    )


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_runtime_extra_granularity_splits_batched_scenario_predict_proba(frame):
    """predict_proba should also split by runtime scenario_id granularity."""
    train_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g1", "g1"],
            "teamid": ["A", "A", "B", "B"],
            "feature": [0.2, 0.2, 0.8, 0.8],
        }
    )
    y_train = np.array([0, 0, 1, 1])

    estimator = GroupByEstimator(
        estimator=_SimpleThresholdClassifier(),
        granularity=["gameid", "teamid"],
    )
    train = train_pd if frame == "pd" else pl.from_pandas(train_pd)
    estimator.fit(train, y_train)

    pred_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g1", "g1"],
            "teamid": ["A", "A", "B", "B"],
            "scenario_id": ["s1", "s2", "s1", "s2"],
            "feature": [0.1, 0.9, 0.3, 0.7],
        }
    )
    pred = pred_pd if frame == "pd" else pl.from_pandas(pred_pd)

    estimator._runtime_extra_granularity = ["scenario_id"]
    proba = estimator.predict_proba(pred)

    expected = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.3, 0.7],
        ],
        dtype=float,
    )
    assert np.allclose(np.asarray(proba, dtype=float), expected)
