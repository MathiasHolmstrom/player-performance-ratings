from unittest.mock import MagicMock

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import nbinom
from sklearn.linear_model import LinearRegression

from spforge import ColumnNames
from spforge.estimator import (
    DistributionManagerPredictor,
    NegativeBinomialEstimator,
    NormalDistributionPredictor,
)
from spforge.pipeline import Pipeline


# Mocking internal dependencies for the sake of the test environment
class MockRolling:
    def __init__(self, **kwargs):
        self.features_out = ["rolling_feature"]

    def fit_transform(self, df, **kwargs):
        return df.with_columns(pl.lit(1.0).alias("rolling_feature"))

    def transform(self, df, **kwargs):
        return df.with_columns(pl.lit(1.0).alias("rolling_feature"))


@pytest.fixture
def sample_data():
    df = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "team_id": [10, 11, 10, 11],
            "player_id": [100, 101, 100, 101],
            "start_date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "update_match_id": [1, 1, 2, 2],
            "pred_mean": [10.5, 12.0, 9.0, 11.0],
            "actual": [10, 15, 8, 12],
        }
    )
    return df


@pytest.fixture
def col_names():
    return ColumnNames(
        match_id="match_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        update_match_id="update_match_id",
    )


def test_initialization_validation(col_names):
    """
    Test Scenario: Ensure the estimator raises a ValueError if r_specific_granularity
    is provided but column_names is missing.
    """
    with pytest.raises(ValueError, match="column names is not provided"):
        NegativeBinomialEstimator(
            point_estimate_pred_column="pred_mean",
            max_value=20,
            r_specific_granularity=["player_id"],
            column_names=None,  # This should trigger the error
        )


def test_fit_and_predict_shape(sample_data):
    """
    Test Scenario: Verify the basic fit/predict flow. predict_proba should return
    a matrix of shape (n_samples, max_value - min_value + 1).
    """
    min_val, max_val = 0, 5
    estimator = NegativeBinomialEstimator(
        point_estimate_pred_column="pred_mean", min_value=min_val, max_value=max_val
    )

    X = sample_data.drop("actual")
    y = sample_data["actual"]

    estimator.fit(X, y)
    probs = estimator.predict_proba(X)

    assert probs.shape == (len(sample_data), max_val - min_val + 1)
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities must sum to 1


def test_predict_mode_consistency(sample_data):
    """
    Test Scenario: Ensure the predict() method correctly returns the mode
    (the value with the highest probability) from predict_proba.
    """
    estimator = NegativeBinomialEstimator(point_estimate_pred_column="pred_mean", max_value=20)
    estimator.fit(sample_data.drop("actual"), sample_data["actual"])

    probs = estimator.predict_proba(sample_data)
    preds = estimator.predict(sample_data)

    # The index of the max probability should match the prediction (if min_value=0)
    expected_mode = np.argmax(probs, axis=1)
    np.testing.assert_array_equal(preds, expected_mode)


def test_predict_granularity_aggregation(sample_data, col_names):
    """
    Test Scenario: When predict_granularity is set (e.g., to 'match_id'),
    rows within the same group should receive the same probability distribution.
    """
    estimator = NegativeBinomialEstimator(
        point_estimate_pred_column="pred_mean", max_value=10, predict_granularity=["match_id"]
    )
    estimator.column_names = col_names

    estimator.fit(sample_data.drop("actual"), sample_data["actual"])
    probs = estimator.predict_proba(sample_data)

    np.testing.assert_array_almost_equal(probs[0], probs[1])
    np.testing.assert_array_almost_equal(probs[2], probs[3])


def test_unfitted_estimator_error(sample_data):
    """
    Test Scenario: Calling predict_proba before fit should raise a ValueError.
    """
    estimator = NegativeBinomialEstimator(point_estimate_pred_column="pred_mean", max_value=10)
    with pytest.raises(ValueError, match="has not been fitted yet"):
        estimator.predict_proba(sample_data)


def test_zero_division_robustness():
    """
    Test Scenario: Verify the estimator handles rows where the point estimate
    is 0 (which might cause division by zero in the p = r/(r+mu) formula).
    """
    df = pl.DataFrame({"pred_mean": [0.0, 10.0], "actual": [0, 10]})
    estimator = NegativeBinomialEstimator(point_estimate_pred_column="pred_mean", max_value=5)
    estimator.fit(df, df["actual"])

    # Should not crash, and predict_proba for 0.0 mean should peak at 0
    probs = estimator.predict_proba(df)
    assert probs[0, 0] > 0.5  # Most probability should be at index 0


# ============================================================================
# NormalDistributionPredictor Tests
# ============================================================================


def test_normal_distribution_predictor_initialization():
    """Test NormalDistributionPredictor initialization"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred", max_value=10, min_value=0, target="target", sigma=5.0
    )
    assert predictor.point_estimate_pred_column == "pred"
    assert predictor.max_value == 10
    assert predictor.min_value == 0
    assert predictor.sigma == 5.0
    assert len(predictor.classes_) == 11  # 0 to 10 inclusive


def test_normal_distribution_predictor_fit():
    """Test NormalDistributionPredictor fit"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred", max_value=10, min_value=0, target="target", sigma=5.0
    )
    X = pd.DataFrame({"pred": [5.0, 6.0, 7.0], "other_col": [1, 2, 3]})
    y = pd.Series([5, 6, 7])
    predictor.fit(X, y)
    assert predictor._classes is not None
    assert len(predictor._classes) == 11


def test_normal_distribution_predictor_fit_missing_column():
    """Test NormalDistributionPredictor fit raises error if point_estimate_pred_column missing"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred", max_value=10, min_value=0, target="target"
    )
    X = pd.DataFrame({"other_col": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="point_estimate_pred_column"):
        predictor.fit(X, y)


def test_normal_distribution_predictor_predict_proba():
    """Test NormalDistributionPredictor predict_proba"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred", max_value=10, min_value=0, target="target", sigma=5.0
    )
    X_train = pd.DataFrame({"pred": [5.0, 6.0, 7.0], "other_col": [1, 2, 3]})
    y_train = pd.Series([5, 6, 7])
    predictor.fit(X_train, y_train)

    X_pred = pd.DataFrame({"pred": [5.5, 6.5], "other_col": [1, 2]})
    probabilities = predictor.predict_proba(X_pred)
    assert probabilities.shape == (2, 11)  # 2 samples, 11 classes (0-10)
    # Probabilities may not sum to exactly 1.0 due to discretization, but should be close
    # Normal distribution discretization can have lower coverage
    assert np.all(probabilities.sum(axis=1) > 0.7)


def test_normal_distribution_predictor_predict():
    """Test NormalDistributionPredictor predict"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred", max_value=10, min_value=0, target="target", sigma=5.0
    )
    X_train = pd.DataFrame({"pred": [5.0, 6.0, 7.0], "other_col": [1, 2, 3]})
    y_train = pd.Series([5, 6, 7])
    predictor.fit(X_train, y_train)

    X_pred = pd.DataFrame({"pred": [5.5, 6.5], "other_col": [1, 2]})
    predictions = predictor.predict(X_pred)
    assert len(predictions) == 2
    assert all(0 <= p <= 10 for p in predictions)


def test_normal_distribution_predictor_predict_proba_not_fitted():
    """Test NormalDistributionPredictor predict_proba raises error if not fitted"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred", max_value=10, min_value=0, target="target"
    )
    X = pd.DataFrame({"pred": [5.0, 6.0]})
    with pytest.raises(ValueError, match="not been fitted"):
        predictor.predict_proba(X)


# ============================================================================
# DistributionManagerPredictor Tests
# ============================================================================


def test_distribution_manager_predictor_initialization():
    """Test DistributionManagerPredictor initialization"""
    point_predictor = Pipeline(
        estimator=LinearRegression(),
        feature_names=['x']
    )
    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction", max_value=10, min_value=0, target="target"
    )
    manager = DistributionManagerPredictor(
        point_estimator=point_predictor, distribution_estimator=distribution_predictor
    )
    assert manager.point_estimator == point_predictor
    assert manager.distribution_estimator == distribution_predictor


def test_distribution_manager_predictor_fit():
    """Test DistributionManagerPredictor fit"""


    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction", max_value=10, min_value=0, target="target"
    )
    manager = DistributionManagerPredictor(
        point_estimator=LinearRegression(), distribution_estimator=distribution_predictor
    )

    X = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([5, 6, 7, 8, 9], name="target")
    manager.fit(X, y)

    assert hasattr(manager.point_estimator, "coef_")
    assert distribution_predictor._classes is not None


def test_distribution_manager_predictor_predict_proba():
    """Test DistributionManagerPredictor predict_proba"""

    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction",
        max_value=10,
        min_value=0,
        target="target",
        sigma=5.0,
    )
    manager = DistributionManagerPredictor(
        point_estimator=LinearRegression(), distribution_estimator=distribution_predictor
    )

    X_train = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_train = pd.Series([5, 6, 7, 8, 9], name="target")
    manager.fit(X_train, y_train)

    X_pred = pd.DataFrame({"feature1": [6.0, 7.0]})
    probabilities = manager.predict_proba(X_pred)
    assert probabilities.shape == (2, 11)  # 2 samples, 11 classes (0-10)


def test_distribution_manager_predictor_predict():
    """Test DistributionManagerPredictor predict"""


    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction",
        max_value=10,
        min_value=0,
        target="target",
        sigma=5.0,
    )
    manager = DistributionManagerPredictor(
        point_estimator=LinearRegression(), distribution_estimator=distribution_predictor
    )

    X_train = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_train = pd.Series([5, 6, 7, 8, 9], name="target")
    manager.fit(X_train, y_train)

    X_pred = pd.DataFrame({"feature1": [6.0, 7.0]})
    predictions = manager.predict(X_pred)
    assert len(predictions) == 2
    assert all(0 <= p <= 10 for p in predictions)


def test_distribution_manager_predictor_properties():
    """Test DistributionManagerPredictor properties"""

    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction", max_value=10, min_value=0, target="target"
    )
    manager = DistributionManagerPredictor(
        point_estimator=LinearRegression(), distribution_estimator=distribution_predictor
    )

    X_train = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
    y_train = pd.Series([5, 6, 7], name="target")
    manager.fit(X_train, y_train)
