import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spforge.estimator import (
    NegativeBinomialEstimator,
    NormalDistributionPredictor,
    DistributionManagerPredictor,
)
from spforge.pipeline import Pipeline


# ============================================================================
# NegativeBinomialPredictor Tests
# ============================================================================

def test_negative_binomial_predictor_initialization():
    """Test NegativeBinomialPredictor initialization"""
    predictor = NegativeBinomialEstimator(
        point_estimate_pred_column="pred",
        max_value=10,
        target="target",
        min_value=0
    )
    assert predictor.point_estimate_pred_column == "pred"
    assert predictor.max_value == 10
    assert predictor.min_value == 0
    assert predictor.target == "target"
    assert predictor.classes_ is not None
    assert len(predictor.classes_) == 11  # 0 to 10 inclusive


def test_negative_binomial_predictor_fit():
    """Test NegativeBinomialPredictor fit"""
    predictor = NegativeBinomialEstimator(
        point_estimate_pred_column="pred",
        max_value=10,
        target="target",
        min_value=0
    )
    X = pd.DataFrame({
        "pred": [5.0, 6.0, 7.0, 8.0, 9.0],
        "other_col": [1, 2, 3, 4, 5]
    })
    y = pd.Series([5, 6, 7, 8, 9])
    predictor.fit(X, y)
    assert predictor._mean_r is not None


def test_negative_binomial_predictor_fit_missing_column():
    """Test NegativeBinomialPredictor fit raises error if point_estimate_pred_column missing"""
    predictor = NegativeBinomialEstimator(
        point_estimate_pred_column="pred",
        max_value=10,
        target="target",
        min_value=0
    )
    X = pd.DataFrame({"other_col": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="point_estimate_pred_column"):
        predictor.fit(X, y)


def test_negative_binomial_predictor_predict_proba():
    """Test NegativeBinomialPredictor predict_proba"""
    predictor = NegativeBinomialEstimator(
        point_estimate_pred_column="pred",
        max_value=10,
        target="target",
        min_value=0
    )
    X_train = pd.DataFrame({
        "pred": [5.0, 6.0, 7.0],
        "other_col": [1, 2, 3]
    })
    y_train = pd.Series([5, 6, 7])
    predictor.fit(X_train, y_train)
    
    X_pred = pd.DataFrame({
        "pred": [5.5, 6.5],
        "other_col": [1, 2]
    })
    probabilities = predictor.predict_proba(X_pred)
    assert probabilities.shape == (2, 11)  # 2 samples, 11 classes (0-10)
    # Probabilities may not sum to exactly 1.0 due to discretization, but should be close
    assert np.all(probabilities.sum(axis=1) > 0.9)


def test_negative_binomial_predictor_predict():
    """Test NegativeBinomialPredictor predict"""
    predictor = NegativeBinomialEstimator(
        point_estimate_pred_column="pred",
        max_value=10,
        target="target",
        min_value=0
    )
    X_train = pd.DataFrame({
        "pred": [5.0, 6.0, 7.0],
        "other_col": [1, 2, 3]
    })
    y_train = pd.Series([5, 6, 7])
    predictor.fit(X_train, y_train)
    
    X_pred = pd.DataFrame({
        "pred": [5.5, 6.5],
        "other_col": [1, 2]
    })
    predictions = predictor.predict(X_pred)
    assert len(predictions) == 2
    assert all(0 <= p <= 10 for p in predictions)


def test_negative_binomial_predictor_predict_proba_not_fitted():
    """Test NegativeBinomialPredictor predict_proba raises error if not fitted"""
    predictor = NegativeBinomialEstimator(
        point_estimate_pred_column="pred",
        max_value=10,
        target="target",
        min_value=0
    )
    X = pd.DataFrame({"pred": [5.0, 6.0]})
    with pytest.raises(ValueError, match="not been fitted"):
        predictor.predict_proba(X)


def test_negative_binomial_predictor_pred_column_property():
    """Test NegativeBinomialPredictor pred_column property"""
    predictor = NegativeBinomialEstimator(
        point_estimate_pred_column="pred",
        max_value=10,
        target="target",
        min_value=0
    )
    assert predictor.pred_column == "target_probabilities"


# ============================================================================
# NormalDistributionPredictor Tests
# ============================================================================

def test_normal_distribution_predictor_initialization():
    """Test NormalDistributionPredictor initialization"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred",
        max_value=10,
        min_value=0,
        target="target",
        sigma=5.0
    )
    assert predictor.point_estimate_pred_column == "pred"
    assert predictor.max_value == 10
    assert predictor.min_value == 0
    assert predictor.target == "target"
    assert predictor.sigma == 5.0
    assert len(predictor.classes_) == 11  # 0 to 10 inclusive


def test_normal_distribution_predictor_fit():
    """Test NormalDistributionPredictor fit"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred",
        max_value=10,
        min_value=0,
        target="target",
        sigma=5.0
    )
    X = pd.DataFrame({
        "pred": [5.0, 6.0, 7.0],
        "other_col": [1, 2, 3]
    })
    y = pd.Series([5, 6, 7])
    predictor.fit(X, y)
    assert predictor._classes is not None
    assert len(predictor._classes) == 11


def test_normal_distribution_predictor_fit_missing_column():
    """Test NormalDistributionPredictor fit raises error if point_estimate_pred_column missing"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred",
        max_value=10,
        min_value=0,
        target="target"
    )
    X = pd.DataFrame({"other_col": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="point_estimate_pred_column"):
        predictor.fit(X, y)


def test_normal_distribution_predictor_predict_proba():
    """Test NormalDistributionPredictor predict_proba"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred",
        max_value=10,
        min_value=0,
        target="target",
        sigma=5.0
    )
    X_train = pd.DataFrame({
        "pred": [5.0, 6.0, 7.0],
        "other_col": [1, 2, 3]
    })
    y_train = pd.Series([5, 6, 7])
    predictor.fit(X_train, y_train)
    
    X_pred = pd.DataFrame({
        "pred": [5.5, 6.5],
        "other_col": [1, 2]
    })
    probabilities = predictor.predict_proba(X_pred)
    assert probabilities.shape == (2, 11)  # 2 samples, 11 classes (0-10)
    # Probabilities may not sum to exactly 1.0 due to discretization, but should be close
    # Normal distribution discretization can have lower coverage
    assert np.all(probabilities.sum(axis=1) > 0.7)


def test_normal_distribution_predictor_predict():
    """Test NormalDistributionPredictor predict"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred",
        max_value=10,
        min_value=0,
        target="target",
        sigma=5.0
    )
    X_train = pd.DataFrame({
        "pred": [5.0, 6.0, 7.0],
        "other_col": [1, 2, 3]
    })
    y_train = pd.Series([5, 6, 7])
    predictor.fit(X_train, y_train)
    
    X_pred = pd.DataFrame({
        "pred": [5.5, 6.5],
        "other_col": [1, 2]
    })
    predictions = predictor.predict(X_pred)
    assert len(predictions) == 2
    assert all(0 <= p <= 10 for p in predictions)


def test_normal_distribution_predictor_predict_proba_not_fitted():
    """Test NormalDistributionPredictor predict_proba raises error if not fitted"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred",
        max_value=10,
        min_value=0,
        target="target"
    )
    X = pd.DataFrame({"pred": [5.0, 6.0]})
    with pytest.raises(ValueError, match="not been fitted"):
        predictor.predict_proba(X)


def test_normal_distribution_predictor_pred_column_property():
    """Test NormalDistributionPredictor pred_column property"""
    predictor = NormalDistributionPredictor(
        point_estimate_pred_column="pred",
        max_value=10,
        min_value=0,
        target="target"
    )
    assert predictor.pred_column == "target_probabilities"


# ============================================================================
# DistributionManagerPredictor Tests
# ============================================================================

def test_distribution_manager_predictor_initialization():
    """Test DistributionManagerPredictor initialization"""
    point_predictor = Pipeline(
        estimator=LinearRegression(),
        target="target",
        features=["feature1"]
    )
    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction",
        max_value=10,
        min_value=0,
        target="target"
    )
    manager = DistributionManagerPredictor(
        point_predictor=point_predictor,
        distribution_predictor=distribution_predictor
    )
    assert manager.point_predictor == point_predictor
    assert manager.distribution_predictor == distribution_predictor


def test_distribution_manager_predictor_fit():
    """Test DistributionManagerPredictor fit"""
    point_predictor = Pipeline(
        estimator=LinearRegression(),
        target="target",
        features=["feature1"]
    )
    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction",
        max_value=10,
        min_value=0,
        target="target"
    )
    manager = DistributionManagerPredictor(
        point_predictor=point_predictor,
        distribution_predictor=distribution_predictor
    )
    
    X = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y = pd.Series([5, 6, 7, 8, 9])
    manager.fit(X, y)
    
    # Both predictors should be fitted
    assert hasattr(point_predictor.estimator, 'coef_')
    assert distribution_predictor._classes is not None


def test_distribution_manager_predictor_predict_proba():
    """Test DistributionManagerPredictor predict_proba"""
    point_predictor = Pipeline(
        estimator=LinearRegression(),
        target="target",
        features=["feature1"]
    )
    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction",
        max_value=10,
        min_value=0,
        target="target",
        sigma=5.0
    )
    manager = DistributionManagerPredictor(
        point_predictor=point_predictor,
        distribution_predictor=distribution_predictor
    )
    
    X_train = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y_train = pd.Series([5, 6, 7, 8, 9])
    manager.fit(X_train, y_train)
    
    X_pred = pd.DataFrame({
        "feature1": [6.0, 7.0]
    })
    probabilities = manager.predict_proba(X_pred)
    assert probabilities.shape == (2, 11)  # 2 samples, 11 classes (0-10)


def test_distribution_manager_predictor_predict():
    """Test DistributionManagerPredictor predict"""
    point_predictor = Pipeline(
        estimator=LinearRegression(),
        target="target",
        features=["feature1"]
    )
    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction",
        max_value=10,
        min_value=0,
        target="target",
        sigma=5.0
    )
    manager = DistributionManagerPredictor(
        point_predictor=point_predictor,
        distribution_predictor=distribution_predictor
    )
    
    X_train = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y_train = pd.Series([5, 6, 7, 8, 9])
    manager.fit(X_train, y_train)
    
    X_pred = pd.DataFrame({
        "feature1": [6.0, 7.0]
    })
    predictions = manager.predict(X_pred)
    assert len(predictions) == 2
    assert all(0 <= p <= 10 for p in predictions)


def test_distribution_manager_predictor_properties():
    """Test DistributionManagerPredictor properties"""
    point_predictor = Pipeline(
        estimator=LinearRegression(),
        target="target",
        features=["feature1", "feature2"]
    )
    distribution_predictor = NormalDistributionPredictor(
        point_estimate_pred_column="target_prediction",
        max_value=10,
        min_value=0,
        target="target"
    )
    manager = DistributionManagerPredictor(
        point_predictor=point_predictor,
        distribution_predictor=distribution_predictor
    )
    
    assert manager.target == "target"
    assert manager.pred_column == "target_probabilities"
    assert manager.features == ["feature1", "feature2"]

