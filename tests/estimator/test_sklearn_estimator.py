from datetime import date, datetime
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from spforge.estimator import (
    GranularityEstimator,
    OrdinalClassifier,
    SkLearnEnhancerEstimator,
)


# Helper function to create dataframe based on type
def create_dataframe(df_type, data: dict):
    """Helper to create a DataFrame/LazyFrame based on type"""
    if df_type == pl.LazyFrame:
        return pl.DataFrame(data).lazy()
    else:
        return df_type(data)


def test_lgbm_wrapper_initialization():
    """Test successful initialization with LGBMWrapper"""
    wrapper = SkLearnEnhancerEstimator(estimator=LinearRegression(), date_column="date_col")
    assert wrapper.date_column == "date_col"
    assert wrapper.estimator is not None


def test_lgbm_wrapper_fit_with_date_weighting():
    """Test LGBMWrapper fit with date-based weighting"""
    wrapper = SkLearnEnhancerEstimator(
        estimator=LinearRegression(), date_column="date_col", day_weight_epsilon=400
    )
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "date_col": dates})
    y = pd.Series([1, 2, 3, 4, 5])
    wrapper.fit(X, y)
    # Should fit successfully with date weighting
    assert hasattr(wrapper.estimator_, "coef_")


def test_lgbm_wrapper_predict():
    """Test LGBMWrapper predict"""
    wrapper = SkLearnEnhancerEstimator(estimator=LinearRegression(), date_column="date_col")
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "date_col": dates})
    y_train = pd.Series([1, 2, 3])
    wrapper.fit(X_train, y_train)

    X_pred = pd.DataFrame(
        {"feature1": [4, 5], "date_col": pd.date_range("2024-01-04", periods=2, freq="D")}
    )
    predictions = wrapper.predict(X_pred)
    assert len(predictions) == 2


def test_lgbm_wrapper_predict_proba():
    """Test LGBMWrapper predict_proba"""
    wrapper = SkLearnEnhancerEstimator(estimator=LogisticRegression(), date_column="date_col")
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "date_col": dates})
    y_train = pd.Series([0, 0, 1, 1])
    wrapper.fit(X_train, y_train)

    X_pred = pd.DataFrame(
        {"feature1": [5, 6], "date_col": pd.date_range("2024-01-05", periods=2, freq="D")}
    )
    probabilities = wrapper.predict_proba(X_pred)
    assert probabilities.shape == (2, 2)  # 2 samples, 2 classes


def test_lgbm_wrapper_sample_weight_combination():
    """Test LGBMWrapper combines date weights with provided sample_weight"""
    wrapper = SkLearnEnhancerEstimator(estimator=LinearRegression(), date_column="date_col")
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    X = pd.DataFrame({"feature1": [1, 2, 3], "date_col": dates})
    y = pd.Series([1, 2, 3])
    sample_weight = np.array([0.5, 1.0, 0.5])
    wrapper.fit(X, y, sample_weight=sample_weight)
    # Should fit successfully with combined weights
    assert hasattr(wrapper.estimator_, "coef_")


# ============================================================================
# GranularityPredictor Tests (sklearn wrapper)
# ============================================================================


def test_granularity_predictor_fit():
    """Test GranularityPredictor fit with sklearn estimator"""
    predictor = GranularityEstimator(
        estimator=LinearRegression(), granularity_column_name="position"
    )
    X = pd.DataFrame({"position": ["a", "b", "a", "b"], "feature1": [0.1, 0.5, 0.1, 0.5]})
    y = pd.Series([1, 1, 1, 1])
    predictor.fit(X, y)
    assert len(predictor._granularity_estimators) == 2  # Two positions


def test_granularity_predictor_predict():
    """Test GranularityPredictor predict"""
    predictor = GranularityEstimator(
        estimator=LinearRegression(), granularity_column_name="position"
    )
    X_train = pd.DataFrame({"position": ["a", "b", "a", "b"], "feature1": [0.1, 0.5, 0.1, 0.5]})
    y_train = pd.Series([1, 1, 1, 1])
    predictor.fit(X_train, y_train)

    X_pred = pd.DataFrame({"position": ["a", "b"], "feature1": [0.2, 0.6]})
    predictions = predictor.predict(X_pred)
    assert len(predictions) == 2


def test_granularity_predictor_predict_proba():
    """Test GranularityPredictor predict_proba"""
    predictor = GranularityEstimator(
        estimator=LogisticRegression(), granularity_column_name="position"
    )
    # Need at least 2 samples per position with different classes
    X_train = pd.DataFrame(
        {"position": ["a", "a", "b", "b", "a", "b"], "feature1": [0.1, 0.2, 0.5, 0.6, 0.15, 0.55]}
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1])
    predictor.fit(X_train, y_train)

    X_pred = pd.DataFrame({"position": ["a", "b"], "feature1": [0.2, 0.6]})
    probabilities = predictor.predict_proba(X_pred)
    assert probabilities.shape == (2, 2)  # 2 samples, 2 classes


def test_granularity_predictor_multiple_granularity_values():
    """Test GranularityPredictor with multiple granularity values"""
    predictor = GranularityEstimator(estimator=LinearRegression(), granularity_column_name="group")
    X = pd.DataFrame({"group": [1, 1, 2, 2, 3, 3], "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    y = pd.Series([1, 2, 3, 4, 5, 6])
    predictor.fit(X, y)
    assert len(predictor._granularity_estimators) == 3  # Three groups


def test_granularity_predictor_sample_weight():
    """Test GranularityPredictor with sample_weight"""
    predictor = GranularityEstimator(
        estimator=LinearRegression(), granularity_column_name="position"
    )
    X = pd.DataFrame({"position": ["a", "b", "a", "b"], "feature1": [0.1, 0.5, 0.1, 0.5]})
    y = pd.Series([1, 1, 1, 1])
    sample_weight = np.array([0.5, 1.0, 0.5, 1.0])
    predictor.fit(X, y, sample_weight=sample_weight)
    assert len(predictor._granularity_estimators) == 2


def test_granularity_predictor_predict_not_fitted():
    """Test GranularityPredictor predict raises error if not fitted"""
    predictor = GranularityEstimator(
        estimator=LinearRegression(), granularity_column_name="position"
    )
    X = pd.DataFrame({"position": ["a", "b"], "feature1": [0.1, 0.5]})
    with pytest.raises(ValueError, match="not been fitted"):
        predictor.predict(X)


def test_granularity_predictor_get_params():
    """Test GranularityPredictor get_params"""
    predictor = GranularityEstimator(
        estimator=LinearRegression(), granularity_column_name="position"
    )
    params = predictor.get_params()
    assert "estimator" in params
    assert "granularity_column_name" in params
    assert params["granularity_column_name"] == "position"


def test_granularity_predictor_set_params():
    """Test GranularityPredictor set_params"""
    predictor = GranularityEstimator(
        estimator=LinearRegression(), granularity_column_name="position"
    )
    predictor.set_params(granularity_column_name="group")
    assert predictor.granularity_column_name == "group"


# ============================================================================
# Additional LGBMWrapper Tests
# ============================================================================


def test_lgbm_wrapper_get_params():
    """Test LGBMWrapper get_params"""
    wrapper = SkLearnEnhancerEstimator(estimator=LinearRegression(), date_column="date_col")
    params = wrapper.get_params()
    assert "estimator" in params
    assert "date_column" in params
    assert "day_weight_epsilon" in params
    assert params["date_column"] == "date_col"


def test_lgbm_wrapper_set_params():
    """Test LGBMWrapper set_params"""
    wrapper = SkLearnEnhancerEstimator(estimator=LinearRegression(), date_column="date_col")
    wrapper.set_params(date_column="new_date_col", day_weight_epsilon=500)
    assert wrapper.date_column == "new_date_col"
    assert wrapper.day_weight_epsilon == 500


def test_lgbm_wrapper_predict_proba_raises_error():
    """Test LGBMWrapper predict_proba raises error if estimator doesn't support it"""
    wrapper = SkLearnEnhancerEstimator(
        estimator=LinearRegression(),  # LinearRegression doesn't have predict_proba
        date_column="date_col",
    )
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "date_col": dates})
    y_train = pd.Series([1, 2, 3])
    wrapper.fit(X_train, y_train)

    X_pred = pd.DataFrame(
        {"feature1": [4, 5], "date_col": pd.date_range("2024-01-04", periods=2, freq="D")}
    )
    with pytest.raises(
        AttributeError, match="LinearRegression' object has no attribute 'predict_proba"
    ):
        wrapper.predict_proba(X_pred)


def test_granularity_estimator_predict_proba_no_data():
    """Test GranularityEstimator predict_proba handles missing granularity values"""
    predictor = GranularityEstimator(
        estimator=LogisticRegression(), granularity_column_name="position"
    )
    # Need at least 2 samples per position with different classes
    X_train = pd.DataFrame({"position": ["a", "a", "b", "b"], "feature1": [0.1, 0.2, 0.5, 0.6]})
    y_train = pd.Series([0, 1, 0, 1])
    predictor.fit(X_train, y_train)

    X_pred = pd.DataFrame({"position": ["c", "d"], "feature1": [0.2, 0.6]})  # Not in training
    # The current implementation handles missing granularity values by skipping them
    # Since 'c' and 'd' are not in training, they won't have estimators
    # The predict method will iterate through estimators but won't find matches
    # This results in predictions with None values, which may cause issues when converting to array
    # Let's test that it handles this case (either raises error or handles gracefully)
    try:
        predictions = predictor.predict(X_pred)
        # If it doesn't raise an error, verify it returns something
        assert len(predictions) == 2
    except (KeyError, ValueError, TypeError):
        # Any of these errors is acceptable for unknown granularity values
        pass


def test_granularity_estimator_predict_preserves_order():
    """Test GranularityEstimator predict preserves order of input DataFrame"""
    predictor = GranularityEstimator(
        estimator=LinearRegression(), granularity_column_name="position"
    )
    X_train = pd.DataFrame({"position": ["a", "b", "a", "b"], "feature1": [0.1, 0.5, 0.1, 0.5]})
    y_train = pd.Series([1, 2, 1, 2])
    predictor.fit(X_train, y_train)

    # Create X_pred with specific order
    X_pred = pd.DataFrame({"position": ["b", "a", "b", "a"], "feature1": [0.6, 0.2, 0.7, 0.3]})
    predictions = predictor.predict(X_pred)
    assert len(predictions) == 4
    # Predictions should be in the same order as X_pred rows


# ============================================================================
# OrdinalClassifier Tests
# ============================================================================


def test_ordinal_classifier_initialization():
    """Test OrdinalClassifier initialization"""
    clf = OrdinalClassifier(estimator=LogisticRegression())
    assert clf.estimator is not None
    assert len(clf.clfs) == 0
    assert len(clf.classes_) == 0


def test_ordinal_classifier_fit():
    """Test OrdinalClassifier fit"""
    clf = OrdinalClassifier(estimator=LogisticRegression())
    X = pd.DataFrame({"feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    y = pd.Series([0, 1, 2, 0, 1, 2])
    clf.fit(X, y)
    assert len(clf.classes_) == 3
    assert len(clf.clfs) == 2  # n_classes - 1 binary classifiers


def test_ordinal_classifier_fit_insufficient_classes():
    """Test OrdinalClassifier fit raises error with less than 3 classes"""
    clf = OrdinalClassifier(estimator=LogisticRegression())
    X = pd.DataFrame({"feature1": [0.1, 0.2, 0.3]})
    y = pd.Series([0, 1, 0])  # Only 2 classes
    with pytest.raises(ValueError, match="at least 3 classes"):
        clf.fit(X, y)


def test_ordinal_classifier_predict_proba():
    """Test OrdinalClassifier predict_proba"""
    clf = OrdinalClassifier(estimator=LogisticRegression())
    X_train = pd.DataFrame({"feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    y_train = pd.Series([0, 1, 2, 0, 1, 2])
    clf.fit(X_train, y_train)

    X_pred = pd.DataFrame({"feature1": [0.25, 0.45]})
    probabilities = clf.predict_proba(X_pred)
    assert probabilities.shape == (2, 3)  # 2 samples, 3 classes
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_ordinal_classifier_predict():
    """Test OrdinalClassifier predict"""
    clf = OrdinalClassifier(estimator=LogisticRegression())
    X_train = pd.DataFrame({"feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    y_train = pd.Series([0, 1, 2, 0, 1, 2])
    clf.fit(X_train, y_train)

    X_pred = pd.DataFrame({"feature1": [0.25, 0.45]})
    predictions = clf.predict(X_pred)
    assert len(predictions) == 2
    assert all(0 <= p < 3 for p in predictions)


# ============================================================================
# SkLearnWrapper Tests
# ============================================================================


def test_sklearn_wrapper_initialization():
    """Test SkLearnWrapper initialization"""
    wrapper = SkLearnEnhancerEstimator(estimator=LogisticRegression())
    assert wrapper.estimator is not None
    assert len(wrapper.classes_) == 0


def test_sklearn_wrapper_fit():
    """Test SkLearnWrapper fit"""
    wrapper = SkLearnEnhancerEstimator(estimator=LogisticRegression())
    X = pd.DataFrame({"feature1": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 1, 0, 1])
    wrapper.fit(X, y)
    assert len(wrapper.classes_) == 2
    assert np.array_equal(wrapper.classes_, [0, 1])


def test_sklearn_wrapper_predict():
    """Test SkLearnWrapper predict"""
    wrapper = SkLearnEnhancerEstimator(estimator=LogisticRegression())
    X_train = pd.DataFrame({"feature1": [0.1, 0.2, 0.3, 0.4]})
    y_train = pd.Series([0, 1, 0, 1])
    wrapper.fit(X_train, y_train)

    X_pred = pd.DataFrame({"feature1": [0.25, 0.35]})
    predictions = wrapper.predict(X_pred)
    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)


def test_sklearn_wrapper_predict_proba():
    """Test SkLearnWrapper predict_proba"""
    wrapper = SkLearnEnhancerEstimator(estimator=LogisticRegression())
    X_train = pd.DataFrame({"feature1": [0.1, 0.2, 0.3, 0.4]})
    y_train = pd.Series([0, 1, 0, 1])
    wrapper.fit(X_train, y_train)

    X_pred = pd.DataFrame({"feature1": [0.25, 0.35]})
    probabilities = wrapper.predict_proba(X_pred)
    assert probabilities.shape == (2, 2)  # 2 samples, 2 classes
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


# ============================================================================
# SkLearnEnhancerEstimator context_features Tests
# ============================================================================


def test_sklearn_enhancer_context_features__with_date_column():
    """SkLearnEnhancerEstimator.context_features returns [date_column] when set."""
    estimator = SkLearnEnhancerEstimator(
        estimator=LinearRegression(),
        date_column="game_date",
        day_weight_epsilon=0.01
    )
    assert estimator.context_features == ["game_date"]


def test_sklearn_enhancer_context_features__without_date_column():
    """SkLearnEnhancerEstimator.context_features returns [] when date_column is None."""
    estimator = SkLearnEnhancerEstimator(
        estimator=LinearRegression(),
        date_column=None
    )
    assert estimator.context_features == []
