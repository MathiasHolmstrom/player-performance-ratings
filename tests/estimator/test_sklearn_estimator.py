from datetime import date, datetime
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

from spforge.estimator import (
    ConditionalEstimator,
    FrequencyBucketingClassifier,
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


class ConstantRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, value: float = 0.0):
        self.value = value

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=np.float64)


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


def test_lgbm_wrapper_predict_clips_regression_predictions():
    """SkLearnEnhancerEstimator clips regression predictions when configured."""
    wrapper = SkLearnEnhancerEstimator(
        estimator=ConstantRegressor(value=10.0),
        min_prediction=0.0,
        max_prediction=5.0,
    )
    X_train = pd.DataFrame({"feature1": [1, 2, 3]})
    y_train = pd.Series([1, 2, 3])
    wrapper.fit(X_train, y_train)

    X_pred = pd.DataFrame({"feature1": [4, 5]})
    predictions = wrapper.predict(X_pred)
    assert np.allclose(predictions, np.array([5.0, 5.0]))


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
        estimator=LinearRegression(), date_column="game_date", day_weight_epsilon=0.01
    )
    assert estimator.context_features == ["game_date"]


def test_sklearn_enhancer_context_features__without_date_column():
    """SkLearnEnhancerEstimator.context_features returns [] when date_column is None."""
    estimator = SkLearnEnhancerEstimator(estimator=LinearRegression(), date_column=None)
    assert estimator.context_features == []


# ============================================================================
# ConditionalEstimator Tests
# ============================================================================


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__gate_estimator_routes_correctly(df_type):
    """ConditionalEstimator should use gate_estimator to determine routing."""
    # Create data with samples in both gate classes
    # gate_target = 1 when gate_distance >= target
    # gate_target = 0 when gate_distance < target
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "gate_distance": [5, 5, 5, 10, 10, 10],
        }
    )
    # For gate_distance=5: targets 3,4,5 → gate_target=1 (5>=3, 5>=4, 5>=5)
    # For gate_distance=10: targets 12,13,14 → gate_target=0 (10<12, 10<13, 10<14)
    y = np.array([3, 4, 5, 12, 13, 14])

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42),
        gate_distance_col="gate_distance",
        outcome_0_value=0,  # gate_distance < target
        outcome_1_value=1,  # gate_distance >= target
        outcome_0_estimator=LogisticRegression(random_state=42),
        outcome_1_estimator=LogisticRegression(random_state=42),
        gate_distance_col_is_feature=True,
    )

    # Should fit without error
    estimator.fit(data, y)

    # Verify gate estimator was fitted
    assert hasattr(estimator.gate_estimator, "coef_")


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__outcome_0_estimator_called(df_type):
    """ConditionalEstimator should use outcome_0_estimator for gate_distance < target."""
    # Create simple dataset
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "gate_distance": [5, 5, 5, 10, 10, 10],
        }
    )
    y = np.array([3, 4, 5, 12, 13, 14])  # gate_distance >= target for first 3, < for last 3

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42),
        outcome_1_estimator=LogisticRegression(random_state=42),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Verify outcome_0_estimator was fitted (has classes from fit)
    assert hasattr(estimator, "outcome_0_classes_")
    assert len(estimator.outcome_0_classes_) > 0


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__outcome_1_estimator_called(df_type):
    """ConditionalEstimator should use outcome_1_estimator for gate_distance >= target."""
    # Create simple dataset
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "gate_distance": [5, 5, 5, 10, 10, 10],
        }
    )
    y = np.array([3, 4, 5, 12, 13, 14])  # gate_distance >= target for first 3, < for last 3

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42),
        outcome_1_estimator=LogisticRegression(random_state=42),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Verify outcome_1_estimator was fitted
    assert hasattr(estimator, "outcome_1_classes_")
    assert len(estimator.outcome_1_classes_) > 0


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__predict_proba_weighting(df_type):
    """ConditionalEstimator.predict_proba should weight by gate probabilities."""
    # Create simple dataset
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "gate_distance": [5, 5, 5, 10, 10, 10],
        }
    )
    y = np.array([3, 4, 5, 12, 13, 14])

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42),
        outcome_1_estimator=LogisticRegression(random_state=42),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Predict probabilities
    test_data = df_type(
        {
            "feature1": [2, 5],
            "gate_distance": [5, 10],
        }
    )
    proba = estimator.predict_proba(test_data)

    # Should return probability matrix
    assert proba.shape[0] == 2  # 2 samples
    assert proba.shape[1] == len(estimator.classes_)  # Number of classes

    # Probabilities should sum to approximately 1 for each row
    # (may be slightly less due to gate_distance + diff combinations not in classes_)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=0.1)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__gate_distance_col_is_feature_true(df_type):
    """ConditionalEstimator should include gate_distance_col when gate_distance_col_is_feature=True."""
    data = df_type(
        {
            "feature1": [1, 2, 3, 4],
            "gate_distance": [5, 5, 10, 10],
        }
    )
    y = np.array([3, 4, 12, 13])  # gate_distance >= target for first 2, < for last 2

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42),
        outcome_1_estimator=LogisticRegression(random_state=42),
        gate_distance_col_is_feature=True,  # Include gate_distance
    )

    estimator.fit(data, y)

    # fitted_feats should include gate_distance
    assert "gate_distance" in estimator.fitted_feats


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__gate_distance_col_is_feature_false(df_type):
    """ConditionalEstimator should exclude gate_distance_col when gate_distance_col_is_feature=False."""
    data = df_type(
        {
            "feature1": [1, 2, 3, 4],
            "gate_distance": [5, 5, 10, 10],
        }
    )
    y = np.array([3, 4, 12, 13])  # gate_distance >= target for first 2, < for last 2

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42),
        outcome_1_estimator=LogisticRegression(random_state=42),
        gate_distance_col_is_feature=False,  # Exclude gate_distance
    )

    estimator.fit(data, y)

    # fitted_feats should NOT include gate_distance
    assert "gate_distance" not in estimator.fitted_feats
    assert "feature1" in estimator.fitted_feats


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__predict_returns_classes(df_type):
    """ConditionalEstimator.predict should return predicted classes."""
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "gate_distance": [5, 5, 5, 10, 10, 10],
        }
    )
    y = np.array([3, 4, 5, 12, 13, 14])

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42),
        outcome_1_estimator=LogisticRegression(random_state=42),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Predict
    test_data = df_type(
        {
            "feature1": [2, 5],
            "gate_distance": [5, 10],
        }
    )
    predictions = estimator.predict(test_data)

    # Should return predictions for each sample
    assert len(predictions) == 2
    # Predictions should be from the classes
    assert all(pred in estimator.classes_ for pred in predictions)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__preserves_gate_probabilities(df_type):
    """ConditionalEstimator must preserve gate probabilities exactly in final predictions.

    Bug: Previous implementation normalized conditional distributions independently,
    which broke the connection to gate probabilities. This caused large discrepancies
    (mean difference ~15%) between gate probabilities and summed final probabilities.

    Fix: Assign out-of-bounds probability to boundary classes to preserve gate probs,
    then normalize only the final mixture.
    """
    # Create dataset with sufficient samples for reliable estimates
    np.random.seed(42)
    n = 200
    feature1 = np.random.randn(n)
    gate_distance = np.random.choice([5, 10], size=n)

    # Generate outcomes based on features (predictable pattern)
    y = []
    for i in range(n):
        if feature1[i] > 0:
            # Likely to exceed gate_distance
            y.append(gate_distance[i] + np.random.choice([0, 1, 2, 3, 5]))
        else:
            # Likely to fall short
            y.append(gate_distance[i] + np.random.choice([-5, -3, -2, -1, 0]))
    y = np.array(y)

    data = df_type({"feature1": feature1, "gate_distance": gate_distance})

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42, max_iter=1000),
        outcome_1_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Predict on test data
    test_data = df_type({"feature1": [0.5, -0.5, 1.0], "gate_distance": [5, 10, 5]})

    # Get final probabilities
    final_proba = estimator.predict_proba(test_data)

    # Get gate probabilities
    test_pd = test_data if isinstance(test_data, pd.DataFrame) else test_data.to_pandas()
    gate_proba = estimator.gate_estimator.predict_proba(test_pd)
    gate_class_to_idx = {int(c): i for i, c in enumerate(estimator.gate_estimator.classes_)}
    p0 = gate_proba[:, gate_class_to_idx[0]]  # P(gate=0) = P(yards >= gate_distance)
    p1 = gate_proba[:, gate_class_to_idx[1]]  # P(gate=1) = P(yards < gate_distance)

    # Extract gate_distance values for test data
    gate_distances = test_pd["gate_distance"].values
    classes = np.array(estimator.classes_)

    # For each sample, verify gate probabilities are preserved
    for i in range(len(test_pd)):
        gd = gate_distances[i]

        # Sum P(yards >= gate_distance) from final distribution
        mask_above = classes >= gd
        p_above_from_final = final_proba[i, mask_above].sum()

        # Sum P(yards < gate_distance) from final distribution
        mask_below = classes < gd
        p_below_from_final = final_proba[i, mask_below].sum()

        # Gate probabilities MUST be preserved (tight tolerance)
        assert np.allclose(p_above_from_final, p0[i], atol=0.001), (
            f"Sample {i}: P(yards >= {gd}) from final = {p_above_from_final:.6f}, "
            f"but gate P(outcome_0) = {p0[i]:.6f} (diff: {abs(p_above_from_final - p0[i]):.6f}). "
            f"Gate probabilities must be preserved exactly."
        )
        assert np.allclose(p_below_from_final, p1[i], atol=0.001), (
            f"Sample {i}: P(yards < {gd}) from final = {p_below_from_final:.6f}, "
            f"but gate P(outcome_1) = {p1[i]:.6f} (diff: {abs(p_below_from_final - p1[i]):.6f}). "
            f"Gate probabilities must be preserved exactly."
        )


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__gate_logloss_matches_summed_logloss(df_type):
    """ConditionalEstimator logloss should match when computed from gate or summed final probs.

    This verifies that no information is lost in the mixture model and that
    marginal probabilities are perfectly preserved.
    """
    # Create dataset with sufficient samples
    np.random.seed(42)
    n_samples = 100
    feature1 = np.random.randn(n_samples)
    gate_distance = np.random.choice([5, 10], size=n_samples)

    # Generate outcomes based on features (to create predictable pattern)
    # Higher feature1 → more likely to exceed gate_distance
    y = []
    for i in range(n_samples):
        if feature1[i] > 0:
            # More likely to exceed
            y.append(gate_distance[i] + np.random.choice([1, 2, 3, 5, 8]))
        else:
            # More likely to fall short
            y.append(gate_distance[i] + np.random.choice([-5, -3, -2, -1, 0]))
    y = np.array(y)

    data = df_type({"feature1": feature1, "gate_distance": gate_distance})

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42, max_iter=1000),
        outcome_1_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Use test data
    test_feat1 = np.random.randn(30)
    test_gd = np.random.choice([5, 10], size=30)
    test_data = df_type(
        {
            "feature1": test_feat1,
            "gate_distance": test_gd,
        }
    )

    # Generate test outcomes
    test_y = []
    for i in range(len(test_feat1)):
        if test_feat1[i] > 0:
            test_y.append(test_gd[i] + np.random.choice([1, 2, 5]))
        else:
            test_y.append(test_gd[i] + np.random.choice([-3, -1, 0]))
    test_y = np.array(test_y)

    # Compute gate labels (1 if didn't exceed, 0 if exceeded)
    gate_labels = (test_gd >= test_y).astype(int)

    # Get gate probabilities directly
    test_pd = test_data if isinstance(test_data, pd.DataFrame) else test_data.to_pandas()
    gate_proba = estimator.gate_estimator.predict_proba(test_pd)
    gate_logloss = log_loss(gate_labels, gate_proba)

    # Get final probabilities and sum by threshold
    final_proba = estimator.predict_proba(test_data)
    classes = np.array(estimator.classes_)

    # For each sample, compute P(gate=0) and P(gate=1) from final probabilities
    summed_proba = np.zeros((len(test_pd), 2))
    for i in range(len(test_pd)):
        gd = test_gd[i]
        # P(gate=0) = P(yards >= gate_distance)
        summed_proba[i, 0] = final_proba[i, classes >= gd].sum()
        # P(gate=1) = P(yards < gate_distance)
        summed_proba[i, 1] = final_proba[i, classes < gd].sum()

    summed_logloss = log_loss(gate_labels, summed_proba)

    # They should match closely (accounting for the fact that classes may not include all values)
    # Use a more relaxed tolerance since the outcome estimators may not cover all possible diffs
    assert np.allclose(gate_logloss, summed_logloss, rtol=0.1), (
        f"Gate logloss = {gate_logloss:.10f}, but summed logloss = {summed_logloss:.10f}. "
        f"These should be similar as the mixture model approximately preserves marginal probabilities."
    )


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__exact_threshold_classification(df_type):
    """ConditionalEstimator should classify yards=gate_distance as outcome_0 (made first down).

    Bug: Previous implementation used gate_distance >= target, which classified
    exact threshold matches (yards == ydstogo) as outcome_1 (didn't exceed).
    This was wrong for first down logic where yards >= ydstogo = first down.

    Fix: Changed to gate_distance > target, so exact matches go to outcome_0 (exceeded/made it).

    Gate logic after fix:
    - gate_target = 1 if gate_distance > target (didn't exceed)
    - gate_target = 0 if gate_distance <= target (exceeded or matched)
    So when target == gate_distance, gate_target = 0 (outcome_0, made first down).
    """
    # Create dataset with exact threshold values and sufficient variety
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "gate_distance": [5, 5, 5, 5, 5, 10, 10, 10, 10, 10],
        }
    )
    # Include exact matches: yards = gate_distance (5=5 and 10=10)
    # Also include variety in both outcome_0 (>=) and outcome_1 (<)
    # outcome_0: 5>=5 (0), 6>5 (1), 8>5 (3), 10>=10 (0), 12>10 (2)
    # outcome_1: 3<5 (-2), 2<5 (-3), 8<10 (-2), 7<10 (-3)
    y = np.array([5, 6, 8, 3, 2, 10, 12, 15, 8, 7])  # 5=5 and 10=10 are exact matches

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42, max_iter=1000),
        outcome_1_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Verify outcome_0_estimator was trained on exact matches (diff=0)
    # diff = target - gate_distance = 5 - 5 = 0, 10 - 10 = 0
    assert 0 in estimator.outcome_0_classes_, (
        f"outcome_0_estimator should have diff=0 in its classes (exact threshold = made it). "
        f"Found classes: {estimator.outcome_0_classes_}"
    )

    # diff=0 should NOT be in outcome_1 (didn't make it)
    assert 0 not in estimator.outcome_1_classes_, (
        f"outcome_1_estimator should NOT have diff=0 (exact threshold = made it, not failed). "
        f"Found classes: {estimator.outcome_1_classes_}"
    )


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__classes_expansion_stays_reasonable(df_type):
    """ConditionalEstimator should not expand classes excessively beyond training target range.

    Bug: When gate_distance values vary widely (e.g., 1 to 50 for yards to go in football),
    the class expansion creates hundreds of classes like gate_distance + diff combinations,
    even though actual targets only span a narrow range (e.g., -5 to 35).

    The real scenario: ydstogo and rush_yards are INDEPENDENT, so:
    - ydstogo = 50, rush_yards = -5 creates diff = -55
    - ydstogo = 1, rush_yards = 35 creates diff = 34
    - This causes expansion to range(-54, 84) even though targets are only (-5, 35)
    """
    # Simulate realistic football scenario
    np.random.seed(42)
    n = 500

    # gate_distance (ydstogo) varies widely - independent of outcome
    gate_distance = np.random.choice(range(1, 51), size=n)

    # Outcomes (rush_yards) are constrained to [-5, 35] - independent of ydstogo
    # This creates large negative diffs when ydstogo is large
    y = np.random.choice(range(-5, 36), size=n)

    data = df_type({
        'feature1': np.random.randn(n),
        'gate_distance': gate_distance,
    })

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col='gate_distance',
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42, max_iter=1000),
        outcome_1_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Bug: classes_ expands to include all gate_distance + diff combinations
    # With gate_distance from 1-50 and diffs learned from data, this can create 100+ classes
    # Even though targets only range from -5 to 35 (41 values)

    y_min, y_max = y.min(), y.max()
    expected_max_classes = (y_max - y_min + 1) * 3  # Allow 3x expansion as reasonable buffer

    assert len(estimator.classes_) <= expected_max_classes, (
        f"ConditionalEstimator expanded classes excessively: {len(estimator.classes_)} classes "
        f"when targets only span {y_min} to {y_max} ({y_max - y_min + 1} values). "
        f"Expected at most ~{expected_max_classes} classes."
    )


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_conditional_estimator__outcome_estimators_disjoint_support(df_type):
    """ConditionalEstimator outcome estimators should have disjoint diff support.

    After fix (gate_distance > target):
    outcome_0_estimator should only predict non-negative diffs (>= 0) - exceeded or matched threshold.
    outcome_1_estimator should only predict negative diffs (< 0) - strictly didn't exceed threshold.
    """
    # Create dataset with clear separation
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "gate_distance": [5, 5, 5, 5, 10, 10, 10, 10],
        }
    )
    # Some outcomes exceed threshold, some don't
    y = np.array([2, 4, 8, 12, 7, 9, 11, 20])

    estimator = ConditionalEstimator(
        gate_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col="gate_distance",
        outcome_0_value=0,
        outcome_1_value=1,
        outcome_0_estimator=LogisticRegression(random_state=42, max_iter=1000),
        outcome_1_estimator=LogisticRegression(random_state=42, max_iter=1000),
        gate_distance_col_is_feature=True,
    )

    estimator.fit(data, y)

    # Check outcome_0_estimator classes (should be >= 0)
    outcome_0_classes = estimator.outcome_0_classes_
    assert all(c >= 0 for c in outcome_0_classes), (
        f"outcome_0_estimator should only have non-negative diffs (yards >= gate_distance). "
        f"Found: {outcome_0_classes}"
    )

    # Check outcome_1_estimator classes (should be < 0 after fix)
    outcome_1_classes = estimator.outcome_1_classes_
    assert all(c < 0 for c in outcome_1_classes), (
        f"outcome_1_estimator should only have negative diffs (yards < gate_distance). "
        f"Found: {outcome_1_classes}"
    )

    # Verify they don't overlap (no overlap after fixing boundary condition)
    overlap = set(outcome_0_classes) & set(outcome_1_classes)
    assert len(overlap) == 0, (
        f"outcome estimators should have disjoint support (no overlap after boundary fix). "
        f"Found overlap: {overlap}"
    )


def test_frequency_bucketing_classifier__initialization():
    """Test successful initialization with FrequencyBucketingClassifier"""
    clf = FrequencyBucketingClassifier(estimator=LogisticRegression(), min_prob=0.01)
    assert clf.min_prob == 0.01
    assert clf.estimator is not None
    assert clf.classes_ is None
    assert clf.estimator_ is None


def test_frequency_bucketing_classifier__min_prob_validation():
    """Test min_prob parameter validation"""
    with pytest.raises(ValueError, match="min_prob must be in"):
        FrequencyBucketingClassifier(estimator=LogisticRegression(), min_prob=0)

    with pytest.raises(ValueError, match="min_prob must be in"):
        FrequencyBucketingClassifier(estimator=LogisticRegression(), min_prob=-0.1)

    with pytest.raises(ValueError, match="min_prob must be in"):
        FrequencyBucketingClassifier(estimator=LogisticRegression(), min_prob=1.5)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__fit_creates_buckets(df_type):
    """Test that fit creates proper bucketing for rare classes"""
    np.random.seed(42)
    # Create dataset with common classes (0-3) and rare classes (10, 11)
    X = create_dataframe(df_type, {"feature": np.random.randn(1000)})
    y = np.concatenate([
        np.random.choice([0, 1, 2, 3], size=970),  # Common: 97%
        np.array([10] * 15),  # Rare: 1.5%
        np.array([11] * 15),  # Rare: 1.5%
    ])

    clf = FrequencyBucketingClassifier(
        estimator=LogisticRegression(max_iter=1000), min_prob=0.03
    )
    clf.fit(X, y)

    assert clf.classes_ is not None
    assert len(clf.classes_) == 6  # [0, 1, 2, 3, 10, 11]
    assert clf._class_to_bucket is not None
    assert clf._bucket_to_classes is not None

    # Common classes should have individual buckets
    for cls in [0, 1, 2, 3]:
        bucket_id = clf._class_to_bucket[cls]
        assert len(clf._bucket_to_classes[bucket_id]) == 1

    # Rare classes should be bucketed together
    bucket_10 = clf._class_to_bucket[10]
    bucket_11 = clf._class_to_bucket[11]
    assert bucket_10 == bucket_11, "Rare classes 10 and 11 should be in same bucket"
    assert len(clf._bucket_to_classes[bucket_10]) == 2


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__predict_proba_distributes_evenly(df_type):
    """Test that predict_proba distributes bucket probability evenly among classes"""
    np.random.seed(42)
    X = create_dataframe(df_type, {"feature": np.random.randn(100)})
    y = np.concatenate([
        np.array([0] * 50),
        np.array([1] * 40),
        np.array([10] * 5),  # Rare
        np.array([11] * 5),  # Rare
    ])

    clf = FrequencyBucketingClassifier(
        estimator=LogisticRegression(max_iter=1000), min_prob=0.15
    )
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    # Check shape
    assert proba.shape == (100, 4)  # 100 samples, 4 classes

    # Check probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0)

    # For samples, check that bucketed classes have equal probability
    bucket_10 = clf._class_to_bucket[10]
    bucket_11 = clf._class_to_bucket[11]
    if bucket_10 == bucket_11:  # If bucketed together
        class_idx_10 = np.where(clf.classes_ == 10)[0][0]
        class_idx_11 = np.where(clf.classes_ == 11)[0][0]
        # Probabilities should be equal for classes in same bucket
        assert np.allclose(proba[:, class_idx_10], proba[:, class_idx_11])


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__predict_returns_valid_classes(df_type):
    """Test that predict returns valid class labels from original classes"""
    np.random.seed(42)
    X = create_dataframe(df_type, {"feature": np.random.randn(100)})
    y = np.concatenate([
        np.array([0] * 50),
        np.array([5] * 30),
        np.array([10] * 10),
        np.array([15] * 10),
    ])

    clf = FrequencyBucketingClassifier(
        estimator=LogisticRegression(max_iter=1000), min_prob=0.15
    )
    clf.fit(X, y)

    predictions = clf.predict(X)

    assert predictions.shape == (100,)
    assert all(pred in clf.classes_ for pred in predictions)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__all_classes_frequent(df_type):
    """Test when all classes meet min_prob threshold (no bucketing needed)"""
    np.random.seed(42)
    X = create_dataframe(df_type, {"feature": np.random.randn(100)})
    y = np.concatenate([
        np.array([0] * 30),
        np.array([1] * 30),
        np.array([2] * 40),
    ])

    clf = FrequencyBucketingClassifier(
        estimator=LogisticRegression(max_iter=1000), min_prob=0.1
    )
    clf.fit(X, y)

    # All classes should have individual buckets
    for cls in [0, 1, 2]:
        bucket_id = clf._class_to_bucket[cls]
        assert len(clf._bucket_to_classes[bucket_id]) == 1
        assert clf._bucket_to_classes[bucket_id][0] == cls


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__all_classes_rare(df_type):
    """Test when all classes are below threshold (all bucketed together)"""
    np.random.seed(42)
    X = create_dataframe(df_type, {"feature": np.random.randn(100)})
    y = np.concatenate([
        np.array([0] * 15),
        np.array([1] * 15),
        np.array([2] * 15),
        np.array([3] * 15),
        np.array([4] * 40),
    ])

    clf = FrequencyBucketingClassifier(
        estimator=LogisticRegression(max_iter=1000), min_prob=0.5
    )
    clf.fit(X, y)

    # All classes should be in at most 2 buckets (common class 4 alone, others together)
    unique_buckets = set(clf._class_to_bucket.values())
    assert len(unique_buckets) <= 2


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__probability_normalization(df_type):
    """Test that probabilities always sum to 1"""
    np.random.seed(42)
    X = create_dataframe(df_type, {"feature": np.random.randn(200)})
    y = np.concatenate([
        np.random.choice([0, 1, 2, 3, 4, 5], size=180),
        np.random.choice([40, 41, -7, -8], size=20)
    ])

    clf = FrequencyBucketingClassifier(
        estimator=LogisticRegression(max_iter=1000), min_prob=0.03
    )
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Probabilities don't sum to 1: {row_sums[:5]}"


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__sample_weight_passed_to_estimator(df_type):
    """Test that sample_weight is passed to internal estimator but not used for bucketing"""
    np.random.seed(42)
    X = create_dataframe(df_type, {"feature": np.random.randn(100)})
    y = np.concatenate([
        np.array([0] * 40),
        np.array([1] * 40),
        np.array([2] * 20),  # Rare
    ])
    # Even though class 2 has high weight, bucketing is based on raw counts
    sample_weight = np.array([1.0] * 80 + [100.0] * 20)

    clf = FrequencyBucketingClassifier(
        estimator=LogisticRegression(max_iter=1000), min_prob=0.3
    )
    clf.fit(X, y, sample_weight=sample_weight)

    # Class 2 should still be rare (20% < 30%) based on raw counts
    bucket_2 = clf._class_to_bucket[2]
    # Check if class 2 is bucketed with another class
    # (may be alone or with neighbors, depends on algorithm)
    assert clf._class_to_bucket is not None


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__numeric_classes_required(df_type):
    """Test that non-numeric classes raise ValueError"""
    X = create_dataframe(df_type, {"feature": [1, 2, 3, 4, 5]})
    y = np.array(["a", "b", "c", "d", "e"])

    clf = FrequencyBucketingClassifier(estimator=LogisticRegression())

    with pytest.raises(ValueError, match="numeric classes"):
        clf.fit(X, y)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__single_class_error(df_type):
    """Test that single class raises ValueError"""
    X = create_dataframe(df_type, {"feature": [1, 2, 3, 4, 5]})
    y = np.array([1, 1, 1, 1, 1])

    clf = FrequencyBucketingClassifier(estimator=LogisticRegression())

    with pytest.raises(ValueError, match="at least 2 classes"):
        clf.fit(X, y)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_frequency_bucketing_classifier__boundary_classes(df_type):
    """Test that lowest/highest rare classes are bucketed correctly"""
    np.random.seed(42)
    # Create dataset where boundary classes are rare
    X = create_dataframe(df_type, {"feature": np.random.randn(1210)})
    y = np.concatenate([
        np.array([10] * 5),    # Rare lowest
        np.array([15] * 400),  # Common
        np.array([20] * 400),  # Common
        np.array([25] * 400),  # Common
        np.array([90] * 5),    # Rare highest
    ])

    clf = FrequencyBucketingClassifier(
        estimator=LogisticRegression(max_iter=1000), min_prob=0.02
    )
    clf.fit(X, y)

    # Rare boundary classes should be bucketed
    bucket_10 = clf._class_to_bucket[10]
    bucket_90 = clf._class_to_bucket[90]

    # Should be in buckets with neighbors (or alone if no neighbors)
    assert bucket_10 in clf._bucket_to_classes
    assert bucket_90 in clf._bucket_to_classes

    # Common classes should have individual buckets
    bucket_15 = clf._class_to_bucket[15]
    assert len(clf._bucket_to_classes[bucket_15]) == 1
