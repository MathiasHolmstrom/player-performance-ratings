import pandas as pd
from sklearn.linear_model import LinearRegression

from spforge.estimator import GranularityEstimator, SkLearnEnhancerEstimator
from spforge.transformers import EstimatorTransformer


def test_estimator_transformer_context__with_sklearn_enhancer():
    """EstimatorTransformer.context_features forwards from SkLearnEnhancerEstimator."""
    inner_estimator = SkLearnEnhancerEstimator(
        estimator=LinearRegression(), date_column="game_date", day_weight_epsilon=0.01
    )
    transformer = EstimatorTransformer(
        estimator=inner_estimator, prediction_column_name="pred", features=["feature1", "feature2"]
    )

    assert transformer.context_features == ["game_date"]


def test_estimator_transformer_context__nested_estimators():
    """EstimatorTransformer walks through nested estimators to find context."""
    # Nested: GranularityEstimator wrapping SkLearnEnhancerEstimator wrapping LinearRegression
    inner = SkLearnEnhancerEstimator(estimator=LinearRegression(), date_column="game_date")
    grouped = GranularityEstimator(estimator=inner, granularity_column_name="team_id")
    transformer = EstimatorTransformer(
        estimator=grouped, prediction_column_name="pred", features=["feature1"]
    )

    # Should walk through and find date_column from inner SkLearnEnhancerEstimator
    assert transformer.context_features == ["game_date"]


def test_estimator_transformer_context__plain_sklearn():
    """EstimatorTransformer with plain sklearn estimator returns empty context."""
    transformer = EstimatorTransformer(
        estimator=LinearRegression(),
        prediction_column_name="pred",
        features=["feature1", "feature2"],
    )

    # Plain sklearn estimator has no context_features property or date_column
    assert transformer.context_features == []
