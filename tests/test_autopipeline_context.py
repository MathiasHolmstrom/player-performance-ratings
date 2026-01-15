import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spforge.autopipeline import AutoPipeline
from spforge.estimator import SkLearnEnhancerEstimator


def test_autopipeline_context__sklearn_enhancer_with_property():
    """AutoPipeline detects context_features from SkLearnEnhancerEstimator."""
    estimator = SkLearnEnhancerEstimator(
        estimator=LinearRegression(), date_column="game_date", day_weight_epsilon=0.01
    )

    pipeline = AutoPipeline(
        estimator=estimator, estimator_features=["feature1", "feature2"], granularity=[]
    )

    # Should detect game_date from estimator.context_features
    assert "game_date" in pipeline.context_feature_names
    assert "game_date" in pipeline.required_features


def test_autopipeline_context__plain_sklearn_no_context():
    """AutoPipeline with plain sklearn estimator has no estimator context."""
    pipeline = AutoPipeline(
        estimator=LinearRegression(), estimator_features=["feature1", "feature2"], granularity=[]
    )

    # Plain sklearn estimator has no context_features
    # context_feature_names should be empty (no estimator context, no granularity, no filters)
    assert pipeline.context_feature_names == []
