import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from spforge.transformers import EstimatorTransformer


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_estimator_transformer__basic_fit_transform(df_type):
    """EstimatorTransformer should fit estimator and add prediction column."""
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
        }
    )
    y = np.array([3, 6, 9, 12, 15])  # y = feature1 + feature2

    transformer = EstimatorTransformer(
        estimator=LinearRegression(),
        prediction_column_name="pred",
        features=["feature1", "feature2"],
    )

    # Fit
    transformer.fit(data, y)

    # Verify estimator was fitted
    assert transformer.estimator_ is not None
    assert hasattr(transformer.estimator_, "coef_")

    # Transform
    result = transformer.transform(data)

    # Should have prediction column
    assert "pred" in result.columns

    # Result should only contain prediction column (output column only)
    assert list(result.columns) == ["pred"]


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_estimator_transformer__features_none_uses_all_columns(df_type):
    """EstimatorTransformer with features=None should use all input columns."""
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "feature3": [1, 1, 1, 1, 1],
        }
    )
    y = np.array([3, 6, 9, 12, 15])

    transformer = EstimatorTransformer(
        estimator=LinearRegression(),
        prediction_column_name="pred",
        features=None,  # Use all columns
    )

    transformer.fit(data, y)

    # Should use all 3 features
    assert transformer.features_ == ["feature1", "feature2", "feature3"]


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_estimator_transformer__features_explicit_list(df_type):
    """EstimatorTransformer with explicit features should use only those columns."""
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "feature3": [100, 200, 300, 400, 500],  # Not used
        }
    )
    y = np.array([3, 6, 9, 12, 15])

    transformer = EstimatorTransformer(
        estimator=LinearRegression(),
        prediction_column_name="pred",
        features=["feature1", "feature2"],  # Only use these
    )

    transformer.fit(data, y)

    # Should only use specified features
    assert transformer.features_ == ["feature1", "feature2"]

    # Transform should work
    result = transformer.transform(data)
    assert "pred" in result.columns


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_estimator_transformer__get_feature_names_out(df_type):
    """EstimatorTransformer.get_feature_names_out should return prediction column name."""
    data = df_type(
        {
            "feature1": [1, 2, 3],
            "feature2": [2, 4, 6],
        }
    )
    y = np.array([3, 6, 9])

    transformer = EstimatorTransformer(
        estimator=LinearRegression(),
        prediction_column_name="my_prediction",
        features=["feature1", "feature2"],
    )

    transformer.fit(data, y)

    # Should return only the prediction column name
    feature_names = transformer.get_feature_names_out()
    assert feature_names == ["my_prediction"]


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_estimator_transformer__output_column_only(df_type):
    """EstimatorTransformer.transform should output only prediction column."""
    data = df_type(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [2, 4, 6, 8],
            "feature3": [1, 1, 1, 1],
        }
    )
    y = np.array([3, 6, 9, 12])

    transformer = EstimatorTransformer(
        estimator=LinearRegression(),
        prediction_column_name="pred",
        features=["feature1", "feature2"],
    )

    transformer.fit(data, y)
    result = transformer.transform(data)

    # Output should contain ONLY the prediction column
    assert list(result.columns) == ["pred"]
    # Original columns should not be in output
    assert "feature1" not in result.columns
    assert "feature2" not in result.columns
    assert "feature3" not in result.columns


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_estimator_transformer__transform_without_fit_raises_error(df_type):
    """EstimatorTransformer.transform should raise error if not fitted."""
    data = df_type(
        {
            "feature1": [1, 2, 3],
            "feature2": [2, 4, 6],
        }
    )

    transformer = EstimatorTransformer(
        estimator=LinearRegression(),
        prediction_column_name="pred",
        features=["feature1", "feature2"],
    )

    # Transform without fit should raise error
    with pytest.raises(RuntimeError, match="not fitted"):
        transformer.transform(data)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_estimator_transformer__classification(df_type):
    """EstimatorTransformer should work with classification estimators."""
    data = df_type(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [1, 1, 2, 2, 3, 3],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])  # Binary classification

    transformer = EstimatorTransformer(
        estimator=LogisticRegression(random_state=42),
        prediction_column_name="class_pred",
        features=["feature1", "feature2"],
    )

    transformer.fit(data, y)
    result = transformer.transform(data)

    # Should have prediction column
    assert "class_pred" in result.columns
    assert len(result) == 6

    # Predictions should be 0 or 1
    if isinstance(result, pd.DataFrame):
        predictions = result["class_pred"].values
    else:
        predictions = result["class_pred"].to_numpy()

    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_estimator_transformer__narwhalify_compatibility(df_type):
    """EstimatorTransformer should handle both pandas and polars DataFrames."""
    data = df_type(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
        }
    )
    y = np.array([3, 6, 9, 12, 15])

    transformer = EstimatorTransformer(
        estimator=LinearRegression(),
        prediction_column_name="pred",
        features=["x1", "x2"],
    )

    transformer.fit(data, y)
    result = transformer.transform(data)

    # Result should be same type as input
    assert type(result).__name__ == type(data).__name__

    # Should have correct shape
    assert len(result) == 5
    assert "pred" in result.columns
