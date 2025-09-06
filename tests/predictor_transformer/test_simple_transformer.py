import pandas as pd
import pytest
from polars.testing import assert_frame_equal

from spforge.predictor_transformer import OperatorTransformer


@pytest.mark.parametrize("df", [pd.DataFrame, pd.DataFrame])
def test_operator_transformer_subtract(df):

    data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
        }
    )

    transformer = OperatorTransformer(
        feature1="feature_a",
        operation="subtract",
        feature2="feature_b",
    )
    transformed_data = transformer.transform(data)
    expected_data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
            "feature_a_minus_feature_b": [9, 18, 27],
        }
    )
    if isinstance(transformed_data, pd.DataFrame):
        pd.testing.assert_frame_equal(
            transformed_data, expected_data, check_dtype=False
        )
    else:
        expected_data = expected_data.select(transformed_data.columns)
        assert_frame_equal(transformed_data, expected_data, check_dtype=False)


@pytest.mark.parametrize("df", [pd.DataFrame, pd.DataFrame])
def test_operator_transformer_add(df):

    data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
        }
    )

    transformer = OperatorTransformer(
        feature1="feature_a",
        operation="add",
        feature2="feature_b",
    )
    transformed_data = transformer.transform(data)
    expected_data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
            transformer.alias: [11, 22, 33],
        }
    )
    if isinstance(transformed_data, pd.DataFrame):
        pd.testing.assert_frame_equal(
            transformed_data, expected_data, check_dtype=False
        )
    else:
        expected_data = expected_data.select(transformed_data.columns)
        assert_frame_equal(transformed_data, expected_data, check_dtype=False)


@pytest.mark.parametrize("df", [pd.DataFrame, pd.DataFrame])
def test_operator_transformer_divide(df):

    data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
        }
    )

    transformer = OperatorTransformer(
        feature1="feature_a",
        operation="divide",
        feature2="feature_b",
    )
    transformed_data = transformer.transform(data)
    expected_data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
            transformer.alias: [10.0, 10.0, 10.0],
        }
    )
    if isinstance(transformed_data, pd.DataFrame):
        pd.testing.assert_frame_equal(
            transformed_data, expected_data, check_dtype=False
        )
    else:
        expected_data = expected_data.select(transformed_data.columns)
        assert_frame_equal(transformed_data, expected_data, check_dtype=False)


@pytest.mark.parametrize("df", [pd.DataFrame, pd.DataFrame])
def test_operator_transformer_multiply(df):

    data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
        }
    )

    transformer = OperatorTransformer(
        feature1="feature_a",
        operation="multiply",
        feature2="feature_b",
    )
    transformed_data = transformer.transform(data)
    expected_data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
            transformer.alias: [10, 40, 90],
        }
    )
    if isinstance(transformed_data, pd.DataFrame):
        pd.testing.assert_frame_equal(
            transformed_data, expected_data, check_dtype=False
        )
    else:
        expected_data = expected_data.select(transformed_data.columns)
        assert_frame_equal(transformed_data, expected_data, check_dtype=False)
