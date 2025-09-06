import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spforge.predictor._operators_predictor import OperatorsPredictor
from spforge.predictor_transformer import OperatorTransformer


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_operators_predictors(df):
    data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
            "target": [5, 10, 15],
        }
    )
    operators_predictor = OperatorsPredictor(
        transformers=[
            OperatorTransformer(
                feature1="feature_a",
                operation="subtract",
                feature2="feature_b",
                alias="predicted",
            )
        ],
        pred_column="predicted",
        target="target",
    )
    operators_predictor.train(data)
    predicted = operators_predictor.predict(data)

    expected_data = df(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1, 2, 3],
            "target": [5, 10, 15],
            "predicted": [9, 18, 27],
        }
    )

    if isinstance(expected_data, pd.DataFrame):
        pd.testing.assert_frame_equal(expected_data, predicted, check_dtype=False)
    else:
        expected_data = expected_data.select(expected_data.columns)
        assert_frame_equal(expected_data, predicted, check_dtype=False)
