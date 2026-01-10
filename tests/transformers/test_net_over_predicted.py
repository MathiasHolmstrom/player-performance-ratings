import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from sklearn.base import BaseEstimator, RegressorMixin

from spforge.transformers import NetOverPredictedTransformer


class ConstantPredRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, preds):
        self.preds = np.asarray(preds)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # ensure correct length if X changes shape
        return np.asarray(self.preds)[: len(X)]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_net_over_predicted_pandas_polars_and_lazy(backend: str):
    mock_estimator = ConstantPredRegressor([0.4, 0.8, 2.0, 3.0])

    base_data = {
        "feature1": [1, 2, 3, 4],
        "target": [0.5, 1.0, 2.0, 3.0],
    }

    transformer = NetOverPredictedTransformer(
        estimator=mock_estimator,
        features=["target_prediction", "net_over_predicted"],
        target_name="target",
        net_over_predicted_col="net_over_predicted",
        pred_column="target_prediction",
    )

    if backend == "pandas":
        X_fit = pd.DataFrame(base_data)
        y_fit = X_fit["target"]
    elif backend == "polars":
        X_fit = pl.DataFrame(base_data)
        y_fit = X_fit["target"]
    else:
        X_fit = pl.DataFrame(base_data).lazy()
        y_fit = pl.Series("target", base_data["target"])

    fit_out = transformer.fit_transform(X_fit, y=y_fit)

    if backend == "pandas":
        expected_fit = pd.DataFrame(base_data)
        expected_fit["target_prediction"] = [0.4, 0.8, 2.0, 3.0]
        expected_fit["net_over_predicted"] = [0.1, 0.2, 0.0, 0.0]
        pd.testing.assert_frame_equal(
            fit_out,
            expected_fit[fit_out.columns],
            check_dtype=False,
        )
    elif backend == "polars":
        expected_fit = pl.DataFrame(base_data).with_columns(
            pl.Series("target_prediction", [0.4, 0.8, 2.0, 3.0]),
            pl.Series("net_over_predicted", [0.1, 0.2, 0.0, 0.0]),
        )
        pl_assert_frame_equal(
            fit_out,
            expected_fit.select(fit_out.columns),
            check_dtype=False,
        )

    transform_data = {
        "feature1": [1, 2, 3, 4],
        "target": [0.5, 1.0, 2.0, 3.0],
    }

    if backend == "pandas":
        X_tr = pd.DataFrame(transform_data)
    elif backend == "polars":
        X_tr = pl.DataFrame(transform_data)
    else:
        X_tr = pl.DataFrame(transform_data).lazy()

    transformed_out = transformer.transform(X_tr)

    if backend == "pandas":
        expected_tr = pd.DataFrame(transform_data)
        expected_tr["target_prediction"] = [0.4, 0.8, 2.0, 3.0]
        expected_tr["net_over_predicted"] = [0.1, 0.2, 0.0, 0.0]
        pd.testing.assert_frame_equal(
            transformed_out,
            expected_tr[transformed_out.columns],
            check_dtype=False,
        )
    elif backend == "polars":
        expected_tr = pl.DataFrame(transform_data).with_columns(
            pl.Series("target_prediction", [0.4, 0.8, 2.0, 3.0]),
            pl.Series("net_over_predicted", [0.1, 0.2, 0.0, 0.0]),
        )
        pl_assert_frame_equal(
            transformed_out,
            expected_tr.select(transformed_out.columns),
            check_dtype=False,
        )
