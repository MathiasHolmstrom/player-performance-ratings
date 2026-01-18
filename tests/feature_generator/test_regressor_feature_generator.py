import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from spforge import AutoPipeline, ColumnNames
from spforge.feature_generator import RegressorFeatureGenerator


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="match_id",
        team_id="team_id",
        start_date="start_date",
    )


def _make_df(frame):
    data = {
        "match_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "team_id": [1, 2, 1, 2, 1, 2, 1, 2],
        "start_date": pd.to_datetime(
            [
                "2023-01-01",
                "2023-01-01",
                "2023-01-02",
                "2023-01-02",
                "2023-01-03",
                "2023-01-03",
                "2023-01-04",
                "2023-01-04",
            ]
        ),
        "x1": [1.0, 2.0, 1.5, 2.5, 3.0, 4.0, 3.5, 4.5],
        "x2": [0.5, 1.0, 0.7, 1.1, 0.9, 1.3, 1.0, 1.4],
        "y": [1.2, 2.1, 1.4, 2.4, 1.6, 2.6, 1.8, 2.8],
    }
    if frame == "pd":
        return pd.DataFrame(data)
    return pl.DataFrame(data)


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_regressor_feature_generator__adds_cv_predictions_and_future_predictions(
    frame, column_names
):
    df = _make_df(frame)
    pipeline = AutoPipeline(estimator=LinearRegression(), estimator_features=["x1", "x2"])
    transformer = RegressorFeatureGenerator(
        estimator=pipeline,
        target_name="y",
        prediction_column_name="pred",
        column_names=column_names,
        n_splits=2,
    )

    out = transformer.fit_transform(df, column_names=column_names)
    assert len(out) == len(df)
    assert "pred" in out.columns
    assert "is_validation" not in out.columns

    if frame == "pd":
        assert out["pred"].isna().sum() == 0
        future_df = df.drop(columns=["y"])
    else:
        assert out["pred"].null_count() == 0
        future_df = df.drop("y")

    future_out = transformer.future_transform(future_df)
    assert len(future_out) == len(df)
    assert "pred" in future_out.columns


def test_regressor_feature_generator__requires_regressor(column_names):
    df = _make_df("pd")
    pipeline = AutoPipeline(estimator=LogisticRegression(), estimator_features=["x1", "x2"])
    transformer = RegressorFeatureGenerator(
        estimator=pipeline,
        target_name="y",
        prediction_column_name="pred",
        column_names=column_names,
        n_splits=2,
    )

    with pytest.raises(ValueError, match="regressor"):
        transformer.fit_transform(df, column_names=column_names)
