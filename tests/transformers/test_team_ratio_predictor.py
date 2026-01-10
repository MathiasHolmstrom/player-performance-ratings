import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from spforge.transformers import RatioEstimatorTransformer


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_default_returns_only_ratio(df_factory):
    data = df_factory(
        {
            "performance": [0.2, 0.8, 0.5, 0.5],
            "target": [1.0, 0.0, 1.0, 0.0],
            "team_id": [1, 1, 2, 2],
            "game_id": [10, 10, 10, 10],
        }
    )

    transformer = RatioEstimatorTransformer(
        features=["performance"],
        estimator=LinearRegression(),
        granularity=["team_id", "game_id"],
        ratio_column_name="ratio",
    )

    y = data["target"] if isinstance(data, pd.DataFrame) else data.get_column("target")
    out = transformer.fit_transform(data, y)

    cols = out.columns if isinstance(out, pd.DataFrame) else out.columns
    assert list(cols) == ["ratio"]


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_returns_ratio_and_row_prediction(df_factory):
    data = df_factory(
        {
            "performance": [0.2, 0.8, 0.5, 0.5],
            "target": [1.0, 0.0, 1.0, 0.0],
            "team_id": [1, 1, 2, 2],
            "game_id": [10, 10, 10, 10],
        }
    )

    transformer = RatioEstimatorTransformer(
        features=["performance"],
        estimator=LinearRegression(),
        granularity=["team_id", "game_id"],
        ratio_column_name="ratio",
        prediction_column_name="row_pred",
    )

    y = data["target"] if isinstance(data, pd.DataFrame) else data.get_column("target")
    out = transformer.fit_transform(data, y)
    for e_c in ["row_pred", "ratio"]:
        assert e_c in out


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_returns_ratio_and_granularity_prediction(df_factory):
    data = df_factory(
        {
            "performance": [0.2, 0.8, 0.5, 0.5],
            "target": [1.0, 0.0, 1.0, 0.0],
            "team_id": [1, 1, 2, 2],
            "game_id": [10, 10, 10, 10],
        }
    )

    transformer = RatioEstimatorTransformer(
        features=["performance"],
        estimator=LinearRegression(),
        granularity=["team_id", "game_id"],
        ratio_column_name="ratio",
        granularity_prediction_column_name="team_pred",
    )

    y = data["target"] if isinstance(data, pd.DataFrame) else data.get_column("target")
    out = transformer.fit_transform(data, y)

    cols = out.columns if isinstance(out, pd.DataFrame) else out.columns
    for e_c in ["ratio", "team_pred"]:
        assert e_c in cols


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_returns_ratio_row_and_granularity_predictions(df_factory):
    data = df_factory(
        {
            "performance": [0.2, 0.8, 0.5, 0.5],
            "target": [1.0, 0.0, 1.0, 0.0],
            "team_id": [1, 1, 2, 2],
            "game_id": [10, 10, 10, 10],
        }
    )

    transformer = RatioEstimatorTransformer(
        features=["performance"],
        estimator=LinearRegression(),
        granularity=["team_id", "game_id"],
        ratio_column_name="ratio",
        prediction_column_name="row_pred",
        granularity_prediction_column_name="team_pred",
    )

    y = data["target"] if isinstance(data, pd.DataFrame) else data.get_column("target")
    out = transformer.fit_transform(data, y)

    cols = out.columns if isinstance(out, pd.DataFrame) else out.columns
    for expected_col in ["ratio", "row_pred", "team_pred"]:
        assert expected_col in cols


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_no_passthrough_columns_leak(df_factory):
    data = df_factory(
        {
            "performance": [0.1, 0.9],
            "target": [1.0, 0.0],
            "team_id": [1, 1],
            "game_id": [1, 1],
        }
    )

    transformer = RatioEstimatorTransformer(
        features=["performance"],
        estimator=LinearRegression(),
        granularity=["team_id", "game_id"],
        ratio_column_name="ratio",
    )

    y = data["target"] if isinstance(data, pd.DataFrame) else data.get_column("target")
    out = transformer.fit_transform(data, y)

    cols = set(out.columns if isinstance(out, pd.DataFrame) else out.columns)
    assert cols == {"ratio"}


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_predict_row_false_uses_existing_row_prediction_column(df_factory):
    data = df_factory(
        {
            "performance": [0.2, 0.8, 0.5, 0.5],
            "target": [1.0, 0.0, 1.0, 0.0],
            "team_id": [1, 1, 2, 2],
            "game_id": [10, 10, 10, 10],
            "row_pred": [2.0, 2.0, 4.0, 4.0],
        }
    )

    transformer = RatioEstimatorTransformer(
        features=["performance"],
        estimator=LinearRegression(),
        granularity=["team_id", "game_id"],
        ratio_column_name="ratio",
        prediction_column_name="row_pred",
        predict_row=False,
    )

    y = data["target"] if isinstance(data, pd.DataFrame) else data.get_column("target")
    out = transformer.fit_transform(data, y)

    assert list(out.columns) == ["ratio", "row_pred"]
    ratio = out["ratio"] if isinstance(out, pd.DataFrame) else out.get_column("ratio")
    ratio_values = ratio.to_list() if hasattr(ratio, "to_list") else ratio.tolist()
    assert all(v == 1.0 for v in ratio_values)


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_predict_row_false_uses_existing_row_prediction_column(df_factory):
    data = df_factory(
        {
            "performance": [0.2, 0.8, 0.5, 0.5],
            "target": [1.0, 0.0, 1.0, 0.0],
            "team_id": [1, 1, 2, 2],
            "game_id": [10, 10, 10, 10],
            "row_pred": [2.0, 2.0, 4.0, 4.0],
        }
    )

    transformer = RatioEstimatorTransformer(
        features=["performance"],
        estimator=LinearRegression(),
        granularity=["team_id", "game_id"],
        ratio_column_name="ratio",
        prediction_column_name="row_pred",
        granularity_prediction_column_name="team_pred",
        predict_row=False,
        predict_granularity=True,  # default, explicit for clarity
    )

    y = data["target"] if isinstance(data, pd.DataFrame) else data.get_column("target")
    out = transformer.fit_transform(data, y)
    for e_c in ["ratio", "row_pred", "team_pred"]:
        assert e_c in out

    if isinstance(out, pd.DataFrame):
        ratio_vals = out["ratio"].to_list()
        row_vals = out["row_pred"].to_list()
        team_vals = out["team_pred"].to_list()
        input_row_vals = data["row_pred"].to_list()
    else:
        ratio_vals = out.get_column("ratio").to_list()
        row_vals = out.get_column("row_pred").to_list()
        team_vals = out.get_column("team_pred").to_list()
        input_row_vals = data.get_column("row_pred").to_list()

    assert row_vals == input_row_vals

    expected = [r / t for r, t in zip(row_vals, team_vals, strict=False)]
    assert np.allclose(ratio_vals, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_predict_row_false_requires_prediction_column_name(df_factory):
    df_factory(
        {
            "performance": [0.2, 0.8],
            "target": [1.0, 0.0],
            "team_id": [1, 1],
            "game_id": [10, 10],
        }
    )

    with pytest.raises(ValueError, match="prediction_column_name must be provided"):
        RatioEstimatorTransformer(
            features=["performance"],
            estimator=LinearRegression(),
            granularity=["team_id", "game_id"],
            ratio_column_name="ratio",
            prediction_column_name=None,
            predict_row=False,
        )


@pytest.mark.parametrize("df_factory", [pd.DataFrame, pl.DataFrame])
def test_predict_granularity_false_requires_granularity_prediction_column_name(df_factory):
    df_factory(
        {
            "performance": [0.2, 0.8],
            "target": [1.0, 0.0],
            "team_id": [1, 1],
            "game_id": [10, 10],
        }
    )

    with pytest.raises(ValueError, match="granularity_prediction_column_name must be provided"):
        RatioEstimatorTransformer(
            features=["performance"],
            estimator=LinearRegression(),
            granularity=["team_id", "game_id"],
            ratio_column_name="ratio",
            granularity_prediction_column_name=None,
            predict_granularity=False,
        )
