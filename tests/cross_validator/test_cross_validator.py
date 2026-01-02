import numpy as np
import pandas as pd
import polars as pl
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from spforge import Pipeline
from spforge.cross_validator import MatchKFoldCrossValidator


@pytest.fixture
def df_pd_cv_binary():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    match_ids = [f"m{i:02d}" for i in range(12)]

    rows = []
    for i, (d, mid) in enumerate(zip(dates, match_ids)):
        for team in (0, 1):
            x = float(i) + (0.1 if team == 1 else 0.0)
            y = int(i % 2 == 1)
            rows.append({"date": d, "gameid": mid, "team": team, "x": x, "y": y})
    return pd.DataFrame(rows)


@pytest.fixture
def df_pd_cv_reg():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    match_ids = [f"m{i:02d}" for i in range(12)]
    x = np.arange(12, dtype=float)
    y = 2.0 * x + 1.0
    return pd.DataFrame({"date": dates, "gameid": match_ids, "x": x, "y": y})


def _make_cv(estimator, pred_col="pred", n_splits=3):
    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        estimator=estimator,
        prediction_column_name=pred_col,
        n_splits=n_splits,
    )
    cv.features = ["x"]
    cv.target = "y"
    return cv


def test_match_kfold_cv_binary_validation_only_subset(df_pd_cv_binary):
    cv = _make_cv(LogisticRegression(max_iter=1000))

    out = cv.generate_validation_df(df_pd_cv_binary, add_train_prediction=False)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert "is_validation" in out.columns

    # default: only validation rows are returned
    assert 0 < len(out) < len(df_pd_cv_binary)

    # should be all validation rows
    assert set(out["is_validation"].unique()) == {1}

    # binary proba in [0, 1]
    assert pd.api.types.is_numeric_dtype(out["pred"])
    assert (out["pred"] >= 0).all()
    assert (out["pred"] <= 1).all()

    # ensure it's a subset of original rows (by keys)
    merged = out.merge(
        df_pd_cv_binary[["date", "gameid", "team", "x", "y"]],
        on=["date", "gameid", "team", "x", "y"],
        how="inner",
    )
    assert len(merged) == len(out)


def test_match_kfold_cv_binary_with_train_predictions_full_length(df_pd_cv_binary):
    cv = _make_cv(LogisticRegression(max_iter=1000))

    out = cv.generate_validation_df(df_pd_cv_binary, add_train_prediction=True)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert "is_validation" in out.columns

    # now includes train + val => should match input length (deduped by row index)
    assert len(out) == len(df_pd_cv_binary)

    # should contain both train and validation flags
    assert set(out["is_validation"].unique()) == {0, 1}

    # binary proba in [0, 1]
    assert (out["pred"] >= 0).all()
    assert (out["pred"] <= 1).all()


def test_match_kfold_cv_regression_validation_only_subset(df_pd_cv_reg):
    cv = _make_cv(LinearRegression())

    out = cv.generate_validation_df(df_pd_cv_reg, add_train_prediction=False)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert "is_validation" in out.columns

    assert 0 < len(out) < len(df_pd_cv_reg)
    assert set(out["is_validation"].unique()) == {1}
    assert pd.api.types.is_numeric_dtype(out["pred"])


def test_match_kfold_cv_polars_input_returns_polars(df_pd_cv_binary):
    df_pl = pl.from_pandas(df_pd_cv_binary)

    cv = _make_cv(LogisticRegression(max_iter=1000))

    out = cv.generate_validation_df(df_pl, add_train_prediction=False)

    assert isinstance(out, pl.DataFrame)
    assert "pred" in out.columns
    assert "is_validation" in out.columns
    assert 0 < out.height < df_pl.height


def test_match_kfold_cv_raises_when_train_empty():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "gameid": ["m0", "m1", "m2"],
            "x": [0.0, 1.0, 2.0],
            "y": [0, 1, 0],
        }
    )

    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        estimator=LogisticRegression(max_iter=1000),
        prediction_column_name="pred",
        min_validation_date="2024-01-01",
        n_splits=3,
    )
    cv.features = ["x"]
    cv.target = "y"

    with pytest.raises(ValueError):
        cv.generate_validation_df(df)


@pytest.fixture
def df_pd_cv_reg_rows():
    # 12 matches, 2 rows per match => 24 rows
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    match_ids = [f"m{i:02d}" for i in range(12)]

    rows = []
    for i, (d, mid) in enumerate(zip(dates, match_ids)):
        for team in (0, 1):
            x = float(i) + (0.1 if team == 1 else 0.0)
            y = 2.0 * x + 1.0
            rows.append({"date": d, "gameid": mid, "team": team, "x": x, "y": y})
    return pd.DataFrame(rows)


def test_match_kfold_cv_regressor_falls_back_to_predict_when_predict_proba_raises(df_pd_cv_reg_rows):
    # Pipeline has predict_proba method, but for LGBMRegressor it raises AttributeError internally
    est = Pipeline(
        estimator=LGBMRegressor(verbose=-100),
        target="y",
        features=["x"],
        pred_column="y_pred",
        impute_missing_values=True,
        one_hot_encode_cat_features=False,
        scale_features=False,
    )

    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        estimator=est,
        prediction_column_name="pred",
        n_splits=3,
    )
    cv.features = ["x"]
    cv.target = "y"

    out = cv.generate_validation_df(df_pd_cv_reg_rows, add_train_prediction=False)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert "is_validation" in out.columns

    # default: only validation rows
    assert 0 < len(out) < len(df_pd_cv_reg_rows)
    assert set(out["is_validation"].unique()) == {1}

    # should be numeric predictions (not probs)
    assert pd.api.types.is_numeric_dtype(out["pred"])