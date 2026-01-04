import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from spforge import Pipeline


class CaptureEstimator(BaseEstimator):
    def __init__(self, has_proba: bool = False):
        self.has_proba = has_proba
        self.fit_X_type = None
        self.fit_columns = None
        self.fit_shape = None
        self.predict_X_type = None
        self.predict_columns = None
        self.predict_shape = None
        self.classes_ = np.array([0, 1]) if has_proba else None

    def fit(self, X, y, sample_weight=None):
        self.fit_X_type = type(X)
        self.fit_shape = getattr(X, "shape", None)
        self.fit_columns = list(X.columns) if hasattr(X, "columns") else None
        self._y_is_numeric = pd.api.types.is_numeric_dtype(pd.Series(y))
        return self

    def predict(self, X):
        self.predict_X_type = type(X)
        self.predict_shape = getattr(X, "shape", None)
        self.predict_columns = list(X.columns) if hasattr(X, "columns") else None
        n = len(X)
        return np.zeros(n, dtype=float)

    def predict_proba(self, X):
        if not self.has_proba:
            raise AttributeError("predict_proba not supported")
        n = len(X)
        out = np.zeros((n, 2), dtype=float)
        out[:, 0] = 0.25
        out[:, 1] = 0.75
        return out


@pytest.fixture(params=["pd", "pl"])
def frame(request) -> str:
    return request.param


@pytest.fixture
def df_reg_pd():
    return pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "num2": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0],
            "cat1": ["a", "b", "a", None, "b", "c"],
            "y": [1.2, 2.4, 2.0, 4.2, 5.1, 6.3],
        }
    )


@pytest.fixture
def df_clf_pd():
    return pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "num2": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0, 70.0, 80.0],
            "cat1": ["a", "b", "a", None, "b", "c", "a", "c"],
            "y": [0, 1, 0, 1, 1, 0, 0, 1],
        }
    )


@pytest.fixture
def df_reg(frame, df_reg_pd):
    if frame == "pd":
        return df_reg_pd
    return pl.from_pandas(df_reg_pd)


@pytest.fixture
def df_clf(frame, df_clf_pd):
    if frame == "pd":
        return df_clf_pd
    return pl.from_pandas(df_clf_pd)


def _height(df) -> int:
    return df.height if isinstance(df, pl.DataFrame) else len(df)


def _select(df, cols: list[str]):
    return df.select(cols) if isinstance(df, pl.DataFrame) else df[cols]


def _col(df, name: str):
    return df[name]


def _inner_estimator(model: Pipeline):
    est = model._sk.named_steps["est"]
    if hasattr(est, "_est") and est._est is not None:
        return est._est
    raise AssertionError("Inner estimator not available; pipeline not fitted?")


def test_fit_predict_returns_ndarray(df_reg):
    model = Pipeline(
        estimator=LinearRegression(),
        feature_names=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )
    X = _select(df_reg, ["num1", "num2", "cat1"])
    y = _col(df_reg, "y")
    model.fit(X, y=y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (_height(df_reg),)


def test_drop_rows_where_target_is_nan(df_reg_pd, frame):
    df_pd = df_reg_pd.copy()
    df_pd.loc[2, "y"] = np.nan
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    model = Pipeline(
        estimator=LinearRegression(),
        feature_names=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        drop_rows_where_target_is_nan=True,
    )

    X = _select(df, ["num1", "num2", "cat1"])
    y = _col(df, "y")
    model.fit(X, y=y)
    preds = model.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (_height(df),)


def test_min_max_target_clipping(df_reg):
    model = Pipeline(
        estimator=LinearRegression(),
        feature_names=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        min_target=2.0,
        max_target=5.0,
    )

    X = _select(df_reg, ["num1", "num2", "cat1"])
    y = _col(df_reg, "y")
    model.fit(X, y=y)
    preds = model.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (_height(df_reg),)


def test_predict_proba(df_clf):
    model = Pipeline(
        estimator=LogisticRegression(max_iter=1000),
        feature_names=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )
    X = _select(df_clf, ["num1", "num2", "cat1"])
    y = _col(df_clf, "y")
    model.fit(X, y=y)
    proba = model.predict_proba(X)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (_height(df_clf), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_raises_if_not_supported(df_reg):
    model = Pipeline(
        estimator=LinearRegression(),
        feature_names=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )
    X = _select(df_reg, ["num1", "num2", "cat1"])
    y = _col(df_reg, "y")
    model.fit(X, y=y)
    with pytest.raises(AttributeError):
        model.predict_proba(X)

    def test_estimator_receives_original_input_frame_and_feature_subset(df_pd_reg):
        cap = CaptureEstimator()
        model = Pipeline(
            estimator=cap,
            feature_names=["num1", "num2", "cat1"],
            one_hot_encode_cat_features=True,
            impute_missing_values=True,
            scale_features=True,
        )
        X = df_pd_reg[["num1", "num2", "cat1"]]
        y = df_pd_reg["y"]
        model.fit(X, y=y)

        assert cap.fit_X_type is type(X)
        assert cap.fit_columns == ["num1", "num2", "cat1"]
        assert cap.fit_shape[0] == len(df_pd_reg)

        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(df_pd_reg),)

        assert cap.predict_X_type is type(X)
        assert cap.predict_columns == cap.fit_columns


def test_infer_numeric_from_feature_names_when_only_cat_features_given(df_reg):
    cap = CaptureEstimator()
    model = Pipeline(
        estimator=cap,
        feature_names=["num1", "num2", "cat1"],
        categorical_features=["cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )
    X = _select(df_reg, ["num1", "num2", "cat1"])
    y = _col(df_reg, "y")
    model.fit(X, y=y)

    assert cap.fit_columns is not None
    assert any(c.endswith("num1") and c.startswith("num__") for c in cap.fit_columns)
    assert any(c.endswith("num2") and c.startswith("num__") for c in cap.fit_columns)
    assert any(c.startswith("cat__") for c in cap.fit_columns)


def test_infer_categorical_from_feature_names_when_only_numeric_features_given(df_reg):
    cap = CaptureEstimator()
    model = Pipeline(
        estimator=cap,
        feature_names=["num1", "num2", "cat1"],
        numeric_features=["num1", "num2"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )
    X = _select(df_reg, ["num1", "num2", "cat1"])
    y = _col(df_reg, "y")
    model.fit(X, y=y)

    assert cap.fit_columns is not None
    assert any(c.endswith("num1") and c.startswith("num__") for c in cap.fit_columns)
    assert any(c.endswith("num2") and c.startswith("num__") for c in cap.fit_columns)
    assert any(c.startswith("cat__") for c in cap.fit_columns)


def test_granularity_groups_rows_before_estimator_fit_and_predict(df_reg):
    model = Pipeline(
        estimator=CaptureEstimator(),
        feature_names=["gameid", "num1", "num2", "cat1"],
        categorical_features=["cat1", "gameid"],
        granularity=["gameid"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
        remainder="drop",
    )

    X = _select(df_reg, ["gameid", "num1", "num2", "cat1"])
    y = _col(df_reg, "y")
    model.fit(X, y=y)

    inner = _inner_estimator(model)

    if isinstance(df_reg, pl.DataFrame):
        n_groups = df_reg.select(pl.col("gameid").n_unique()).item()
    else:
        n_groups = df_reg["gameid"].nunique()

    assert inner.fit_shape[0] == n_groups

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (n_groups,)


def test_pipeline_uses_feature_names_subset_even_if_extra_columns_present(df_reg):
    cap = CaptureEstimator()
    model = Pipeline(
        estimator=cap,
        feature_names=["num1", "cat1"],
        one_hot_encode_cat_features=True,
        categorical_features=['cat1'],
        impute_missing_values=True,
        scale_features=True,
        remainder="drop",
    )

    if isinstance(df_reg, pl.DataFrame):
        df = df_reg.with_columns(pl.arange(0, df_reg.height).alias("junk"))
        X = df.select(["num1", "cat1", "junk"])
        y = df["y"]
    else:
        df = df_reg.copy()
        df["junk"] = np.arange(len(df))
        X = df[["num1", "cat1", "junk"]]
        y = df["y"]

    model.fit(X, y=y)

    assert cap.fit_columns is not None
    assert all("junk" not in c for c in cap.fit_columns)
    assert any(c.startswith("num__") for c in cap.fit_columns)
    assert any(c.startswith("cat__") for c in cap.fit_columns)
