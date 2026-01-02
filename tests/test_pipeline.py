import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression

from spforge import Pipeline


@pytest.fixture
def df_pd_reg():
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "num2": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0],
            "cat1": ["a", "b", "a", None, "b", "c"],
            "y": [1.2, 2.4, 2.0, 4.2, 5.1, 6.3],
        }
    )


@pytest.fixture
def df_pd_clf():
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "num2": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0, 70.0, 80.0],
            "cat1": ["a", "b", "a", None, "b", "c", "a", "c"],
            "y": [0, 1, 0, 1, 1, 0, 0, 1],
        }
    )


def test_rejects_numpy_array_fit():
    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
    )
    X = np.array([[1.0, 2.0]])
    y = np.array([1.0])
    with pytest.raises(TypeError):
        model.fit(X, y=y)


def test_rejects_numpy_array_predict(df_pd_reg):
    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )
    model.fit(df_pd_reg, features=["num1", "num2", "cat1"])
    X = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(TypeError):
        model.predict(X)


def test_fit_predict_pandas_returns_ndarray(df_pd_reg):
    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )

    X = df_pd_reg[["num1", "num2", "cat1"]]
    y = df_pd_reg["y"]

    model.fit(X, y=y)
    preds = model.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(df_pd_reg),)


def test_fit_predict_polars_returns_df_with_pred_column(df_pd_reg):
    df_pl = pl.from_pandas(df_pd_reg)

    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )

    model.fit(df_pl, features=["num1", "num2", "cat1"])
    out = model.predict(df_pl)

    assert isinstance(out, pl.DataFrame)
    assert model.pred_column in out.columns
    assert out.height == df_pl.height


def test_narwhals_interface_df_contains_target(df_pd_reg):
    df_pl = pl.from_pandas(df_pd_reg)

    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
        drop_rows_where_target_is_nan=True,
    )

    out_fit = model.fit(df_pl, features=["num1", "num2", "cat1"])
    assert out_fit is model

    out = model.predict(df_pl)
    assert isinstance(out, pl.DataFrame)
    assert model.pred_column in out.columns


def test_drop_rows_where_target_is_nan(df_pd_reg):
    df_pd = df_pd_reg.copy()
    df_pd.loc[2, "y"] = np.nan
    df_pl = pl.from_pandas(df_pd)

    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        drop_rows_where_target_is_nan=True,
    )

    model.fit(df_pl, features=["num1", "num2", "cat1"])
    out = model.predict(df_pl)

    assert isinstance(out, pl.DataFrame)
    assert out.height == df_pl.height
    assert model.pred_column in out.columns


def test_min_max_target_clipping(df_pd_reg):
    df_pl = pl.from_pandas(df_pd_reg)

    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        min_target=2.0,
        max_target=5.0,
    )

    model.fit(df_pl, features=["num1", "num2", "cat1"])
    out = model.predict(df_pl)

    assert isinstance(out, pl.DataFrame)
    assert model.pred_column in out.columns
    assert out.height == df_pl.height


def test_predict_proba_pandas(df_pd_clf):
    model = Pipeline(
        estimator=LogisticRegression(max_iter=1000),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )

    X = df_pd_clf[["num1", "num2", "cat1"]]
    y = df_pd_clf["y"]

    model.fit(X, y=y)
    proba = model.predict_proba(X)

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(df_pd_clf), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_raises_if_not_supported(df_pd_reg):
    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )

    X = df_pd_reg[["num1", "num2", "cat1"]]
    y = df_pd_reg["y"]

    model.fit(X, y=y)
    with pytest.raises(AttributeError):
        model.predict_proba(X)


def test_return_features_false_only_input_plus_pred(df_pd_reg):
    df_pl = pl.from_pandas(df_pd_reg)
    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
    )
    model.fit(df_pl, features=["num1", "num2", "cat1"])
    out = model.predict(df_pl, return_features=False)

    assert isinstance(out, pl.DataFrame)
    assert set(out.columns) == set(df_pl.columns) | {model.pred_column}


def test_return_features_true_keeps_all_columns(df_pd_reg):
    df_pl = pl.from_pandas(df_pd_reg)
    model = Pipeline(
        estimator=LinearRegression(),
        target="y",
        features=["num1", "num2", "cat1"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=True,
        remainder="drop",
    )
    model.fit(df_pl, features=["num1", "num2", "cat1"])
    out = model.predict(df_pl, return_features=True)

    assert isinstance(out, pl.DataFrame)
    assert model.pred_column in out.columns
    assert out.height == df_pl.height
