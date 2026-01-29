import pickle

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from narwhals._native import IntoFrameT
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from spforge import AutoPipeline
from spforge.estimator import SkLearnEnhancerEstimator
from spforge.scorer import Filter, Operator
from spforge.transformers import EstimatorTransformer


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
            return self.predict(X)
        n = len(X)
        out = np.zeros((n, 2), dtype=float)
        out[:, 0] = 0.25
        out[:, 1] = 0.75
        return out


class CaptureDtypesEstimator(CaptureEstimator):
    def __init__(self, has_proba: bool = False):
        super().__init__(has_proba=has_proba)
        self.fit_dtypes = None

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        self.fit_dtypes = list(X.dtypes) if hasattr(X, "dtypes") else None
        return self


class FakeLGBMRegressor(BaseEstimator):
    __module__ = "lightgbm.sklearn"

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class EstimatorHoldingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        else:
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if getattr(self, "feature_names_in_", None) is not None:
                return np.asarray(self.feature_names_in_, dtype=object)
            return np.asarray([], dtype=object)
        return np.asarray(list(input_features), dtype=object)

    def set_output(self, *, transform=None):
        return self


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


def _inner_estimator(model: AutoPipeline):
    est = model.sklearn_pipeline.named_steps["est"]
    if hasattr(est, "_est") and est._est is not None:
        return est._est
    return est


def test_fit_predict_returns_ndarray(df_reg):
    model = AutoPipeline(
        estimator=LinearRegression(),
        estimator_features=["num1", "num2", "cat1"],
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

    model = AutoPipeline(
        estimator=LinearRegression(),
        estimator_features=["num1", "num2", "cat1"],
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
    model = AutoPipeline(
        estimator=LinearRegression(),
        estimator_features=["num1", "num2", "cat1"],
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
    model = AutoPipeline(
        estimator=LogisticRegression(max_iter=1000),
        estimator_features=["num1", "num2", "cat1"],
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


def test_filter_columns_not_passed_to_estimator(frame):
    df_pd = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "keep": [1, 0, 1, 0], "y": [1.0, 2.0, 3.0, 4.0]}
    )
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    model = AutoPipeline(
        estimator=CaptureEstimator(),
        estimator_features=["x"],
        filters=[Filter(column_name="keep", value=1, operator=Operator.EQUALS)],
    )

    X = _select(df, ["x", "keep"])
    y = _col(df, "y")
    model.fit(X, y=y)

    est = _inner_estimator(model)
    assert "keep" in model.required_features
    assert "keep" not in est.fit_columns


def test_predict_proba_raises_if_not_supported(df_reg):
    model = AutoPipeline(
        estimator=LinearRegression(),
        estimator_features=["num1", "num2", "cat1"],
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
        model = AutoPipeline(
            estimator=cap,
            estimator_features=["num1", "num2", "cat1"],
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
    model = AutoPipeline(
        estimator=cap,
        estimator_features=["num1", "num2", "cat1"],
        categorical_features=["cat1"],
        impute_missing_values=True,
        scale_features=True,
    )
    X = _select(df_reg, ["num1", "num2", "cat1"])
    y = _col(df_reg, "y")
    model.fit(X, y=y)

    assert cap.fit_columns is not None
    assert any(c.endswith("num1") and c.startswith("num") for c in cap.fit_columns)
    assert any(c.endswith("num2") and c.startswith("num") for c in cap.fit_columns)
    assert any(c.startswith("cat") for c in cap.fit_columns)


def test_infer_categorical_from_feature_names_when_only_numeric_features_given(df_reg):
    cap = CaptureEstimator()
    model = AutoPipeline(
        estimator=cap,
        estimator_features=["num1", "num2", "cat1"],
        numeric_features=["num1", "num2"],
        impute_missing_values=True,
        scale_features=True,
    )
    X = _select(df_reg, ["num1", "num2", "cat1"])
    y = _col(df_reg, "y")
    model.fit(X, y=y)

    assert cap.fit_columns is not None
    assert any(c.endswith("num1") and c.startswith("num") for c in cap.fit_columns)
    assert any(c.endswith("num2") and c.startswith("num") for c in cap.fit_columns)
    assert any(c.startswith("cat") for c in cap.fit_columns)


def test_granularity_groups_rows_before_estimator_fit_and_predict(frame):
    df_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "num2": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0],
            "cat1": ["a", "b", "a", None, "b", "c"],
            "y": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    model = AutoPipeline(
        estimator=CaptureEstimator(),
        estimator_features=["gameid", "num1", "num2", "cat1"],
        categorical_features=["cat1", "gameid"],
        granularity=["gameid"],
        impute_missing_values=True,
        scale_features=True,
        remainder="drop",
    )

    X = _select(df, ["gameid", "num1", "num2", "cat1"])
    y = _col(df, "y")
    model.fit(X, y=y)

    inner = _inner_estimator(model)

    if isinstance(df, pl.DataFrame):
        n_groups = df.select(pl.col("gameid").n_unique()).item()
    else:
        n_groups = df["gameid"].nunique()

    assert inner.fit_shape[0] == n_groups

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == len(X)


def test_pipeline_uses_feature_names_subset_even_if_extra_columns_present(df_reg):
    cap = CaptureEstimator()
    model = AutoPipeline(
        estimator=cap,
        estimator_features=["num1", "cat1"],
        categorical_features=["cat1"],
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
    assert any(c.startswith("num") for c in cap.fit_columns)
    assert any(c.startswith("cat") for c in cap.fit_columns)


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_categorical_handling_auto_uses_native_when_lightgbm_in_predictor_transformers(
    frame, df_reg_pd
):
    df = df_reg_pd if frame == "pd" else pl.from_pandas(df_reg_pd)

    cap = CaptureDtypesEstimator()
    model = AutoPipeline(
        estimator=cap,
        estimator_features=["num1", "num2", "cat1"],
        predictor_transformers=[EstimatorHoldingTransformer(estimator=FakeLGBMRegressor())],
        categorical_handling="auto",
        impute_missing_values=True,
        scale_features=False,
        remainder="drop",
    )

    X = _select(df, ["num1", "num2", "cat1"])
    y = _col(df, "y")
    model.fit(X, y=y)

    inner = _inner_estimator(model)
    assert inner.fit_columns is not None
    assert "cat1" in inner.fit_columns

    df_out = model.sklearn_pipeline.named_steps["pre"].transform(X)
    assert isinstance(df_out["cat1"].dtype, pd.CategoricalDtype)


class CaptureFitEstimator(BaseEstimator):
    def __init__(self):
        self.fit_columns = None
        self.fit_X_type = None
        self.fit_shape = None
        self.fit_sample_weight_is_none = None

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y, sample_weight=None):
        X = X.to_pandas()
        self.fit_X_type = type(X)
        self.fit_shape = getattr(X, "shape", None)
        self.fit_sample_weight_is_none = sample_weight is None
        self.fit_columns = list(X.columns) if hasattr(X, "columns") else None
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 0
        return np.zeros(n)


class AddConstantPredictionTransformer(BaseEstimator):
    def __init__(self, col_name: str):
        self.col_name = col_name
        self.features_out = [col_name]

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y=None):
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        return X.with_columns(
            nw.new_series(
                name=self.col_name,
                values=[1.23] * len(X),
                backend=nw.get_native_namespace(X),
            )
        )


def _find_fitted(obj, predicate):
    for _, v in obj.get_params(deep=True).items():
        if predicate(v):
            return v
    raise AssertionError("Did not find fitted object matching predicate")


def _fitted_estimator_transformer(model):
    sk = model.sklearn_pipeline
    ct = sk.named_steps["t1"]  # your second stage ColumnTransformer

    # ct.transformers_ contains the FITTED transformers
    for name, trans, _cols in ct.transformers_:
        if name != "features":
            continue

        # your 'features' transformer is _OnlyOutputColumns(...)
        only = trans

        # depending on your wrapper implementation, the wrapped transformer is usually here:
        if hasattr(only, "transformer"):
            return only.transformer
        if hasattr(only, "_transformer"):
            return only._transformer

        raise AssertionError("Found 'features' transformer but couldn't unwrap _OnlyOutputColumns")

    raise AssertionError("Could not find t1 'features' transformer in ct.transformers_")


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_final_sklearn_enhancer_estimator_gets_expected_feature_columns(frame):
    df_pd = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0],
            "num2": [10.0, 20.0, 30.0, 40.0],
            "location": ["home", "away", "home", "away"],
            "start_date": ["2022-10-18", "2022-10-18", "2022-10-19", "2022-10-20"],
            "player_id": [11, 11, 12, 12],
            "team_id": [1, 1, 2, 2],
            "match_id": [100, 100, 101, 101],
            "y": [1.2, 2.4, 2.0, 4.2],
        }
    )
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    inner = CaptureFitEstimator()
    enhancer = SkLearnEnhancerEstimator(
        estimator=inner,
        date_column="start_date",
        day_weight_epsilon=0.1,
    )

    dummy_prev = AddConstantPredictionTransformer(col_name="points_estimate_raw")

    final_transformer = EstimatorTransformer(
        features=["num1", "num2", "location", "start_date"],
        prediction_column_name="points_estimate",
        estimator=enhancer,
    )

    model = AutoPipeline(
        estimator=CaptureFitEstimator(),
        estimator_features=["num1", "num2", "location"],
        predictor_transformers=[dummy_prev, final_transformer],
        categorical_handling="auto",
        impute_missing_values=True,
        scale_features=False,
        remainder="drop",
    )

    if isinstance(df, pl.DataFrame):
        X = df.select(
            ["num1", "num2", "location", "player_id", "team_id", "match_id", "start_date"]
        )
        y = df["y"]
    else:
        X = df[["num1", "num2", "location", "player_id", "team_id", "match_id", "start_date"]]
        y = df["y"]

    model.fit(X, y=y)

    sk = model.sklearn_pipeline

    fitted_final = _find_fitted(
        sk,
        lambda o: isinstance(o, EstimatorTransformer)
        and getattr(o, "prediction_column_name", None) == "points_estimate",
    )

    expected_cols = ["num1", "num2", "location", "points_estimate_raw"] + list(
        fitted_final.get_feature_names_out()
    )

    assert model.estimator.fit_columns == expected_cols
    assert model.estimator.fit_X_type is pd.DataFrame
    et = _fitted_estimator_transformer(model)
    assert et.estimator_.estimator_.fit_columns == ["num1", "num2", "location"]


def test_autopipeline_is_picklable_after_fit():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="y")

    model = AutoPipeline(
        estimator=DummyRegressor(),
        estimator_features=["x"],
        categorical_handling="ordinal",
    )

    model.fit(df, y)

    pickle.dumps(model)


# --- Feature Importances Tests ---


def test_feature_importances__tree_model():
    from sklearn.ensemble import RandomForestRegressor

    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "cat1": ["a", "b", "a", "b", "a", "b"],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="y")

    model = AutoPipeline(
        estimator=RandomForestRegressor(n_estimators=5, random_state=42),
        estimator_features=["num1", "num2", "cat1"],
        categorical_handling="ordinal",
    )
    model.fit(df, y)

    importances = model.feature_importances_

    assert isinstance(importances, pd.DataFrame)
    assert list(importances.columns) == ["feature", "importance"]
    assert len(importances) == 3
    assert set(importances["feature"].tolist()) == {"num1", "num2", "cat1"}
    assert all(importances["importance"] >= 0)


def test_feature_importances__linear_model():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="y")

    model = AutoPipeline(
        estimator=LogisticRegression(max_iter=1000),
        estimator_features=["num1", "num2"],
        scale_features=True,
    )
    model.fit(df, y)

    importances = model.feature_importances_

    assert isinstance(importances, pd.DataFrame)
    assert list(importances.columns) == ["feature", "importance"]
    assert len(importances) == 2
    assert set(importances["feature"].tolist()) == {"num1", "num2"}
    assert all(importances["importance"] >= 0)


def test_feature_importances__not_fitted_raises():
    model = AutoPipeline(
        estimator=LinearRegression(),
        estimator_features=["x"],
    )

    with pytest.raises(RuntimeError, match="Pipeline not fitted"):
        _ = model.feature_importances_


def test_feature_importances__unsupported_estimator_raises():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="y")

    model = AutoPipeline(
        estimator=DummyRegressor(),
        estimator_features=["x"],
    )
    model.fit(df, y)

    with pytest.raises(RuntimeError, match="does not support feature importances"):
        _ = model.feature_importances_


def test_feature_importances__with_sklearn_enhancer():
    from sklearn.ensemble import RandomForestRegressor

    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "start_date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="y")

    inner = RandomForestRegressor(n_estimators=5, random_state=42)
    enhancer = SkLearnEnhancerEstimator(
        estimator=inner,
        date_column="start_date",
        day_weight_epsilon=0.1,
    )

    model = AutoPipeline(
        estimator=enhancer,
        estimator_features=["num1", "num2"],
    )
    model.fit(df, y)

    importances = model.feature_importances_

    assert isinstance(importances, pd.DataFrame)
    assert list(importances.columns) == ["feature", "importance"]
    assert len(importances) == 2
    assert set(importances["feature"].tolist()) == {"num1", "num2"}


def test_feature_importances__onehot_features():
    from sklearn.ensemble import RandomForestRegressor

    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat1": ["a", "b", "c", "a", "b", "c"],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="y")

    model = AutoPipeline(
        estimator=RandomForestRegressor(n_estimators=5, random_state=42),
        estimator_features=["num1", "cat1"],
        categorical_handling="onehot",
    )
    model.fit(df, y)

    importances = model.feature_importances_

    assert isinstance(importances, pd.DataFrame)
    assert list(importances.columns) == ["feature", "importance"]
    # Should have expanded features: num1 + cat1_a, cat1_b, cat1_c
    assert len(importances) == 4
    assert "num1" in importances["feature"].tolist()
    assert any("cat1_" in f for f in importances["feature"].tolist())


def test_feature_importance_names__granularity_uses_deep_feature_names():
    from sklearn.ensemble import RandomForestRegressor

    df = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2", "g2"],
            "num1": [1.0, 2.0, 3.0, 4.0],
            "num2": [10.0, 20.0, 30.0, 40.0],
            "y": [1.0, 1.0, 2.0, 2.0],
        }
    )
    y = df["y"]

    model = AutoPipeline(
        estimator=RandomForestRegressor(n_estimators=5, random_state=42),
        estimator_features=["gameid", "num1", "num2"],
        predictor_transformers=[AddConstantPredictionTransformer(col_name="const_pred")],
        granularity=["gameid"],
        categorical_features=["gameid"],
        categorical_handling="ordinal",
        remainder="drop",
    )
    model.fit(df, y)

    names = model.feature_importance_names

    inner = _inner_estimator(model)
    assert list(names.keys()) == list(inner.feature_names_in_)
    assert "gameid" not in names
    assert "const_pred" in names


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_granularity_with_aggregation_weight__features_weighted(frame):
    df_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2", "g2"],
            "num1": [10.0, 30.0, 20.0, 40.0],
            "weight": [0.25, 0.75, 0.5, 0.5],
            "y": [1.0, 1.0, 2.0, 2.0],
        }
    )
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    cap = CaptureEstimator()
    model = AutoPipeline(
        estimator=cap,
        estimator_features=["num1"],
        granularity=["gameid"],
        aggregation_weight="weight",
        remainder="drop",
    )

    X = _select(df, ["gameid", "num1", "weight"])
    y = _col(df, "y")
    model.fit(X, y=y)

    inner = _inner_estimator(model)
    assert inner.fit_shape[0] == 2

    preds = model.predict(X)
    assert preds.shape[0] == len(X)


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_granularity_aggregation_weight__weighted_mean_correct(frame):
    df_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "num1": [10.0, 30.0],
            "weight": [0.25, 0.75],
            "y": [1.0, 1.0],
        }
    )
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    from spforge.transformers._other_transformer import GroupByReducer

    reducer = GroupByReducer(granularity=["gameid"], aggregation_weight="weight")
    transformed = reducer.fit_transform(df)

    if frame == "pl":
        num1_val = transformed["num1"].to_list()[0]
    else:
        num1_val = transformed["num1"].iloc[0]

    expected = (10.0 * 0.25 + 30.0 * 0.75) / (0.25 + 0.75)
    assert abs(num1_val - expected) < 1e-6


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_reduce_y_raises_when_target_not_uniform_per_group(frame):
    df_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "num1": [10.0, 30.0],
        }
    )
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    from spforge.transformers._other_transformer import GroupByReducer

    reducer = GroupByReducer(granularity=["gameid"])

    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Target.*must be uniform"):
        reducer.reduce_y(df, y)


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_reduce_y_works_when_target_uniform_per_group(frame):
    df_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2", "g2"],
            "num1": [10.0, 30.0, 20.0, 40.0],
        }
    )
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    from spforge.transformers._other_transformer import GroupByReducer

    reducer = GroupByReducer(granularity=["gameid"])

    y = np.array([1.0, 1.0, 2.0, 2.0])
    y_out, _ = reducer.reduce_y(df, y)

    assert len(y_out) == 2
    assert set(y_out) == {1.0, 2.0}


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_aggregation_weight_sums_weight_column(frame):
    df_pd = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "num1": [10.0, 30.0],
            "weight": [0.25, 0.75],
            "y": [1.0, 1.0],
        }
    )
    df = df_pd if frame == "pd" else pl.from_pandas(df_pd)

    from spforge.transformers._other_transformer import GroupByReducer

    reducer = GroupByReducer(granularity=["gameid"], aggregation_weight="weight")
    transformed = reducer.fit_transform(df)

    if frame == "pl":
        weight_val = transformed["weight"].to_list()[0]
    else:
        weight_val = transformed["weight"].iloc[0]

    expected = 0.25 + 0.75
    assert abs(weight_val - expected) < 1e-6
