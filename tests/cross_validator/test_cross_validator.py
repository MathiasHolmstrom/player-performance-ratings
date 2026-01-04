import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

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


def _make_cv(estimator, pred_col="pred", n_splits=3, features=None):
    features = features or ['x', 'y']
    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=estimator,
        prediction_column_name=pred_col,
        n_splits=n_splits,
        features=features,
    )
    return cv


def test_match_kfold_cv_binary_validation_only_subset(df_pd_cv_binary):
    cv = _make_cv(LinearRegression())
    out = cv.generate_validation_df(df_pd_cv_binary)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns

    assert 0 < len(out) < len(df_pd_cv_binary)

    assert pd.api.types.is_numeric_dtype(out["pred"])



def test_match_kfold_cv_regression_subset(df_pd_cv_reg):
    cv = _make_cv(LinearRegression())
    out = cv.generate_validation_df(df_pd_cv_reg)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert 0 < len(out) < len(df_pd_cv_reg)
    assert pd.api.types.is_numeric_dtype(out["pred"])


def test_match_kfold_cv_polars_input_returns_polars(df_pd_cv_binary):
    df_pl = pl.from_pandas(df_pd_cv_binary)
    cv = _make_cv(LogisticRegression(max_iter=1000))
    out = cv.generate_validation_df(df_pl)

    assert isinstance(out, pl.DataFrame)
    assert "pred" in out.columns
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
        target_column="y",
        estimator=LogisticRegression(),
        prediction_column_name="pred",
        min_validation_date="2024-01-01",
        n_splits=2,
        features=["x"],
    )

    with pytest.raises(ValueError):
        cv.generate_validation_df(df)


def test_match_kfold_cv_handles_missing_min_date(df_pd_cv_binary):
    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LogisticRegression(),
        prediction_column_name="pred",
        n_splits=2,
        min_validation_date=None,
        features=["x"],
    )

    out = cv.generate_validation_df(df_pd_cv_binary)
    assert len(out) > 0
    assert "pred" in out.columns


def test_match_kfold_cv_auto_infers_features(df_pd_cv_binary):
    """Test that features are auto-inferred when not provided."""
    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LinearRegression(),
        prediction_column_name="pred",
        n_splits=3,
        features=None,
    )

    out = cv.generate_validation_df(df_pd_cv_binary)
    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert 0 < len(out) < len(df_pd_cv_binary)
    assert pd.api.types.is_numeric_dtype(out["pred"])


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        Pipeline(estimator=LinearRegression(), feature_names=['x']),
    ],
)
def test_match_kfold_cv_auto_infers_features_regression(df_pd_cv_reg, estimator):
    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=estimator,
        prediction_column_name="pred",
        n_splits=3,
        features=None,
    )

    out = cv.generate_validation_df(df_pd_cv_reg)
    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert 0 < len(out) < len(df_pd_cv_reg)
    assert pd.api.types.is_numeric_dtype(out["pred"])
    assert not out["pred"].isna().all()
    assert out["pred"].min() < out["pred"].max()


def test_match_kfold_cv_regression_with_multiple_features():
    """Test regression estimator with multiple features and auto-inference."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    match_ids = [f"m{i:02d}" for i in range(20)]

    np.random.seed(42)
    x1 = np.arange(20, dtype=float)
    x2 = np.random.randn(20) * 2
    x3 = np.random.randn(20) * 0.5
    y = 2.0 * x1 + 1.5 * x2 - 0.5 * x3 + np.random.randn(20) * 0.1

    df = pd.DataFrame(
        {
            "date": dates,
            "gameid": match_ids,
            "feature1": x1,
            "feature2": x2,
            "feature3": x3,
            "y": y,
        }
    )

    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LinearRegression(),
        prediction_column_name="pred",
        n_splits=3,
        features=None,
    )

    out = cv.generate_validation_df(df)
    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert 0 < len(out) < len(df)
    assert pd.api.types.is_numeric_dtype(out["pred"])
    assert not out["pred"].isna().all()
    assert out["pred"].min() < out["pred"].max()

    cv_explicit = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LinearRegression(),
        prediction_column_name="pred",
        n_splits=3,
        features=["feature1", "feature2", "feature3"],
    )

    out_explicit = cv_explicit.generate_validation_df(df)
    assert len(out_explicit) == len(out)
    assert pd.api.types.is_numeric_dtype(out_explicit["pred"])


def test_match_kfold_cv_auto_infers_features_polars(df_pd_cv_binary):
    """Test auto-inference with Polars DataFrame."""
    df_pl = pl.from_pandas(df_pd_cv_binary)
    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LogisticRegression(max_iter=1000),
        prediction_column_name="pred",
        n_splits=3,
        features=None,
    )

    out = cv.generate_validation_df(df_pl)
    assert isinstance(out, pl.DataFrame)
    assert "pred" in out.columns
    assert 0 < out.height < df_pl.height


def test_match_kfold_cv_explicit_features_vs_auto_infer(df_pd_cv_binary):
    """Test that auto-inferred features work and produce reasonable results."""
    cv_explicit = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        prediction_column_name="pred",
        n_splits=3,
        features=["x"],
    )
    out_explicit = cv_explicit.generate_validation_df(df_pd_cv_binary)
    assert len(out_explicit) > 0
    assert "pred" in out_explicit.columns

    cv_auto_infer = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        prediction_column_name="pred",
        n_splits=3,
        features=None,
    )
    out_auto_infer = cv_auto_infer.generate_validation_df(df_pd_cv_binary)
    assert len(out_auto_infer) > 0
    assert "pred" in out_auto_infer.columns
    assert len(out_auto_infer) == len(out_explicit)

    cv_explicit_both = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        prediction_column_name="pred",
        n_splits=3,
        features=["x", "team"],
    )
    out_explicit_both = cv_explicit_both.generate_validation_df(df_pd_cv_binary)

    pd.testing.assert_frame_equal(
        out_auto_infer.sort_values(["date", "gameid", "team"]).reset_index(drop=True),
        out_explicit_both.sort_values(["date", "gameid", "team"]).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-5,
    )


def test_match_kfold_cv_auto_infer_excludes_internal_columns():
    """Test that auto-inference excludes internal columns like __match_num."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "gameid": [f"m{i:02d}" for i in range(6)],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0, 1, 0, 1, 0, 1],
            "__match_num": [0, 0, 1, 1, 2, 2],
        }
    )

    cv = MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=LogisticRegression(max_iter=1000),
        prediction_column_name="pred",
        n_splits=2,
        features=None,
    )

    out = cv.generate_validation_df(df)
    assert "pred" in out.columns
    assert len(out) > 0


@pytest.fixture
def df_pd_cv_multiclass():
    # 30 matches, 2 rows per match (team 0/1) => enough for folds
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    match_ids = [f"m{i:02d}" for i in range(30)]
    rows = []
    for i, (d, mid) in enumerate(zip(dates, match_ids)):
        for team in (0, 1):
            x = float(i) + (0.1 if team == 1 else 0.0)
            # 3-class target
            y = int(i % 3)
            rows.append({"date": d, "gameid": mid, "team": team, "x": x, "y": y})
    return pd.DataFrame(rows)


def _make_cv(estimator, pred_col="pred", n_splits=3, features=None):
    return MatchKFoldCrossValidator(
        match_id_column_name="gameid",
        date_column_name="date",
        target_column="y",
        estimator=estimator,
        prediction_column_name=pred_col,
        n_splits=n_splits,
        features=features,
    )


def _assert_multiclass_pred_column_is_vectorized(pred_series, n_classes: int):
    # pandas: object column of list/ndarray
    # polars: List dtype column
    if isinstance(pred_series, pd.Series):
        assert pred_series.dtype == "object"
        assert len(pred_series) > 0
        first = pred_series.iloc[0]
        assert isinstance(first, (list, tuple, np.ndarray))
        assert len(first) == n_classes
        # values should look like probabilities
        v = np.asarray(first, dtype=float)
        assert np.all(v >= 0.0) and np.all(v <= 1.0)
        assert np.isclose(v.sum(), 1.0, atol=1e-6)
    else:
        assert isinstance(pred_series.dtype, pl.List)

        first = pred_series[0]

        if isinstance(first, pl.Series):
            first = first.to_list()

        assert isinstance(first, (list, tuple))
        assert len(first) == n_classes

        v = np.asarray(first, dtype=float)
        assert np.all(v >= 0.0) and np.all(v <= 1.0)
        assert np.isclose(v.sum(), 1.0, atol=1e-6)


def test_match_kfold_cv_multiclass_predict_proba_pandas_creates_vector_column(df_pd_cv_multiclass):
    cv = _make_cv(
        LogisticRegression(max_iter=2000, multi_class="auto"),
        features=["x", "team"],
        n_splits=3,
    )
    out = cv.generate_validation_df(df_pd_cv_multiclass)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert 0 < len(out) < len(df_pd_cv_multiclass)

    _assert_multiclass_pred_column_is_vectorized(out["pred"], n_classes=3)


def test_match_kfold_cv_multiclass_predict_proba_polars_creates_vector_column(df_pd_cv_multiclass):
    df_pl = pl.from_pandas(df_pd_cv_multiclass)

    cv = _make_cv(
        LogisticRegression(max_iter=2000, multi_class="auto"),
        features=["x", "team"],
        n_splits=3,
    )
    out = cv.generate_validation_df(df_pl)

    assert isinstance(out, pl.DataFrame)
    assert "pred" in out.columns
    assert 0 < out.height < df_pl.height

    _assert_multiclass_pred_column_is_vectorized(out["pred"], n_classes=3)


class _PredictProbaRaisesAttributeError:
    # Has attribute, but calling it throws AttributeError
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        raise AttributeError("boom")

    def predict(self, X):
        Xn = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return np.zeros(Xn.shape[0], dtype=float) + 0.123


def test_predict_smart_falls_back_to_predict_on_attributeerror(df_pd_cv_multiclass):
    cv = _make_cv(
        _PredictProbaRaisesAttributeError(),
        features=["x", "team"],
        n_splits=3,
    )
    out = cv.generate_validation_df(df_pd_cv_multiclass)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    assert pd.api.types.is_numeric_dtype(out["pred"])
    assert np.isclose(out["pred"].iloc[0], 0.123)


def test_predict_smart_binary_predict_proba_is_vector(df_pd_cv_multiclass):
    df_bin = df_pd_cv_multiclass.copy()
    df_bin["y"] = (df_bin["y"] == 1).astype(int)

    cv = _make_cv(
        LogisticRegression(max_iter=2000),
        features=["x", "team"],
        n_splits=3,
    )
    out = cv.generate_validation_df(df_bin)

    assert isinstance(out, pd.DataFrame)
    assert "pred" in out.columns
    _assert_vector_pred_pandas(out["pred"], n_classes=2)


def _assert_vector_pred_pandas(pred: pd.Series, n_classes: int):
    assert pred.dtype == "object"
    first = pred.iloc[0]
    assert isinstance(first, (list, tuple, np.ndarray))
    assert len(first) == n_classes
    v = np.asarray(first, dtype=float)
    assert np.all(v >= 0.0) and np.all(v <= 1.0)
    assert np.isclose(v.sum(), 1.0, atol=1e-6)
