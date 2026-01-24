import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from spforge import EstimatorHyperparameterTuner, ParamSpec
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.estimator import SkLearnEnhancerEstimator
from spforge.scorer import MeanBiasScorer


class FakeLGBMClassifier(BaseEstimator):
    __module__ = "lightgbm.sklearn"

    def __init__(
        self,
        n_estimators: int = 100,
        num_leaves: int = 31,
        max_depth: int = 5,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
    ):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        n = len(X)
        if len(self.classes_) < 2:
            return np.ones((n, 1))
        return np.tile([0.4, 0.6], (n, 1))

    def predict(self, X):
        n = len(X)
        if len(self.classes_) == 1:
            return np.full(n, self.classes_[0])
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array(self.classes_)[idx]


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "mid": f"M{i // 2}",
                "date": date,
                "x1": float(i),
                "y": 1 if i % 2 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def scorer():
    return MeanBiasScorer(
        pred_column="y_pred",
        target="y",
        validation_column="is_validation",
    )


def test_estimator_tuner_requires_search_space(sample_df, scorer):
    estimator = LogisticRegression()

    cv = MatchKFoldCrossValidator(
        match_id_column_name="mid",
        date_column_name="date",
        target_column="y",
        estimator=estimator,
        prediction_column_name="y_pred",
        n_splits=2,
        features=["x1"],
    )

    tuner = EstimatorHyperparameterTuner(
        estimator=estimator,
        cross_validator=cv,
        scorer=scorer,
        direction="minimize",
        n_trials=2,
        show_progress_bar=False,
    )

    with pytest.raises(ValueError, match="param_search_space is required"):
        tuner.optimize(sample_df)


def test_estimator_tuner_custom_search_space(sample_df, scorer):
    estimator = SkLearnEnhancerEstimator(estimator=LogisticRegression())

    cv = MatchKFoldCrossValidator(
        match_id_column_name="mid",
        date_column_name="date",
        target_column="y",
        estimator=estimator,
        prediction_column_name="y_pred",
        n_splits=2,
        features=["x1"],
    )

    tuner = EstimatorHyperparameterTuner(
        estimator=estimator,
        cross_validator=cv,
        scorer=scorer,
        direction="minimize",
        param_search_space={
            "C": ParamSpec(
                param_type="float",
                low=0.1,
                high=2.0,
                log=True,
            )
        },
        n_trials=2,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_df)

    assert "estimator__C" in result.best_params
    assert isinstance(result.best_value, float)


def test_estimator_tuner_lgbm_defaults(sample_df, scorer):
    estimator = FakeLGBMClassifier()

    cv = MatchKFoldCrossValidator(
        match_id_column_name="mid",
        date_column_name="date",
        target_column="y",
        estimator=estimator,
        prediction_column_name="y_pred",
        n_splits=2,
        features=["x1"],
    )

    tuner = EstimatorHyperparameterTuner(
        estimator=estimator,
        cross_validator=cv,
        scorer=scorer,
        direction="minimize",
        n_trials=2,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_df)

    assert "n_estimators" in result.best_params
    assert isinstance(result.best_value, float)
