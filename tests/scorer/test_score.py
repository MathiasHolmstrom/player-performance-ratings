import math

import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import log_loss, mean_absolute_error

from spforge.scorer.score import (
    SklearnScorer,
    MeanBiasScorer,
)

from spforge.scorer import OrdinalLossScorer


def test_ordinal_loss_scorer_multiclass_rename_class_column_name():
    data = pl.DataFrame(
        {
            "predictions": [
                {"0": 0.1, "1": 0.6, "2": 0.3},
                {"0": 0.5, "1": 0.3, "2": 0.2},
                {"0": 0.2, "1": 0.3, "2": 0.5},
            ],
            "__target": [1, 0, 2],
            "total_points_classes": [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        }
    )

    score = OrdinalLossScorer(
        pred_column="predictions",
        target="__target",
    ).score(data)
    assert score > 0
    assert score < 0.693


def test_ordinal_loss_scorer_multiclass():
    data = pd.DataFrame(
        {
            "predictions": [
                {"0": 0.1, "1": 0.6, "2": 0.3},
                {"0": 0.5, "1": 0.3, "2": 0.2},
                {"0": 0.2, "1": 0.3, "2": 0.5},
            ],
            "__target": [1, 0, 2],
            "classes": [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        }
    )
    score = OrdinalLossScorer(pred_column="predictions", target="__target").score(data)
    assert score > 0
    assert score < 0.693


def test_sklearn_scorer_multiclass_log_loss():
    data = pd.DataFrame(
        {
            "predictions": [[0.1, 0.6, 0.3], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]],
            "__target": [1, 0, 2],
            "classes": [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        }
    )
    score = SklearnScorer(
        pred_column="predictions", scorer_function=log_loss, target="__target"
    ).score(data)
    assert score > 0
    assert score < 0.693


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"predictions": [0.1, 0.6, 0.3], "__target": [0, 1, 0]}),
        pl.DataFrame({"predictions": [0.1, 0.6, 0.3], "__target": [0, 1, 0]}),
    ],
)
def test_sk_learn_scorer_mean_absolute_error(df):
    score = SklearnScorer(
        pred_column="predictions",
        scorer_function=mean_absolute_error,
        target="__target",
    ).score(df)

    expected_score = sum([0.1, 0.4, 0.3]) / 3
    assert score == expected_score


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"predictions": [0.5, 0.6, 0.3], "__target": [0, 1, 0]}),
        pl.DataFrame({"predictions": [0.5, 0.6, 0.3], "__target": [0, 1, 0]}),
    ],
)
def test_mean_absolute_error(df):

    scorer = MeanBiasScorer(pred_column="predictions", target="__target")

    score = scorer.score(df)
    expected_score = sum([0.5, -0.4, 0.3]) / 3
    assert score == expected_score
