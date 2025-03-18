from unittest import mock

import pandas as pd
import polars as pl
import pytest

from player_performance_ratings import ColumnNames
from sklearn.metrics import mean_absolute_error

from player_performance_ratings.scorer.score import SklearnScorer

from player_performance_ratings.cross_validator.cross_validator import (
    MatchKFoldCrossValidator,
)


@pytest.fixture
def column_names():
    column_names = ColumnNames(
        match_id="match_id",
        start_date="date",
        team_id="team_id",
        player_id="player_id",
    )
    return column_names


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_match_k_fold_cross_validator(df, column_names):
    data = df(
        {
            "__target": [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            column_names.match_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            column_names.team_id: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            column_names.player_id: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "classes": [[0, 1] for _ in range(10)],
            column_names.start_date: [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-02"),
                pd.to_datetime("2020-01-03"),
                pd.to_datetime("2020-01-04"),
                pd.to_datetime("2020-01-05"),
                pd.to_datetime("2020-01-06"),
                pd.to_datetime("2020-01-07"),
                pd.to_datetime("2020-01-08"),
                pd.to_datetime("2020-01-09"),
                pd.to_datetime("2020-01-10"),
            ],
        }
    )

    scorer = SklearnScorer(
        pred_column="__target_prediction",
        target="__target",
        scorer_function=mean_absolute_error,
    )

    if isinstance(data, pd.DataFrame):
        return_add_prediction1 = data.head(5)
        return_add_prediction1["__target_prediction"] = [1, 1, 1, 1, 1]
        return_add_prediction1["__cv_match_number"] = [0, 1, 2, 3, 4]
        return_add_prediction2 = data.tail(5)
        return_add_prediction2["__target_prediction"] = [0, 0, 0, 0, 1]
        return_add_prediction2["__cv_match_number"] = [5, 6, 7, 8, 9]
    else:
        return_add_prediction1 = data.head(5).with_columns(
            [
                pl.lit(1.0).alias("__target_prediction"),
                pl.Series("__cv_match_number", [0, 1, 2, 3, 4]),
            ]
        )

        return_add_prediction2 = data.tail(5).with_columns(
            [
                pl.Series("__target_prediction", [0.0, 0.0, 0.0, 0.0, 1.0]),
                pl.Series("__cv_match_number", [5, 6, 7, 8, 9]),
            ]
        )

    predictor = mock.Mock()
    predictor.predict.side_effect = [
        return_add_prediction1,
        return_add_prediction2,
    ]
    predictor.columns_added = ["__target_prediction"]

    cv = MatchKFoldCrossValidator(
        scorer=scorer,
        match_id_column_name="match_id",
        n_splits=2,
        date_column_name="date",
        min_validation_date="2020-01-02",
        predictor=predictor,
    )

    validation_df = cv.generate_validation_df(
        df=data, column_names=column_names, add_train_prediction=False
    )

    score = cv.cross_validation_score(validation_df=validation_df)
    assert score == 0.1


def test_match_k_fold_cross_validator_add_train_prediction(column_names):
    df = pd.DataFrame(
        {
            "__target": [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            column_names.match_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            column_names.team_id: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            column_names.player_id: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "classes": [[0, 1] for _ in range(10)],
            column_names.start_date: [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2020-01-02"),
                pd.to_datetime("2020-01-03"),
                pd.to_datetime("2020-01-04"),
                pd.to_datetime("2020-01-05"),
                pd.to_datetime("2020-01-06"),
                pd.to_datetime("2020-01-07"),
                pd.to_datetime("2020-01-08"),
                pd.to_datetime("2020-01-09"),
                pd.to_datetime("2020-01-10"),
            ],
        }
    )

    scorer = SklearnScorer(
        pred_column="__target_prediction",
        target="__target",
        scorer_function=mean_absolute_error,
    )
    predictor = mock.Mock()
    predictor.columns_added = ["__target_prediction"]
    return_value = df.copy()
    return_value["__target_prediction"] = 3.2
    return_value["__cv_match_number"] = list(range(len(df)))
    predictor.predict.return_value = return_value

    cv = MatchKFoldCrossValidator(
        scorer=scorer,
        match_id_column_name="match_id",
        n_splits=2,
        date_column_name="date",
        min_validation_date="2020-01-02",
        predictor=predictor,
    )

    validation_df = cv.generate_validation_df(
        df=df, add_train_prediction=True, column_names=column_names
    )

    assert validation_df["__target_prediction"].unique()[0] == 3.2
    assert len(validation_df["__target_prediction"].unique()) == 1
