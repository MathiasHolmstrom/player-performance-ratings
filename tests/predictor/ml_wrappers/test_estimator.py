from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import pytest
from lightgbm import LGBMClassifier

from sklearn.linear_model import LinearRegression, LogisticRegression

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor import (
    GameTeamPredictor,
    OrdinalClassifier,
    Predictor,
)
from player_performance_ratings.predictor.predictor import GranularityPredictor


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_game_team_predictor_add_predictiondf(df):
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array(
        [[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    )
    mock_model.estimator = LogisticRegression()

    predictor = GameTeamPredictor(
        game_id_colum="game_id", team_id_column="team_id", estimator=mock_model
    )
    predictor._estimator_features = ["feature1", "feature2"]
    data = df(
        {
            "game_id": [1, 1, 2],
            "team_id": [1, 2, 1],
            "feature1": [0.1, 0.5, 0.3],
            "feature2": [0.4, 0.6, 0.8],
            PredictColumnNames.TARGET: [1, 0, 1],
        }
    )

    result = predictor.add_prediction(data)
    if isinstance(data, pd.DataFrame):
        expected_df = data.copy()
        expected_df[predictor.pred_column] = [0.8, 0.4, 0.7]

        pd.testing.assert_frame_equal(
            result, expected_df, check_like=True, check_dtype=False
        )

    else:
        expected_df = data.with_columns(
            pl.Series(predictor.pred_column, [0.8, 0.4, 0.7])
        )
        assert_frame_equal(result, expected_df, check_dtype=False)


@pytest.mark.parametrize(
    "predictor",
    [
        GameTeamPredictor(
            multiclass_output_as_struct=True,
            game_id_colum="game_id",
            team_id_column="team_id",
        ),
        Predictor(),
        GranularityPredictor(granularity_column_name="position"),
    ],
)
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_multiclass_train(predictor, df):
    data = df(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "team_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.3, 0.4, 0.4, 0.3, 0.6, 0.4],
            "__target": [1, 1, 0, 0, 2, 2, 3, 3],
            "position": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    )

    predictor.train(data, estimator_features=["feature1"])

    df_with_predictions = predictor.add_prediction(data)
    assert predictor.pred_column in df_with_predictions.columns
    if isinstance(df_with_predictions, pd.DataFrame):
        df_with_predictions = pl.DataFrame(df_with_predictions)
    probs_list = df_with_predictions.select(
        pl.concat_list(pl.col(predictor.pred_column).struct.unnest()).alias("fields")
    )["fields"].to_list()

    for values in probs_list:
        assert sum(values) == 1


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_game_team_predictor_game_player(df):
    """ """

    mock_model = Mock()
    mock_model.estimator = LGBMClassifier()
    predictor = GameTeamPredictor(
        game_id_colum="game_id",
        team_id_column="team_id",
        estimator=mock_model,
        pre_transformers=[],
    )

    data = df(
        {
            "game_id": [1, 1, 1, 1],
            "team_id": [1, 1, 2, 2],
            "player_id": [1, 2, 3, 4],
            "feature1": [0.1, 0.5, 0.3, 0.4],
            "__target": [1, 1, 0, 0],
        }
    )

    predictor.train(data, estimator_features=["feature1"])
    feature_team1 = (0.1 * 0.5 + 0.5 * 0.5) / (0.5 + 0.5)
    feature_team2 = (0.3 * 0.5 + 0.4 * 0.5) / (0.5 + 0.5)

    if isinstance(data, pd.DataFrame):
        expected_features = pd.DataFrame(
            {
                "feature1": [feature_team1, feature_team2],
            }
        )
        pd.testing.assert_frame_equal(
            mock_model.fit.call_args[0][0].to_native(),
            expected_features,
            check_dtype=False,
        )
    else:
        expected_features = pl.DataFrame(
            {
                "feature1": [feature_team1, feature_team2],
            },
        )
        assert_frame_equal(
            mock_model.fit.call_args[0][0].to_native(),
            expected_features,
            check_dtype=False,
        )

    assert mock_model.fit.call_args[0][1].to_list() == [1, 0]


@pytest.mark.parametrize("target_values", [[1, 0, 1, 0], [0.3, 0.2, 24, 0.5]])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_game_team_predictor(target_values, df):
    "should identify it's a regressor and train and predict works as intended"

    predictor = GameTeamPredictor(
        game_id_colum="game_id", team_id_column="team_id", estimator=LinearRegression()
    )

    data = df(
        {
            "game_id": [1, 1, 1, 1],
            "team_id": [1, 1, 1, 1],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "weight": [0.3, 0.8, 0.6, 0.2],
            "__target": target_values,
        }
    )

    predictor.train(data, estimator_features=["feature1"])
    df = predictor.add_prediction(data)
    assert predictor.pred_column in df.columns


@pytest.mark.parametrize("target_values", [[1, 0, 1, 0], [0.3, 0.2, 24, 0.5]])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_predictor_regressor(target_values, df):
    "should identify it's a regressor and train and predict works as intended"

    predictor = Predictor(estimator=LinearRegression())

    data = df(
        {
            "game_id": [1, 1, 1, 1],
            "team_id": [1, 1, 1, 1],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "weight": [0.3, 0.8, 0.6, 0.2],
            "__target": target_values,
        }
    )

    predictor.train(data, estimator_features=["feature1"])
    df = predictor.add_prediction(data)
    assert predictor.pred_column in df.columns


@pytest.mark.parametrize("target_values", [[1, 0, 1, 0], [0.3, 0.2, 24, 0.5]])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_granularity_predictor(target_values, df):
    "should identify it's a regressor and train and predict works as intended"

    predictor = GranularityPredictor(
        estimator=LinearRegression(), granularity_column_name="position"
    )

    data = df(
        {
            "position": ["a", "b", "a", "b"],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "weight": [0.3, 0.8, 0.6, 0.2],
            "__target": [1, 1, 1, 1],
        }
    )

    predictor.train(data, estimator_features=["feature1"])
    df = predictor.add_prediction(data)
    assert predictor.pred_column in df.columns
