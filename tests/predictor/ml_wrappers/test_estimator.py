from unittest.mock import Mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor.estimators.estimator import SkLearnGameTeamPredictor, SklearnPredictor


def test_sklearn_game_team_predictor_add_prediction():
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])

    predictor = SkLearnGameTeamPredictor(game_id_colum='game_id', team_id_column='team_id',
                                         features=['feature1', 'feature2'], model=mock_model)

    df = pd.DataFrame(
        {'game_id': [1, 1, 2],
         'team_id': [1, 2, 1],
         'feature1': [0.1, 0.5, 0.3],
         'feature2': [0.4, 0.6, 0.8],
         PredictColumnNames.TARGET: [1, 0, 1]
         }
    )

    result = predictor.add_prediction(df)

    expected_df = df.copy()
    expected_df[predictor.pred_column] = [0.8, 0.4, 0.7]

    pd.testing.assert_frame_equal(result, expected_df, check_like=True)


def test_sklearn_wrapper_game_player():
    """
    When weight_column is used -->
      the injected model.train() should be called with weighted * feature1 grouped by game_id, team_id
    """

    mock_model = Mock()

    predictor = SkLearnGameTeamPredictor(game_id_colum='game_id', team_id_column='team_id',
                                         features=['feature1'], model=mock_model, weight_column='weight')

    df = pd.DataFrame(
        {
            'game_id': [1, 1, 1, 1],
            'team_id': [1, 1, 2, 2],
            "player_id": [1, 2, 3, 4],
            'feature1': [0.1, 0.5, 0.3, 0.4],
            'weight': [0.3, 0.8, 0.6, 0.45],
            "__target": [1, 1, 0, 0]
        }
    )

    predictor.train(df)
    feature_team1 = (0.1 * 0.3 + 0.5 * 0.8) / (0.3 + 0.8)
    feature_team2 = (0.3 * 0.6 + 0.4 * 0.45) / (0.6 + 0.45)

    expected_features = pd.DataFrame(
        {'feature1': [feature_team1, feature_team2],
         }
    )

    pd.testing.assert_frame_equal(mock_model.fit.call_args[0][0], expected_features, check_like=True)
    assert mock_model.fit.call_args[0][1].tolist() == [1, 0]


def test_sklearn_wrapper_sub_game_player():
    """
    When sub-game are used the same player can exist multiple times in the same game_id for the same team_id.
    In this case the weighted average should be calculated for each player and then the average of the players.
    The calculation should be the same as if it was different players on the same team
    -->

    """

    mock_model = Mock()

    predictor = SkLearnGameTeamPredictor(game_id_colum='game_id', team_id_column='team_id',
                                         features=['feature1'], model=mock_model, weight_column='weight')

    df = pd.DataFrame(
        {
            'game_id': [1, 1, 1, 1],
            'team_id': [1, 1, 1, 1],
            "player_id": [1, 2, 1, 2],
            'feature1': [0.1, 0.5, 0.1, 0.5],
            'weight': [0.3, 0.8, 0.6, 0.2],
            "__target": [1, 1, 1, 1]
        }
    )

    predictor.train(df)
    feature_team1 = (0.1 * 0.3 + 0.5 * 0.8 + 0.1 * 0.6 + 0.5 * 0.2) / (0.3 + 0.8 + 0.6 + 0.2)

    expected_features = pd.DataFrame(
        {'feature1': [feature_team1],
         }
    )

    pd.testing.assert_frame_equal(mock_model.fit.call_args[0][0], expected_features, check_like=True)
    assert mock_model.fit.call_args[0][1].tolist() == [1]

def test_sklearn_game_team_predictor_regressor():
    "should identify it's a regressor and train and predict works as intended"

    predictor = SkLearnGameTeamPredictor(game_id_colum='game_id', team_id_column='team_id',
                                         features=['feature1'], model=LinearRegression())

    df = pd.DataFrame(
        {
            'game_id': [1, 1, 1, 1],
            'team_id': [1, 1, 1, 1],
            "player_id": [1, 2, 1, 2],
            'feature1': [0.1, 0.5, 0.1, 0.5],
            'weight': [0.3, 0.8, 0.6, 0.2],
            "__target": [1, 1, 1, 1]
        }
    )

    predictor.train(df)
    df = predictor.add_prediction(df)
    assert predictor.pred_column in df.columns



def test_sklearn_wrapper_regressor():
    "should identify it's a regressor and train and predict works as intended"

    predictor = SklearnPredictor(
                                         features=['feature1'], model=LinearRegression())

    df = pd.DataFrame(
        { 'feature1': [0.1, 0.5, 0.1, 0.5],
            "__target": [1, 1, 1, 1]
        }
    )

    predictor.train(df)
    df = predictor.add_prediction(df)
    assert predictor.pred_column in df.columns


