from unittest.mock import Mock

import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression, LogisticRegression

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor import GameTeamPredictor, OrdinalClassifier, Predictor


def test_game_team_predictor_add_prediction():
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])

    predictor = GameTeamPredictor(game_id_colum='game_id', team_id_column='team_id', estimator=mock_model)
    predictor._estimator_features = ['feature1', 'feature2']
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


def test_game_team_predictor_multiclass_train():
    predictor = Predictor(estimator=OrdinalClassifier())

    df = pd.DataFrame(
        {
            'game_id': [1, 1, 1, 1],
            'team_id': [1, 1, 2, 2],
            "player_id": [1, 2, 3, 4],
            'feature1': [0.1, 0.5, 0.3, 0.4],
            "__target": [1, 0, 2, 3]
        }
    )

    predictor.train(df, estimator_features=['feature1'])
    assert len(predictor.estimator.classes_) == 4

    df_with_predictions = predictor.add_prediction(df)
    assert predictor.pred_column in df_with_predictions.columns



def test_game_team_predictor_game_player():
    """
    """

    mock_model = Mock()
    mock_model.estimator = LogisticRegression()
    predictor = GameTeamPredictor(game_id_colum='game_id', team_id_column='team_id', estimator=mock_model)

    df = pd.DataFrame(
        {
            'game_id': [1, 1, 1, 1],
            'team_id': [1, 1, 2, 2],
            "player_id": [1, 2, 3, 4],
            'feature1': [0.1, 0.5, 0.3, 0.4],
            "__target": [1, 1, 0, 0]
        }
    )

    predictor.train(df, estimator_features=['feature1'])
    feature_team1 = (0.1 * 0.5 + 0.5 * 0.5) / (0.5 + 0.5)
    feature_team2 = (0.3 * 0.5 + 0.4 * 0.5) / (0.5 + 0.5)

    expected_features = pd.DataFrame(
        {'feature1': [feature_team1, feature_team2],
         }
    )

    pd.testing.assert_frame_equal(mock_model.fit.call_args[0][0], expected_features, check_like=True)
    assert mock_model.fit.call_args[0][1].tolist() == [1, 0]



def test_game_team_predictor_regressor():
    "should identify it's a regressor and train and predict works as intended"

    predictor = GameTeamPredictor(game_id_colum='game_id', team_id_column='team_id',estimator=LinearRegression())

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

    predictor.train(df, estimator_features=['feature1'])
    df = predictor.add_prediction(df)
    assert predictor.pred_column in df.columns


def test_predictor_regressor():
    "should identify it's a regressor and train and predict works as intended"

    predictor = Predictor(estimator=LinearRegression())

    df = pd.DataFrame(
        {'feature1': [0.1, 0.5, 0.1, 0.5],
         "__target": [1, 1, 1, 1]
         }
    )

    predictor.train(df, estimator_features=['feature1'])
    df = predictor.add_prediction(df)
    assert predictor.pred_column in df.columns
