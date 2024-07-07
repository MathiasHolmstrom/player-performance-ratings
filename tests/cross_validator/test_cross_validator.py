from unittest import mock

import pandas as pd
import pytest

from player_performance_ratings import ColumnNames
from sklearn.metrics import mean_absolute_error

from player_performance_ratings.scorer.score import SklearnScorer

from player_performance_ratings.cross_validator.cross_validator import \
    MatchKFoldCrossValidator

@pytest.fixture
def column_names():
    column_names = ColumnNames(
        match_id='match_id',
        start_date='date',
        team_id='team_id',
        player_id='player_id',
    )
    return column_names

def test_match_k_fold_cross_validator(column_names, df):
    df = pd.DataFrame({
        '__target': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        column_names.match_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        column_names.team_id: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        column_names.player_id: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "classes": [[0, 1] for _ in range(10)],
        column_names.start_date: [pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-02'),
                                  pd.to_datetime('2020-01-03'),
                                  pd.to_datetime('2020-01-04'), pd.to_datetime('2020-01-05'),
                                  pd.to_datetime('2020-01-06'),
                                  pd.to_datetime('2020-01-07'), pd.to_datetime('2020-01-08'),
                                  pd.to_datetime('2020-01-09'),
                                  pd.to_datetime('2020-01-10')],
    })

    scorer = SklearnScorer(
        pred_column='__target_prediction',
        target='__target',
        scorer_function=mean_absolute_error,
    )

    expected_predictor_train1 = df.copy().iloc[0:1]
    expected_predictor_train1['__cv_match_number'] = [0]
    expected_predictor_train2 = df.copy().iloc[:5]
    expected_predictor_train2['__cv_match_number'] = [0, 1, 2, 3, 4]

    expected_predictor_validation1 = df.copy().iloc[1:5]
    expected_predictor_validation1['__cv_match_number'] = [1, 2, 3, 4]
    expected_predictor_validation2 = df.copy().iloc[5:10]
    expected_predictor_validation2['__cv_match_number'] = [5, 6, 7, 8, 9]

    return_add_prediction1 = df.head(5)
    return_add_prediction1['__target_prediction'] = [1, 1, 1, 1, 1]
    return_add_prediction1['__cv_match_number'] = [0, 1, 2, 3, 4]
    return_add_prediction2 = df.tail(5)
    return_add_prediction2['__target_prediction'] = [0, 0, 0, 0, 1]
    return_add_prediction2['__cv_match_number'] = [5, 6, 7, 8, 9]

    predictor = mock.Mock()
    predictor.add_prediction.side_effect = [return_add_prediction1, return_add_prediction2]
    predictor.columns_added = ['__target_prediction']

    cv = MatchKFoldCrossValidator(scorer=scorer, match_id_column_name='match_id', n_splits=2,
                                  date_column_name='date', min_validation_date='2020-01-02')

    validation_df = cv.generate_validation_df(df=df, predictor=predictor, estimator_features=[],
                                              column_names=column_names)
    score = cv.cross_validation_score(validation_df=validation_df)
    assert score == 0.1


def test_match_k_fold_cross_validator_add_train_prediction(column_names):
    df = pd.DataFrame({
        '__target': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        column_names.match_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        column_names.team_id: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        column_names.player_id: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "classes": [[0, 1] for _ in range(10)],
        column_names.start_date: [pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-02'),
                                  pd.to_datetime('2020-01-03'),
                                  pd.to_datetime('2020-01-04'), pd.to_datetime('2020-01-05'),
                                  pd.to_datetime('2020-01-06'),
                                  pd.to_datetime('2020-01-07'), pd.to_datetime('2020-01-08'),
                                  pd.to_datetime('2020-01-09'),
                                  pd.to_datetime('2020-01-10')],
    })

    scorer = SklearnScorer(
        pred_column='__target_prediction',
        target='__target',
        scorer_function=mean_absolute_error,
    )

    cv = MatchKFoldCrossValidator(scorer=scorer, match_id_column_name='match_id', n_splits=2,
                                  date_column_name='date', min_validation_date='2020-01-02')

    predictor = mock.Mock()
    predictor.columns_added = ['__target_prediction']
    return_value  = df.copy()
    return_value['__target_prediction'] = 3.2
    return_value['__cv_match_number'] = list(range(len(df)))
    predictor.add_prediction.return_value = return_value

    validation_df = cv.generate_validation_df(df=df, predictor=predictor, add_train_prediction=True, column_names=column_names)

    assert validation_df['__target_prediction'].unique()[0] == 3.2
    assert len(validation_df['__target_prediction'].unique()) == 1


