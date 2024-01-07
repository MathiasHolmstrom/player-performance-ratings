import mock
import pandas as pd
from sklearn.metrics import mean_absolute_error

from player_performance_ratings.scorer.score import SklearnScorer

from player_performance_ratings.cross_validator.cross_validator import MatchCountCrossValidator, \
    MatchKFoldCrossValidator


def test_match_count_cross_validator():
    scorer = SklearnScorer(
        pred_column='__target_prediction',
        target='__target',
        scorer_function=mean_absolute_error,
    )

    df = pd.DataFrame({
        '__target': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        'match_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })

    expected_predictor_train1 = df.copy().iloc[0:6]
    expected_predictor_train1['__cv_match_number'] = [0, 1, 2, 3, 4, 5]
    expected_predictor_train2 = df.copy().iloc[:8]
    expected_predictor_train2['__cv_match_number'] = [0, 1, 2, 3, 4, 5, 6, 7]

    expected_predictor_validation1 = df.copy().iloc[6:8]
    expected_predictor_validation1['__cv_match_number'] = [6, 7]
    expected_predictor_validation2 = df.copy().iloc[8:10]
    expected_predictor_validation2['__cv_match_number'] = [8, 9]

    return_add_prediction1 = df.copy()
    return_add_prediction1['__target_prediction'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return_add_prediction2 = df.copy()
    return_add_prediction2['__target_prediction'] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]

    predictor = mock.Mock()
    predictor.add_prediction.side_effect = [return_add_prediction1, return_add_prediction2]

    cv = MatchCountCrossValidator(scorer=scorer, match_id_column_name='match_id', n_splits=2,
                                  validation_match_count=2)

    validation_df = cv.generate_validation_df(df, predictor=predictor)
    score = cv.cross_validation_score(validation_df=validation_df)

    assert score == 0.75

    pd.testing.assert_frame_equal(predictor.method_calls[0][1][0], expected_predictor_train1, check_like=True,
                                  check_dtype=False)
    pd.testing.assert_frame_equal(predictor.method_calls[2][1][0], expected_predictor_train2, check_like=True,
                                  check_dtype=False)

    pd.testing.assert_frame_equal(predictor.method_calls[1][1][0], expected_predictor_validation1, check_like=True,
                                  check_dtype=False)
    pd.testing.assert_frame_equal(predictor.method_calls[3][1][0], expected_predictor_validation2, check_like=True,
                                  check_dtype=False)


def test_match_k_fold_cross_validator():
    scorer = SklearnScorer(
        pred_column='__target_prediction',
        target='__target',
        scorer_function=mean_absolute_error,
    )

    df = pd.DataFrame({
        '__target': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        'match_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'date': [pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-02'), pd.to_datetime('2020-01-03'),
                 pd.to_datetime('2020-01-04'), pd.to_datetime('2020-01-05'), pd.to_datetime('2020-01-06'),
                 pd.to_datetime('2020-01-07'), pd.to_datetime('2020-01-08'), pd.to_datetime('2020-01-09'),
                 pd.to_datetime('2020-01-10')],
    })

    expected_predictor_train1 = df.copy().iloc[0:1]
    expected_predictor_train1['__cv_match_number'] = [0]
    expected_predictor_train2 = df.copy().iloc[:5]
    expected_predictor_train2['__cv_match_number'] = [0, 1, 2, 3, 4]

    expected_predictor_validation1 = df.copy().iloc[1:5]
    expected_predictor_validation1['__cv_match_number'] = [1, 2, 3, 4]
    expected_predictor_validation2 = df.copy().iloc[5:10]
    expected_predictor_validation2['__cv_match_number'] = [5, 6, 7, 8, 9]

    return_add_prediction1 = df.copy()
    return_add_prediction1['__target_prediction'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return_add_prediction2 = df.copy()
    return_add_prediction2['__target_prediction'] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]

    predictor = mock.Mock()
    predictor.add_prediction.side_effect = [return_add_prediction1, return_add_prediction2]

    cv = MatchKFoldCrossValidator(scorer=scorer, match_id_column_name='match_id', n_splits=2,
                                  date_column_name='date', min_validation_date='2020-01-02')

    validation_df = cv.generate_validation_df(df, predictor=predictor)
    score = cv.cross_validation_score(validation_df=validation_df)

    assert score == 0.75

    pd.testing.assert_frame_equal(predictor.method_calls[0][1][0], expected_predictor_train1, check_like=True,
                                  check_dtype=False)
    pd.testing.assert_frame_equal(predictor.method_calls[2][1][0], expected_predictor_train2, check_like=True,
                                  check_dtype=False)

    pd.testing.assert_frame_equal(predictor.method_calls[1][1][0], expected_predictor_validation1, check_like=True,
                                  check_dtype=False)
    pd.testing.assert_frame_equal(predictor.method_calls[3][1][0], expected_predictor_validation2, check_like=True,
                                  check_dtype=False)
