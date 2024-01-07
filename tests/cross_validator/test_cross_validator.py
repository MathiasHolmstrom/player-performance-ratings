import mock
import pandas as pd
from sklearn.metrics import mean_absolute_error

from player_performance_ratings.scorer.score import SklearnScorer

from player_performance_ratings.cross_validator.cross_validator import DayCountCrossValidator


def test_day_count_cross_validator():
    scorer = SklearnScorer(
        pred_column='__target_prediction',
        target='__target',
        scorer_function=mean_absolute_error,
    )

    df = pd.DataFrame({
        '__target': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        'date': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-02'), pd.to_datetime('2021-01-03'),
                 pd.to_datetime('2021-01-04'), pd.to_datetime('2021-01-05'), pd.to_datetime('2021-01-06'),
                 pd.to_datetime('2021-01-07'), pd.to_datetime('2021-01-08'), pd.to_datetime('2021-01-09'),
                 pd.to_datetime('2021-01-10')]
    })

    expected_predictor_train1 = df.copy().iloc[0:6]
    expected_predictor_train1['__cv_day_number'] = [0, 1, 2, 3, 4, 5]
    expected_predictor_train2 = df.copy().iloc[:8]
    expected_predictor_train2['__cv_day_number'] = [0, 1, 2, 3, 4, 5, 6, 7]

    expected_predictor_validation1 = df.copy().iloc[6:8]
    expected_predictor_validation1['__cv_day_number'] = [6, 7]
    expected_predictor_validation2 = df.copy().iloc[8:10]
    expected_predictor_validation2['__cv_day_number'] = [8, 9]

    return_add_prediction1 = df.copy()
    return_add_prediction1['__target_prediction'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return_add_prediction2 = df.copy()
    return_add_prediction2['__target_prediction'] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]

    predictor = mock.Mock()
    predictor.add_prediction.side_effect = [return_add_prediction1, return_add_prediction2]

    cv = DayCountCrossValidator(predictor=predictor, scorer=scorer, date_column_name='date', n_splits=2,
                                validation_days=2)

    score = cv.cross_validate(df)

    assert cv.scores[0] == 0.5
    assert cv.scores[1] == 1

    assert score == 0.75

    pd.testing.assert_frame_equal(predictor.method_calls[0][1][0], expected_predictor_train1, check_like=True,
                                  check_dtype=False)
    pd.testing.assert_frame_equal(predictor.method_calls[2][1][0], expected_predictor_train2, check_like=True,
                                  check_dtype=False)

    pd.testing.assert_frame_equal(predictor.method_calls[1][1][0], expected_predictor_validation1, check_like=True,
                                  check_dtype=False)
    pd.testing.assert_frame_equal(predictor.method_calls[3][1][0], expected_predictor_validation2, check_like=True,
                                  check_dtype=False)
