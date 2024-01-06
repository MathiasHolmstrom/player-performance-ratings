from unittest import mock

import numpy as np
import pandas as pd
from deepdiff import DeepDiff

from player_performance_ratings.predictor.estimators import Predictor
from sklearn.linear_model import LogisticRegression

from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner
from player_performance_ratings.tuner.utils import ParameterSearchRange


def test_predictor_tuner():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "rating_difference": [100, -100, -20, 20],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1]
        }
    )

    match_predictor_factory = MatchPredictorFactory(
        predictor=Predictor(estimator=LogisticRegression(), features=["rating_difference"], target="__target"),
        date_column_name="start_date",
        train_split_date="2020-01-02"

    )

    search_ranges = [
        ParameterSearchRange(
            name='C',
            type='categorical',
            choices=[1.0, 0.5]
        )
    ]

    predictor_tuner = PredictorTuner(search_ranges=search_ranges, n_trials=2, date_column_name="start_date",
                                     train_split_date="2020-01-01")
    scorer = mock.Mock()
    scorer.score.side_effect = [0.5, 0.3]
    best_predictor = predictor_tuner.tune(df=df, scorer=scorer, match_predictor_factory=match_predictor_factory)

    expected_best_predictor = Predictor(estimator=LogisticRegression(C=0.5), features=["rating_difference"],
                                        target="__target")

    diff = DeepDiff(best_predictor.estimator, expected_best_predictor.estimator)
    assert diff == {}

    assert expected_best_predictor.features == best_predictor.features
    assert expected_best_predictor.target == best_predictor.target


def test_predictor_tuner_without_explicit_predictor():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "rating_difference": [100, -100, -20, 20],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1]
        }
    )

    match_predictor_factory = MatchPredictorFactory(
        date_column_name="start_date",
        train_split_date="2020-01-02",
        estimator=LogisticRegression(),
        other_features=["rating_difference"],

    )

    search_ranges = [
        ParameterSearchRange(
            name='C',
            type='categorical',
            choices=[1.0, 0.5]
        )
    ]

    predictor_tuner = PredictorTuner(search_ranges=search_ranges, n_trials=2, date_column_name="start_date",
                                     train_split_date="2020-01-01")
    scorer = mock.Mock()
    scorer.score.side_effect = [0.5, 0.3]
    predictor_tuner.tune(df=df, scorer=scorer, match_predictor_factory=match_predictor_factory)
    assert np.array_equal(scorer.score.call_args[1]['classes_'], np.array([0, 1]))


