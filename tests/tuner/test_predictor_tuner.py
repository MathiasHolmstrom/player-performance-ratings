from unittest import mock

import pandas as pd
from deepdiff import DeepDiff

from player_performance_ratings import ColumnNames
from player_performance_ratings.predictor.estimators import Predictor
from sklearn.linear_model import LogisticRegression

from player_performance_ratings.ratings import convert_df_to_matches
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
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="won"
    )

    match_predictor_factory = MatchPredictorFactory(
        predictor=Predictor(estimator=LogisticRegression(), features=["rating_difference"], target="__target"),

    )

    search_ranges = [
        ParameterSearchRange(
            name='C',
            type='categorical',
            choices=[1.0, 0.5]
        )
    ]

    predictor_tuner = PredictorTuner(search_ranges=search_ranges, n_trials=2)
    scorer = mock.Mock()
    scorer.score.side_effect = [0.5, 0.3]
    matches = convert_df_to_matches(df=df, column_names=column_names)
    best_predictor = predictor_tuner.tune(df=df, matches=[matches], scorer=scorer, match_predictor_factory=match_predictor_factory)

    expected_best_predictor = Predictor(estimator=LogisticRegression(C=0.5), features=["rating_difference"], target="__target")

    diff = DeepDiff(best_predictor.estimator, expected_best_predictor.estimator)
    assert diff == {}

    assert expected_best_predictor.features == best_predictor.features
    assert expected_best_predictor.target == best_predictor.target
