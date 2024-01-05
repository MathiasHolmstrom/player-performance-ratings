import copy
from unittest import mock

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from player_performance_ratings import ColumnNames
from player_performance_ratings.predictor.estimators import Predictor
from player_performance_ratings.ratings import RatingColumnNames
from player_performance_ratings.ratings.opponent_adjusted_rating import OpponentAdjustedRatingGenerator
from player_performance_ratings.tuner import PerformancesGeneratorTuner

from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.utils import ParameterSearchRange


def test_transformer_tuner():
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="performance"
    )
    performances_weight_search_ranges = {
        column_names.performance:
            [
                ParameterSearchRange(
                    name='kills',
                    type='uniform',
                    low=0.1,
                    high=0.3
                ),
                ParameterSearchRange(
                    name='won',
                    type='uniform',
                    low=0.25,
                    high=0.85
                )
            ]
    }

    rating_generator1 = OpponentAdjustedRatingGenerator(column_names=column_names,
                                                        features_out=[RatingColumnNames.RATING_DIFFERENCE_PROJECTED])

    rating_generators = [rating_generator1]

    match_predictor_factory = MatchPredictorFactory(
        rating_generators=rating_generators,
        predictor=Predictor(estimator=LogisticRegression(), features=[RatingColumnNames.RATING_DIFFERENCE_PROJECTED]),
        date_column_name="start_date",
    )

    performances_generator_tuner = PerformancesGeneratorTuner(
        performances_weight_search_ranges=performances_weight_search_ranges,
        n_trials=2)

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "kills": [0.6, 0.4, 0.5, 0.5],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1]
        }
    )

    scorer = mock.Mock()
    scorer.score.side_effect = [0.5, 0.3]

    tuned_model = performances_generator_tuner.tune(match_predictor_factory=copy.deepcopy(match_predictor_factory),
                                                    df=df, scorer=scorer)

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    # assert best model belongs in search range
    assert tuned_model.column_weights[0][0].weight >= 0.1
    assert tuned_model.column_weights[0][0].weight <= 0.3

    assert tuned_model.column_weights[0][1].weight >= 0.25
    assert tuned_model.column_weights[0][1].weight <= 0.85


@pytest.mark.parametrize("estimator", [LogisticRegression(), LinearRegression()])
def test_transformer_tuner_2_performances(estimator):
    column_names1 = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="performance1"
    )

    column_names2 = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="performance2"
    )

    performances_weight_search_ranges = {
        column_names1.performance:
            [
                ParameterSearchRange(
                    name='kills',
                    type='uniform',
                    low=0.1,
                    high=0.3
                ),
                ParameterSearchRange(
                    name='won',
                    type='uniform',
                    low=0.25,
                    high=0.85
                )
            ],
        column_names2.performance:
            [
                ParameterSearchRange(
                    name='kills',
                    type='uniform',
                    low=0.7,
                    high=0.9
                ),
                ParameterSearchRange(
                    name='won',
                    type='uniform',
                    low=0.25,
                    high=0.3
                )
            ],

    }

    rating_generator1 = OpponentAdjustedRatingGenerator(column_names=column_names1,
                                                        features_out=[RatingColumnNames.RATING_DIFFERENCE_PROJECTED])
    rating_generator2 = OpponentAdjustedRatingGenerator(column_names=column_names2,
                                                        features_out=[RatingColumnNames.RATING_DIFFERENCE_PROJECTED])

    rating_generators = [rating_generator1, rating_generator2]

    match_predictor_factory = MatchPredictorFactory(
        rating_generators=rating_generators,
        estimator=LogisticRegression(),
        date_column_name="start_date",
    )

    performances_generator_tuner = PerformancesGeneratorTuner(
        performances_weight_search_ranges=performances_weight_search_ranges,
        n_trials=2)

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "kills": [0.6, 0.4, 0.5, 0.5],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1]
        }
    )

    scorer = mock.Mock()
    scorer.score.side_effect = [0.5, 0.3]

    tuned_model = performances_generator_tuner.tune(match_predictor_factory=copy.deepcopy(match_predictor_factory),
                                                    df=df, scorer=scorer)

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    # assert best model belongs in search range
    assert tuned_model.column_weights[0][0].weight < tuned_model.column_weights[0][1].weight
    assert tuned_model.column_weights[1][0].weight > tuned_model.column_weights[1][1].weight
