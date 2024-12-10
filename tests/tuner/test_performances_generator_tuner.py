import copy
from unittest import mock

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from player_performance_ratings import ColumnNames, PipelineFactory
from player_performance_ratings.predictor import Predictor
from player_performance_ratings.ratings import (
    RatingKnownFeatures,
    UpdateRatingGenerator,
)
from player_performance_ratings.ratings.performance_generator import (
    Performance,
    PerformancesGenerator,
    ColumnWeight,
)

from player_performance_ratings.tuner import PerformancesGeneratorTuner
from player_performance_ratings.tuner.performances_generator_tuner import (
    PerformancesSearchRange,
)

from player_performance_ratings.tuner.utils import ParameterSearchRange


def test_transformer_tuner():
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )

    rating_generator1 = UpdateRatingGenerator(
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
    )

    pipeline_factory = PipelineFactory(
        performances_generator=PerformancesGenerator(
            performances=Performance(name="performance", weights=[]), auto_generated_features_prefix=""
        ),
        column_names=column_names,
        rating_generators=rating_generator1,
        predictor=Predictor(
            estimator=LogisticRegression(),
            estimator_features=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        ),
    )

    performances_generator_tuner = PerformancesGeneratorTuner(
        performances_search_range=PerformancesSearchRange(
            search_ranges=[
                ParameterSearchRange(name="kills", type="uniform", low=0.1, high=0.3),
                ParameterSearchRange(name="won", type="uniform", low=0.25, high=0.85),
            ]
        ),
        n_trials=2,
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "kills": [0.6, 0.4, 0.5, 0.5],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1],
        }
    )

    cross_validator = mock.Mock()
    cross_validator.cross_validation_score.side_effect = [0.5, 0.3]

    tuned_model = performances_generator_tuner.tune(
        pipeline_factory=copy.deepcopy(pipeline_factory),
        df=df,
        cross_validator=cross_validator,
        rating_idx=0,
    )

    # tests immutability of match_predictor_factory
    assert pipeline_factory.rating_generators == [rating_generator1]

    # assert best model belongs in search range
    assert tuned_model.performances[0].weights[0].weight >= 0.1
    assert tuned_model.performances[0].weights[0].weight <= 0.3

    assert tuned_model.performances[0].weights[1].weight >= 0.25
    assert tuned_model.performances[0].weights[1].weight <= 0.85


@pytest.mark.parametrize("estimator", [LogisticRegression(), LinearRegression()])
def test_transformer_tuner_2_performances(estimator):
    column_names1 = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )

    performances_search_range = [
        PerformancesSearchRange(
            name="performance1",
            search_ranges=[
                ParameterSearchRange(name="kills", type="uniform", low=0.1, high=0.3),
                ParameterSearchRange(name="won", type="uniform", low=0.25, high=0.85),
            ],
        ),
        PerformancesSearchRange(
            name="performance2",
            search_ranges=[
                ParameterSearchRange(name="kills", type="uniform", low=0.7, high=0.9),
                ParameterSearchRange(name="won", type="uniform", low=0.25, high=0.3),
            ],
        ),
    ]

    rating_generator1 = UpdateRatingGenerator(
        performance_column="performance1",
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
    )

    rating_generator2 = UpdateRatingGenerator(
        performance_column="performance2",
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        prefix="rating2",
    )

    rating_generators = [rating_generator1, rating_generator2]

    match_predictor_factory = PipelineFactory(
        rating_generators=rating_generators,
        predictor=Predictor(
            estimator=estimator,
            estimator_features=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        ),
        column_names=column_names1,
        performances_generator=PerformancesGenerator(
         auto_generated_features_prefix="",
            performances=[
                Performance(
                    name="performance1",
                    weights=[
                        ColumnWeight(name="kills", weight=0.5),
                        ColumnWeight(name="won", weight=0.5),
                    ],
                ),
                Performance(
                    name="performance2",
                    weights=[
                        ColumnWeight(name="kills", weight=0.5),
                        ColumnWeight(name="won", weight=0.5),
                    ],
                ),
            ],
        ),
    )

    performances_generator_tuner = PerformancesGeneratorTuner(
        performances_search_range=performances_search_range, n_trials=2
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "kills": [0.6, 0.4, 0.5, 0.5],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1],
        }
    )

    cross_validator = mock.Mock()
    cross_validator.cross_validation_score.side_effect = [0.5, 0.3]

    tuned_model = performances_generator_tuner.tune(
        pipeline_factory=copy.deepcopy(match_predictor_factory),
        df=df,
        cross_validator=cross_validator,
        rating_idx=0,
    )

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    # assert best model belongs in search range
    assert (
        tuned_model.performances[0].weights[0].weight
        < tuned_model.performances[0].weights[1].weight
    )
    assert (
        tuned_model.performances[0].weights[1].weight
        > tuned_model.performances[1].weights[1].weight
    )
