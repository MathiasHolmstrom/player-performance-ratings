import copy
from unittest import mock

import pandas as pd
from skbase.utils import deep_equals

from player_performance_ratings.predictor import Predictor
from player_performance_ratings.tuner.performances_generator_tuner import PerformancesSearchRange
from player_performance_ratings.tuner.utils import ParameterSearchRange

from player_performance_ratings import ColumnNames, Pipeline
from player_performance_ratings.ratings import UpdateRatingGenerator

from player_performance_ratings.tuner import PipelineTuner, PerformancesGeneratorTuner

from player_performance_ratings.tuner.rating_generator_tuner import UpdateRatingGeneratorTuner


def test_match_predictor_tuner():
    """
    Tests to ensure no mutation of the original match_predictor_factory
    When rating-generator-tuning is used, best rating-generator should not be the same as the one in the factory
     (probability should be extremely low at least, likely less than one in 10000)


    """

    col_names = ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        )



    pipeline = Pipeline(
        column_names=col_names,
        rating_generators=UpdateRatingGenerator(performance_column="performance"),
        predictor=Predictor()

    )

    cv_mock = mock.Mock()
    cv_mock.cross_validation_score.side_effect = [0.5, 0.2, 0.3]

    original_pipeline = copy.deepcopy(pipeline)

    performance_search_range = PerformancesSearchRange(
        search_ranges=[
                ParameterSearchRange(
                    name='won',
                    type='uniform',
                    low=0.8,
                    high=1
                )
            ]
    )

    performances_generator_tuner = PerformancesGeneratorTuner(
        performances_search_range=performance_search_range,
        n_trials=1)

    rating_generator_tuner = UpdateRatingGeneratorTuner(
        team_rating_n_trials=1,
        start_rating_n_trials=0,
    )

    pipeline_tuner = PipelineTuner(
        cross_validator=cv_mock,
        pipeline=pipeline,
        performances_generator_tuners=performances_generator_tuner,
        rating_generator_tuners=rating_generator_tuner,


    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "start_date": [pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-02"),
                           pd.to_datetime("2021-01-02")],
            "won": [1, 0, 1, 0],
            "__target": [1, 0, 1, 0],

        }
    )

    best_rating_model = pipeline_tuner.tune(df=df)
    deep_equals(pipeline, original_pipeline)
    assert deep_equals(best_rating_model.rating_generators, pipeline.rating_generators) == False
    assert deep_equals(best_rating_model.performances_generator, pipeline.performances_generator) == False
