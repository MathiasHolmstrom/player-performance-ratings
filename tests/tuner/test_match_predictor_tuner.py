import copy

import mock
import pandas as pd
from player_performance_ratings.tuner.utils import ParameterSearchRange
from skbase.testing.utils.deep_equals import deep_equals


from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings.rating_calculators import OpponentAdjustedRatingGenerator

from player_performance_ratings.tuner import MatchPredictorTuner, PerformancesGeneratorTuner
from player_performance_ratings.tuner.match_predictor_factory import PipelineFactory
from player_performance_ratings.tuner.rating_generator_tuner import OpponentAdjustedRatingGeneratorTuner


def test_match_predictor_tuner():
    """
    Tests to ensure no mutation of the original match_predictor_factory
    When rating-generator-tuning is used, best rating-generator should not be the same as the one in the factory
     (probability should be extremely low at least, likely less than one in 10000)


    """

    match_predictor_factory = PipelineFactory(
        rating_generators=OpponentAdjustedRatingGenerator(column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="won"
        )),
        match_id_column_name="game_id",
    )

    scorer_mock = mock.Mock()
    scorer_mock.score.side_effect = [0.5, 0.2, 0.3]

    original_match_predictor_factory = copy.deepcopy(match_predictor_factory)

    performances_weight_search_ranges = {
        "performance":
            [
                ParameterSearchRange(
                    name='won',
                    type='uniform',
                    low=0.8,
                    high=1
                )
            ],

    }

    performances_generator_tuner = PerformancesGeneratorTuner(
        performances_weight_search_ranges=performances_weight_search_ranges,
        n_trials=1)

    rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
        team_rating_n_trials=1,
        start_rating_n_trials=0,
    )

    match_predictor_tuner = MatchPredictorTuner(
        scorer=scorer_mock,
        match_predictor_factory=match_predictor_factory,
        performances_generator_tuner=performances_generator_tuner,
        rating_generator_tuners=rating_generator_tuner,
        date_column_name="start_date",
        cv_n_splits = 1

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

    best_rating_model = match_predictor_tuner.tune(df=df)
    deep_equals(match_predictor_factory, original_match_predictor_factory)
    assert deep_equals(best_rating_model.rating_generators, match_predictor_factory.rating_generators) == False
    assert deep_equals(best_rating_model.performances_generator, match_predictor_factory.performances_generator) == False

