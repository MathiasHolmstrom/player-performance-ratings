import copy

import mock
import pandas as pd
from skbase.testing.utils.deep_equals import deep_equals
from sklearn.preprocessing import MinMaxScaler

from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import OpponentAdjustedRatingGenerator
from player_performance_ratings.transformation import SkLearnTransformerWrapper
from player_performance_ratings.tuner import MatchPredictorTuner, TransformerTuner
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.rating_generator_tuner import OpponentAdjustedRatingGeneratorTuner


def test_match_predictor_tuner():
    """
    Tests to ensure no mutation of the original match_predictor_factory
    When rating-generator-tuning is used, best rating-generator should not be the same as the one in the factory
     (probability should be extremely low at least, likely less than one in 10000)


    """

    match_predictor_factory = MatchPredictorFactory(
        rating_generators=OpponentAdjustedRatingGenerator(column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="won"
        )),
    )

    scorer_mock = mock.Mock()
    scorer_mock.score.side_effect = [0.5, 0.2, 0.3]

    original_match_predictor_factory = copy.deepcopy(match_predictor_factory)

    standard_scaler = SkLearnTransformerWrapper(transformer=MinMaxScaler(), features=["won"])
    pre_transformer_search_ranges = [
        (standard_scaler, []),
    ]

    pre_transformer_tuner = TransformerTuner(
        transformer_search_ranges=pre_transformer_search_ranges,
        pre_or_post="pre_rating",
        n_trials=1
    )

    rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
        team_rating_n_trials=1,
        start_rating_n_trials=0,
    )

    post_transformer_tuner = TransformerTuner(
        transformer_search_ranges=pre_transformer_search_ranges,
        pre_or_post="post_rating",
        n_trials=1
    )

    match_predictor_tuner = MatchPredictorTuner(
        scorer=scorer_mock,
        match_predictor_factory=match_predictor_factory,
        pre_transformer_tuner=pre_transformer_tuner,
        rating_generator_tuners=rating_generator_tuner,
        post_transformer_tuner=post_transformer_tuner,

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
    assert deep_equals(best_rating_model.post_rating_transformers, match_predictor_factory.post_transformers) == False
    assert deep_equals(best_rating_model.pre_rating_transformers, match_predictor_factory.pre_transformers) == False


def test_match_predictor_uses_rating_generator_from_factory_if_no_rating_generator_tuning():
    match_predictor_factory = MatchPredictorFactory(
        rating_generators=OpponentAdjustedRatingGenerator(column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="won"
        )),
    )
    original_match_predictor_factory = copy.deepcopy(match_predictor_factory)
    scorer_mock = mock.Mock()
    scorer_mock.score.side_effect = [0.5, 0.2, 0.3]

    standard_scaler = SkLearnTransformerWrapper(transformer=MinMaxScaler(), features=["won"])
    pre_transformer_search_ranges = [
        (standard_scaler, []),
    ]

    pre_transformer_tuner = TransformerTuner(
        transformer_search_ranges=pre_transformer_search_ranges,
        pre_or_post="pre_rating",
        n_trials=1
    )

    post_transformer_tuner = TransformerTuner(
        transformer_search_ranges=pre_transformer_search_ranges,
        pre_or_post="post_rating",
        n_trials=1
    )

    match_predictor_tuner = MatchPredictorTuner(
        scorer=scorer_mock,
        match_predictor_factory=match_predictor_factory,
        pre_transformer_tuner=pre_transformer_tuner,
        post_transformer_tuner=post_transformer_tuner,

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

    best_match_predictor = match_predictor_tuner.tune(df=df)
    deep_equals(best_match_predictor.rating_generators, original_match_predictor_factory.rating_generators)
