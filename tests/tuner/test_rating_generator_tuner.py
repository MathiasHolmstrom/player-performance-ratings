import copy
from unittest import mock

import pandas as pd
from player_performance_ratings.predictor import SklearnPredictor

from player_performance_ratings import ColumnNames, PipelineFactory
from player_performance_ratings.ratings.rating_calculators import MatchRatingGenerator
from player_performance_ratings.ratings import (
    RatingKnownFeatures,
    UpdateRatingGenerator,
)
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.ratings.rating_calculators.performance_predictor import (
    RatingDifferencePerformancePredictor,
)
from player_performance_ratings.ratings.rating_calculators.start_rating_generator import (
    StartRatingGenerator,
)

from player_performance_ratings.tuner.rating_generator_tuner import (
    UpdateRatingGeneratorTuner,
)
from player_performance_ratings.tuner.utils import ParameterSearchRange


def test_opponent_adjusted_rating_generator_tuner_team_rating():
    team_rating_search = [
        ParameterSearchRange(
            name="confidence_days_ago_multiplier",
            type="uniform",
            low=0.02,
            high=0.12,
        )
    ]
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )

    rating_generator1 = UpdateRatingGenerator(
        match_rating_generator=MatchRatingGenerator(confidence_weight=0.5)
    )
    rating_generator2 = UpdateRatingGenerator(
        match_rating_generator=MatchRatingGenerator(confidence_weight=0.4),
        prefix="rating2",
    )
    rating_generators = [rating_generator1, rating_generator2]

    pipeline_factory = PipelineFactory(
        rating_generators=rating_generators,
        predictor=SklearnPredictor(target="__target"),
        column_names=column_names,
    )

    rating_generator_tuner = UpdateRatingGeneratorTuner(
        start_rating_n_trials=0,
        team_rating_n_trials=2,
        team_rating_search_ranges=team_rating_search,
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1],
        }
    )

    matches = convert_df_to_matches(
        df=df, column_names=column_names, performance_column_name="won"
    )

    cross_validator = mock.Mock()
    cross_validator.cross_validation_score.side_effect = [0.5, 0.3]

    tuned_model = rating_generator_tuner.tune(
        pipeline_factory=copy.deepcopy(pipeline_factory),
        rating_idx=1,
        matches=matches,
        df=df,
        cross_validator=cross_validator,
    )

    # tests immutability of match_predictor_factory
    assert pipeline_factory.rating_generators == rating_generators

    # Second model has lowest score so it should be equal to confidence weight of that model
    assert tuned_model.match_rating_generator.confidence_weight == 0.4

    # ratings should be reset
    assert tuned_model.player_ratings == {}
    assert tuned_model.team_ratings == []


def test_opponent_adjusted_rating_generator_tuner_performance_predictor():
    team_rating_search = [
        ParameterSearchRange(
            name="confidence_days_ago_multiplier",
            type="uniform",
            low=0.02,
            high=0.12,
        )
    ]
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )

    rating_generator1 = UpdateRatingGenerator(
        performance_column="won",
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0.5,
            performance_predictor=RatingDifferencePerformancePredictor(
                max_predict_value=0.3
            ),
        ),
    )
    rating_generator2 = UpdateRatingGenerator(
        performance_column="won",
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0.4,
            performance_predictor=RatingDifferencePerformancePredictor(
                max_predict_value=0.4
            ),
        ),
        prefix="rating2",
    )
    rating_generators = [rating_generator1, rating_generator2]

    match_predictor_factory = PipelineFactory(
        rating_generators=rating_generators,
        predictor=SklearnPredictor(
            estimator_features=[
                f"{RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED}0",
                f"{RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED}1",
            ],
            target="__target",
        ),
        column_names=column_names,
    )

    rating_generator_tuner = UpdateRatingGeneratorTuner(
        start_rating_n_trials=0,
        team_rating_n_trials=2,
        team_rating_search_ranges=team_rating_search,
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1],
        }
    )

    matches = convert_df_to_matches(
        df=df, column_names=column_names, performance_column_name="won"
    )

    cross_validator = mock.Mock()
    cross_validator.cross_validation_score.side_effect = [0.5, 0.3]

    tuned_model = rating_generator_tuner.tune(
        pipeline_factory=copy.deepcopy(match_predictor_factory),
        rating_idx=1,
        matches=matches,
        df=df,
        cross_validator=cross_validator,
    )

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    # Second model has lowest score
    assert (
        tuned_model.match_rating_generator.performance_predictor.max_predict_value
        == 0.4
    )
    assert tuned_model.match_rating_generator.confidence_weight == 0.4

    # ratings should be reset
    assert tuned_model.player_ratings == {}
    assert tuned_model.team_ratings == []


def test_update_rating_generator_tuner_start_rating():
    team_rating_search = [
        ParameterSearchRange(
            name="confidence_days_ago_multiplier",
            type="uniform",
            low=0.02,
            high=0.12,
        )
    ]

    start_rating_rating_search = [
        ParameterSearchRange(
            name="league_quantile",
            type="uniform",
            low=0.02,
            high=0.12,
        )
    ]
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )

    rating_generator1 = UpdateRatingGenerator(
        performance_column="won",
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0.5,
            start_rating_generator=StartRatingGenerator(team_weight=0.5),
        ),
    )
    rating_generator2 = UpdateRatingGenerator(
        performance_column="won",
        match_rating_generator=MatchRatingGenerator(
            confidence_weight=0.4,
            start_rating_generator=StartRatingGenerator(team_weight=0.4),
        ),
        prefix="rating_2",
    )
    rating_generators = [rating_generator1, rating_generator2]

    match_predictor_factory = PipelineFactory(
        rating_generators=rating_generators,
        predictor=SklearnPredictor(
            estimator_features=[
                f"{RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED}0",
                f"{RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED}1",
            ],
            target="__target",
        ),
        column_names=column_names,
    )

    rating_generator_tuner = UpdateRatingGeneratorTuner(
        start_rating_n_trials=2,
        team_rating_n_trials=2,
        start_rating_search_ranges=start_rating_rating_search,
        team_rating_search_ranges=team_rating_search,
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1],
        }
    )

    matches = convert_df_to_matches(
        df=df, column_names=column_names, performance_column_name="won"
    )

    cross_validator = mock.Mock()
    cross_validator.cross_validation_score.side_effect = [0.5, 0.3, 0.5, 0.3]

    tuned_model = rating_generator_tuner.tune(
        pipeline_factory=copy.deepcopy(match_predictor_factory),
        rating_idx=1,
        matches=matches,
        df=df,
        cross_validator=cross_validator,
    )

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    # Second model has lowest score so it should be equal to confidence weight of that model and team_weight
    assert tuned_model.match_rating_generator.start_rating_generator.team_weight == 0.4
    assert tuned_model.match_rating_generator.confidence_weight == 0.4

    # ratings should be reset
    assert tuned_model.player_ratings == {}
    assert tuned_model.team_ratings == []
