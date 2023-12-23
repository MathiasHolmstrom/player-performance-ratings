import copy

import mock
import pandas as pd
from player_performance_ratings.predictor.estimators import SKLearnClassifierWrapper

from player_performance_ratings import ColumnNames, PredictColumnNames
from player_performance_ratings.ratings import TeamRatingGenerator, RatingColumnNames
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.ratings.opponent_adjusted_rating.performance_predictor import RatingDifferencePerformancePredictor
from player_performance_ratings.ratings.opponent_adjusted_rating.start_rating_generator import StartRatingGenerator
from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import OpponentAdjustedRatingGenerator
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.rating_generator_tuner import OpponentAdjustedRatingGeneratorTuner
from player_performance_ratings.tuner.utils import ParameterSearchRange


def test_opponent_adjusted_rating_generator_tuner_team_rating():
    team_rating_search = [
        ParameterSearchRange(
            name='confidence_days_ago_multiplier',
            type='uniform',
            low=0.02,
            high=.12,
        )
    ]

    rating_generator1 = OpponentAdjustedRatingGenerator(team_rating_generator=TeamRatingGenerator(
        confidence_weight=0.5
    ))
    rating_generator2 = OpponentAdjustedRatingGenerator(team_rating_generator=TeamRatingGenerator(
        confidence_weight=0.4
    ))
    rating_generators = [rating_generator1, rating_generator2]

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="won"
    )

    match_predictor_factory = MatchPredictorFactory(
        rating_generators=rating_generators,
        column_names=column_names,
        predictor=SKLearnClassifierWrapper(
            features=[f"{RatingColumnNames.RATING_DIFFERENCE}0", f"{RatingColumnNames.RATING_DIFFERENCE}1"],
            target=PredictColumnNames.TARGET
        )
     #   predictor = mock.Mock()
    )

    rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
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
            "__target": [1, 0, 0, 1]
        }
    )

    matches = convert_df_to_matches(df=df, column_names=column_names)

    scorer = mock.Mock()
    scorer.score.side_effect = [0.5, 0.3]

    tuned_model = rating_generator_tuner.tune(match_predictor_factory=copy.deepcopy(match_predictor_factory),
                                              rating_idx=1,
                                              matches=matches, df=df, scorer=scorer)

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    # Second model has lowest score so it should be equal to confidence weight of that model
    assert tuned_model.team_rating_generator.confidence_weight == 0.4

    # ratings should be reset
    assert tuned_model.player_ratings == {}
    assert tuned_model.team_ratings == []


def test_opponent_adjusted_rating_generator_tuner_performance_predictor():
    team_rating_search = [
        ParameterSearchRange(
            name='confidence_days_ago_multiplier',
            type='uniform',
            low=0.02,
            high=.12,
        )
    ]

    rating_generator1 = OpponentAdjustedRatingGenerator(team_rating_generator=TeamRatingGenerator(
        confidence_weight=0.5,
        performance_predictor=RatingDifferencePerformancePredictor(
            max_predict_value=0.3
        )
    ))
    rating_generator2 = OpponentAdjustedRatingGenerator(team_rating_generator=TeamRatingGenerator(
        confidence_weight=0.4,
        performance_predictor=RatingDifferencePerformancePredictor(
            max_predict_value=0.4
        )
    ))
    rating_generators = [rating_generator1, rating_generator2]

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="won"
    )

    match_predictor_factory = MatchPredictorFactory(
        rating_generators=rating_generators,
        column_names=column_names,
        predictor=SKLearnClassifierWrapper(
            features=[f"{RatingColumnNames.RATING_DIFFERENCE}0", f"{RatingColumnNames.RATING_DIFFERENCE}1"],
            target=PredictColumnNames.TARGET
        )
    )

    rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
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
            "__target": [1, 0, 0, 1]
        }
    )

    matches = convert_df_to_matches(df=df, column_names=column_names)

    scorer = mock.Mock()
    scorer.score.side_effect = [0.5, 0.3]

    tuned_model = rating_generator_tuner.tune(match_predictor_factory=copy.deepcopy(match_predictor_factory),
                                              rating_idx=1,
                                              matches=matches, df=df, scorer=scorer)

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    # Second model has lowest score
    assert tuned_model.team_rating_generator.performance_predictor.max_predict_value == 0.4
    assert tuned_model.team_rating_generator.confidence_weight == 0.4

    # ratings should be reset
    assert tuned_model.player_ratings == {}
    assert tuned_model.team_ratings == []


def test_opponent_adjusted_rating_generator_tuner_start_rating():
    team_rating_search = [
        ParameterSearchRange(
            name='confidence_days_ago_multiplier',
            type='uniform',
            low=0.02,
            high=.12,
        )
    ]

    start_rating_rating_search = [
        ParameterSearchRange(
            name='league_quantile',
            type='uniform',
            low=0.02,
            high=.12,
        )
    ]

    rating_generator1 = OpponentAdjustedRatingGenerator(team_rating_generator=TeamRatingGenerator(
        confidence_weight=0.5,
        start_rating_generator=StartRatingGenerator(
            team_weight=0.5
        )
    ))
    rating_generator2 = OpponentAdjustedRatingGenerator(team_rating_generator=TeamRatingGenerator(
        confidence_weight=0.4,
        start_rating_generator=StartRatingGenerator(
            team_weight=0.4
        )
    ))
    rating_generators = [rating_generator1, rating_generator2]

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="won"
    )

    match_predictor_factory = MatchPredictorFactory(
        rating_generators=rating_generators,
        column_names=column_names,
        predictor=SKLearnClassifierWrapper(
            features=[f"{RatingColumnNames.RATING_DIFFERENCE}0", f"{RatingColumnNames.RATING_DIFFERENCE}1"],
            target=PredictColumnNames.TARGET
        )
    )

    rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
        start_rating_n_trials=2,
        team_rating_n_trials=2,
        start_rating_search_ranges=start_rating_rating_search,
        team_rating_search_ranges=team_rating_search
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "won": [1, 0, 0, 1],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "__target": [1, 0, 0, 1]
        }
    )

    matches = convert_df_to_matches(df=df, column_names=column_names)

    scorer = mock.Mock()
    scorer.score.side_effect = [0.5, 0.3, 0.5, 0.3]

    tuned_model = rating_generator_tuner.tune(match_predictor_factory=copy.deepcopy(match_predictor_factory),
                                              rating_idx=1,
                                              matches=matches, df=df, scorer=scorer)

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    # Second model has lowest score so it should be equal to confidence weight of that model and team_weight
    assert tuned_model.team_rating_generator.start_rating_generator.team_weight == 0.4
    assert tuned_model.team_rating_generator.confidence_weight == 0.4

    # ratings should be reset
    assert tuned_model.player_ratings == {}
    assert tuned_model.team_ratings == []
