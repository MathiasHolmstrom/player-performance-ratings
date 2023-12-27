import copy
from unittest import mock

import pandas as pd

from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import OpponentAdjustedRatingGenerator
from player_performance_ratings.transformation.pre_transformers import ColumnsWeighter
from player_performance_ratings.tuner import TransformerTuner
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.utils import ParameterSearchRange


def test_transformer_tuner():
    column_weigher_search_range = [
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
        ),
    ]

    columns_weighter = ColumnsWeighter(weighted_column_name="performance", column_weights=[])

    pre_transformer_search_ranges = [
        (columns_weighter, column_weigher_search_range),
    ]

    rating_generator1 = OpponentAdjustedRatingGenerator()

    rating_generators = [rating_generator1]

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="performance"
    )

    match_predictor_factory = MatchPredictorFactory(
        rating_generators=rating_generators,
        column_names=column_names,
    )

    transformer_tuner = TransformerTuner(pre_or_post="pre_rating", transformer_search_ranges=pre_transformer_search_ranges, n_trials=2)

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

    tuned_model = transformer_tuner.tune(match_predictor_factory=copy.deepcopy(match_predictor_factory),
                                             df=df, scorer=scorer)

    # tests immutability of match_predictor_factory
    assert match_predictor_factory.rating_generators == rating_generators

    #assert best model belongs in search range
    assert tuned_model[0].column_weights[0].weight >= 0.1
    assert tuned_model[0].column_weights[0].weight <= 0.3

    assert tuned_model[0].column_weights[1].weight >= 0.25
    assert tuned_model[0].column_weights[1].weight <= 0.85


