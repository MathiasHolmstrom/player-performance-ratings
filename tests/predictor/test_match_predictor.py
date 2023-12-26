import mock
import pandas as pd
from player_performance_ratings.transformation import ColumnWeight

from player_performance_ratings import ColumnNames
from player_performance_ratings.predictor import MatchPredictor


def test_match_predictor_auto_pre_transformers():
    df = pd.DataFrame({
        "game_id": [1, 1, 2, 2, 3, 3],
        "player_id": [1, 2, 3, 1, 2, 3],
        "team_id": [1, 1, 2, 2, 3, 3],
        "start_date": [1, 1, 2, 2, 3, 3],
        'deaths': [1, 1, 1, 2, 2, 2],
        "kills": [0.2, 0.3, 0.4, 0.5, 2, 0.2],
        "__target": [1, 0, 1, 0, 1, 0],
    })

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True)
    ]

    expected_df = df.copy()
    expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.add_prediction.return_value  = expected_df

    match_predictor = MatchPredictor(
        train_split_date=2,
        auto_create_pre_transformers=True,
        column_weights=column_weights,
        rating_generators=[],
        predictor=predictor_mock,
        column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="weighted_performance"
        ),
    )

    new_df = match_predictor.generate_historical(df=df)

    pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    assert len(match_predictor.pre_rating_transformers) > 0
