import pandas as pd
import pytest

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformation import LagTransformer, RollingMeanTransformer
from player_performance_ratings.transformation.post_transformers import LagLowerGranularityTransformer, \
    NormalizerTransformer, GameTeamMembersColumnsTransformer


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game",
        team_id="team",
        player_id="player",
        start_date="start_date",
        performance="performance"
    )

def test_game_team_members_column_transformer(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'c', "d"],
            "team": [1, 1, 2, 2],
            'game': [1, 1, 1, 1],
            "minutes": [15, 20, 16, 20],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01")]
        }
    )
    expected_df = df.copy()
    transformer = GameTeamMembersColumnsTransformer(column_names=column_names, features=["minutes"], sort_by=["minutes"], players_per_team_per_match_count=2)
    df = transformer.fit_transform(df)
    expected_df["team_player1_minutes"] = [20, 15, 20, 16]
    pd.testing.assert_frame_equal(df, expected_df, check_like=True)



def test_normalizer_transformer(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a', "b", 'a', 'b', 'a', "b"],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
            'game': [1, 1, 1, 1, 2, 2, 2, 2],
            "minutes": [15, 20, 16, 20, 20, 25, 20, 25],
        }
    )

    expected_df = df.copy()
    transformer = NormalizerTransformer(features=["minutes"], granularity=[column_names.match_id, column_names.team_id],
                                        create_target_as_mean=True)
    df = transformer.fit_transform(df)
    mean_minutes = df['minutes'].mean()
    game1_team_1_multiplier = mean_minutes / (15 * 0.5 + 20 * 0.5)
    game1_team_2_multiplier = mean_minutes / (16 * 0.5 + 20 * 0.5)
    game2_team_1_multiplier = mean_minutes / (20 * 0.5 + 25 * 0.5)
    game2_team_2_multiplier = mean_minutes / (20 * 0.5 + 25 * 0.5)

    expected_df["minutes"] = [15 * game1_team_1_multiplier, 20 * game1_team_1_multiplier, 16 * game1_team_2_multiplier,
                              20 * game1_team_2_multiplier,
                              20 * game2_team_1_multiplier, 25 * game2_team_1_multiplier, 20 * game2_team_2_multiplier,
                              25 * game2_team_2_multiplier]

    pd.testing.assert_frame_equal(df, expected_df, check_like=True)


def test_lag_fit_transform(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            "team": [1, 2, 1],
            'game': [1, 1, 2],
            'points': [1, 2, 3],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02")]
        }
    )
    original_df = df.copy()

    lag_transformation = LagTransformer(
        features=['points'],
        lag_length=1,
        granularity=['player'],
        column_names=column_names,
    )

    df_with_lags = lag_transformation.fit_transform(df)

    expected_df = original_df.assign(**{
        "lag_1_points": [None, None, 1]
    })

    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True)


def test_lag_fit_transform_2_features(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [1, 1, 2],
            "team": [1, 2, 1],
            'points': [1, 2, 3],
            'points_per_minute': [0.5, 1, 1.5],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02")]
        }
    )
    original_df = df.copy()

    lag_transformation = LagTransformer(
        features=['points', "points_per_minute"],
        lag_length=1,
        granularity=['player'],
        column_names=column_names
    )

    df_with_lags = lag_transformation.fit_transform(df)

    expected_df = original_df.assign(**{
        "lag_1_points": [None, None, 1],
        "lag_1_points_per_minute": [None, None, 0.5]
    })

    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True)


def test_lag_fit_transform_lag_length_2(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a', "a"],
            'game': [1, 1, 2, 3],
            'points': [1, 2, 3, 4],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-03")],
            "team": [1, 2, 1, 1],
        }
    )
    original_df = df.copy()

    lag_transformation = LagTransformer(
        features=['points'],
        lag_length=2,
        granularity=['player'],
        column_names=column_names
    )

    df_with_lags = lag_transformation.fit_transform(df)

    expected_df = original_df.assign(**{
        "lag_1_points": [None, None, 1, 3],
        "lag_2_points": [None, None, None, 1]
    })

    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True)


def test_lag_fit_transform_and_transform(column_names):
    historical_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [1, 1, 2],
            'points': [1, 2, 3],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02")],
            "team": [1, 2, 1],
        }
    )

    future_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [3, 3, 4],
            "start_date": [pd.to_datetime("2023-01-04"), pd.to_datetime("2023-01-04"), pd.to_datetime("2023-01-05")],
            "team": [1, 2, 1],
        }
    )
    future_df_copy = future_df.copy()

    lag_transformation = LagTransformer(
        features=['points'],
        lag_length=1,
        granularity=['player'],
        column_names=column_names

    )

    _ = lag_transformation.fit_transform(historical_df)
    future_transformed_df = lag_transformation.transform(future_df)

    expected_df = future_df_copy.assign(**{lag_transformation.prefix + "1_points": [3, 2, None]})

    pd.testing.assert_frame_equal(future_transformed_df, expected_df, check_like=True)


def test_lag_lower_granularity_transform(column_names):
    game_player_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [1, 1, 2],
            'points': [1, 2, 3],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-03")],
            "team": [1, 2, 1],
        }
    )

    lower_granularity_df = pd.DataFrame(
        {
            'player': ['a', "a", 'b', "a"],
            'points': [1, 2, 2, 1],
            "game": [1, 1, 1, 2],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-03")],
            "team": [1, 1, 2, 1],
        }
    )

    transformer = LagLowerGranularityTransformer(
        features=['points'],
        lag_length=1,
        granularity=['player'],
        column_names=column_names,
    )

    with pytest.raises(ValueError):
        transformer.transform(diff_granularity_df=lower_granularity_df, game_player_df=game_player_df)


def test_lag_lower_granularity_fit_transform(column_names):
    game_player_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [1, 1, 2],
            'points': [1, 2, 3],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-03")],
            "team": [1, 2, 1],
        }
    )

    lower_granularity_df = pd.DataFrame(
        {
            'player': ['a', "a", 'b', "a"],
            'points': [1, 2, 2, 1],
            "game": [1, 1, 1, 2],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-03")],
            "team": [1, 1, 2, 1],
        }
    )
    original_game_player_df = game_player_df.copy()

    transformer = LagLowerGranularityTransformer(
        features=['points'],
        lag_length=1,
        granularity=['player'],
        column_names=column_names,
    )

    df_with_lags = transformer.fit_transform(diff_granularity_df=lower_granularity_df, game_player_df=game_player_df)

    expected_df = original_game_player_df.assign(**{
        f"{transformer.prefix}1_points": [None, None, 1.5]
    })
    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True)


def test_lag_lower_granularity_fit_transform_and_transform(column_names):
    historical_game_player_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [1, 1, 2],
            'points': [1, 2, 3],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-03")],
            "team": [1, 2, 1],
        }
    )

    historical_lower_granularity_df = pd.DataFrame(
        {
            'player': ['a', "a", 'b', "a"],
            'points': [1, 2, 2, 1],
            "game": [1, 1, 1, 2],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-03")],
            "team": [1, 1, 2, 1],
        }
    )

    future_game_player_df = pd.DataFrame(
        {
            'player': ['a', "c"],
            'game': [4, 4],
            "start_date": [pd.to_datetime("2023-01-05"), pd.to_datetime("2023-01-06")],
            "team": [1, 2],
        }
    )

    future_lower_granularity_df = pd.DataFrame(
        {
            'player': ['a', "c"],
            'points': [1, 2],
            "game": [4, 4],
            "start_date": [pd.to_datetime("2023-01-05"), pd.to_datetime("2023-01-06")],
            "team": [1, 2],
        }
    )

    original_future_game_player_df = future_game_player_df.copy()

    transformer = LagLowerGranularityTransformer(
        features=['points'],
        lag_length=1,
        granularity=['player'],
        column_names=column_names,
    )

    _ = transformer.fit_transform(diff_granularity_df=historical_lower_granularity_df,
                                  game_player_df=historical_game_player_df)
    transformed_game_player_df = transformer.transform(diff_granularity_df=future_lower_granularity_df,
                                                       game_player_df=future_game_player_df)

    expected_df = original_future_game_player_df.assign(**{
        f"{transformer.prefix}1_points": [1, None]
    })
    pd.testing.assert_frame_equal(transformed_game_player_df, expected_df, check_like=True, check_dtype=False)


def test_lag_lower_granularity_with_weights_fit_transform(column_names):
    lower_granularity_df = pd.DataFrame(
        {
            'player': ['a', "a", 'b', 'a'],
            'points': [1, 2, 2, 1],
            "game": [1, 1, 1, 2],
            "weight": [0.9, 0.1, 1, 1],
            'team': [1, 1, 2, 1],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-02")],
        }
    )

    game_player_df = pd.DataFrame({
        "player": ['a', 'b', 'a'],
        "game": [1, 1, 2],
        "team": [1, 2, 1],
        "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02")]
    })

    original_game_player_df = game_player_df.copy()

    lag_transformation = LagLowerGranularityTransformer(
        features=['points'],
        lag_length=1,
        granularity=['player'],
        column_names=column_names,
        weight_column='weight'
    )

    df_with_lags = lag_transformation.fit_transform(diff_granularity_df=lower_granularity_df,
                                                    game_player_df=game_player_df)

    expected_df = original_game_player_df.assign(**{
        f"{lag_transformation.prefix}1_points": [None, None, 0.9 * 1 + 2 * 0.1]
    })

    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True, check_dtype=False)


def test_lag_transformation_transform_2_lags(column_names):
    historical_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [1, 1, 2],
            'points': [1, 2, 3],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02")],
            "team": [1, 2, 1],
        }
    )

    future_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [3, 3, 4],
            "start_date": [pd.to_datetime("2023-01-04"), pd.to_datetime("2023-01-04"), pd.to_datetime("2023-01-05")],
            "team": [1, 2, 1],
        }
    )
    future_df_copy = future_df.copy()

    lag_transformation = LagTransformer(
        features=['points'],
        lag_length=2,
        granularity=['player'],
        column_names=column_names
    )

    _ = lag_transformation.fit_transform(historical_df)
    future_transformed_df = lag_transformation.transform(future_df)

    expected_df = future_df_copy.assign(**{lag_transformation.prefix + "1_points": [3, 2, None]})
    expected_df = expected_df.assign(**{lag_transformation.prefix + "2_points": [1, None, 3]})
    pd.testing.assert_frame_equal(future_transformed_df, expected_df, check_like=True)


def test_rolling_mean_fit_transform(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a', "a"],
            "game": [1, 1, 2, 3],
            'points': [1, 2, 3, 2],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-04")],
            "team": [1, 2, 1, 1],
        }
    )
    original_df = df.copy()

    rolling_mean_transformation = RollingMeanTransformer(
        features=['points'],
        window=2,
        min_periods=1,
        granularity=['player'],
        column_names=column_names
    )

    df_with_rolling_mean = rolling_mean_transformation.fit_transform(df)

    expected_df = original_df.assign(**{
        f"{rolling_mean_transformation.prefix}2_points": [None, None, 1, (3 + 1) / 2]
    })

    pd.testing.assert_frame_equal(df_with_rolling_mean, expected_df, check_like=True)


def test_rolling_mean_fit_transform_and_transform(column_names):
    historical_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a', "a"],
            "game": [1, 1, 2, 3],
            'points': [1, 2, 3, 2],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-04")],
            "team": [1, 2, 1, 1],
        }
    )

    future_df = pd.DataFrame({
        "player": ['a', 'b', 'a'],
        "game": [4, 4, 5],
        "start_date": [pd.to_datetime("2023-01-05"), pd.to_datetime("2023-01-05"), pd.to_datetime("2023-01-06")],
        "team": [1, 2, 1],
    })

    original_future_df = future_df.copy()
    rolling_mean_transformation = RollingMeanTransformer(
        features=['points'],
        window=2,
        min_periods=1,
        granularity=['player'],
        column_names=column_names
    )

    _ = rolling_mean_transformation.fit_transform(historical_df)
    transformed_future_df = rolling_mean_transformation.transform(future_df)

    expected_df = original_future_df.assign(**{
        f"{rolling_mean_transformation.prefix}2_points": [2.5, 2, 2]
    })

    pd.testing.assert_frame_equal(transformed_future_df, expected_df, check_like=True)
