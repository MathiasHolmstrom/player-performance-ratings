import pandas as pd
import pytest

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformation import LagTransformer, RollingMeanTransformer
from player_performance_ratings.transformation.post_transformers import NormalizerTransformer, \
    RollingMeanDaysTransformer, BinaryOutcomeRollingMeanTransformer


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game",
        team_id="team",
        player_id="player",
        start_date="start_date",
        performance="performance"
    )


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


def test_lag_team_fit_transform(column_names):
    "Should calculate average point of prior game"

    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
            'game': [1, 1, 1, 1, 2, 2, 2, 2],
            'points': [1, 2, 3, 2, 4, 5, 6, 7],
            "start_date": [
                pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02")
            ]
        }
    )
    original_df = df.copy()

    lag_transformation = LagTransformer(
        features=['points'],
        lag_length=1,
        granularity=['team'],
        column_names=column_names,
    )

    df_with_lags = lag_transformation.fit_transform(df)

    expected_df = original_df.assign(**{
        "lag_1_points": [None, None, None, None, 1.5, 1.5, 2.5, 2.5]
    })
    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True, check_dtype=False)


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
    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
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
    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')

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
    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
    pd.testing.assert_frame_equal(future_transformed_df, expected_df, check_like=True)


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
    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
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
    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')

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
    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
    pd.testing.assert_frame_equal(transformed_future_df, expected_df, check_like=True)


def test_rolling_mean_transformer_fit_transformer_team_stat(column_names):
    historical_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'c', "d", 'a', 'b', 'c', "d"],
            "game": [1, 1, 1, 1, 2, 2, 2, 2],
            'score_difference': [10, 10, -10, -10, 15, 15, -15, -15],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02")],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )

    expected_df = historical_df.copy()

    rolling_mean_transformation = RollingMeanTransformer(
        features=['score_difference'],
        window=2,
        min_periods=1,
        granularity=['team'],
        column_names=column_names
    )

    transformed_data = rolling_mean_transformation.fit_transform(historical_df)
    expected_df[rolling_mean_transformation.prefix + "2_score_difference"] = [None, None, None, None, 10, 10, -10, -10]
    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
    pd.testing.assert_frame_equal(transformed_data, expected_df, check_like=True, check_dtype=False)


def test_rolling_mean_days_fit_transform(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'a', 'b', 'a', "a"],
            "game": [1, 2, 2, 3, 4],
            'points': [1, 1, 2, 3, 2],
            'points2': [2, 2, 4, 6, 4],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-10")],
            "team": [1, 1, 2, 1, 1],
        }
    )

    original_df = df.copy()

    rolling_mean_transformation = RollingMeanDaysTransformer(
        features=['points', 'points2'],
        days=2,
        granularity=['player'],
        column_names=column_names,
        add_count=True
    )

    transformed_df = rolling_mean_transformation.fit_transform(df)

    expected_df = original_df.assign(**{
        rolling_mean_transformation.features_out[0]: [None, None, None, 1, None],
        rolling_mean_transformation.features_out[1]: [None, None, None, 2, None],
        rolling_mean_transformation.features_out[2]: [0, 0, 0, 2, 0],
    })

    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
    pd.testing.assert_frame_equal(transformed_df, expected_df, check_like=True, check_dtype=False)


def test_rolling_mean_days_fit_transform_40_days(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'a', 'a', 'b', "a", "b"],
            "game": [1, 2, 3, 4, 5, 6],
            'points': [1, 1.5, 2, 3, 2, 4],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-10"),
                           pd.to_datetime("2023-01-10"),
                           pd.to_datetime("2023-02-15")],
            "team": [1, 1, 1, 2, 1, 2],
        }
    )

    original_df = df.copy()

    rolling_mean_transformation = RollingMeanDaysTransformer(
        features=['points'],
        days=40,
        granularity=['player'],
        column_names=column_names,
    )

    transformed_df = rolling_mean_transformation.fit_transform(df)

    expected_df = original_df.assign(**{
        rolling_mean_transformation.features_out[0]: [None, 1, 1, None, 1.5, 3],
    })

    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
    pd.testing.assert_frame_equal(transformed_df, expected_df, check_like=True, check_dtype=False)


def test_rolling_mean_days_fit_transform_opponent(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'b', "c", "d", 'a', 'b', "c", "d"],
            "game": [1, 1, 1, 1, 2, 2, 2, 2],
            'points': [1, 1.5, 2, 3, 2, 4, 1, 2],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-02"), ],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )

    original_df = df.copy()

    rolling_mean_transformation = RollingMeanDaysTransformer(
        features=['points'],
        days=10,
        granularity=['player'],
        column_names=column_names,
        add_opponent=True
    )

    transformed_df = rolling_mean_transformation.fit_transform(df)

    expected_df = original_df.assign(**{
        rolling_mean_transformation.features_out[0]: [None, None, None, None, 1, 1.5, 2, 3],
        rolling_mean_transformation.features_out[1]: [None, None, None, None, 2.5, 2.5, 1.25, 1.25],
    })

    expected_df['team'] = expected_df['team'].astype('str')
    expected_df['game'] = expected_df['game'].astype('str')
    expected_df['player'] = expected_df['player'].astype('str')
    pd.testing.assert_frame_equal(transformed_df, expected_df, check_like=True, check_dtype=False)


def test_binary_granularity_rolling_mean_transformer(column_names):
    historical_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'c', "d", 'a', 'b', 'c', "d", "c", "d"],
            "game": [1, 1, 1, 1, 2, 2, 2, 3, 4, 4],
            'score_difference': [10, 10, -10, -10, 15, 15, -15, -20, 2, 2],
            "won": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02")],
            "team": [1, 1, 2, 2, 1, 1, 2, 2, 2, 2],
            'prob': [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.3, 0.3, 0.2, 0.2]
        }
    )

    historical_df['team'] = historical_df['team'].astype('str')
    historical_df['game'] = historical_df['game'].astype('str')
    historical_df['player'] = historical_df['player'].astype('str')

    expected_df = historical_df.copy()

    rolling_mean_transformation = BinaryOutcomeRollingMeanTransformer(
        features=['score_difference'],
        binary_column="won",
        window=10,
        min_periods=1,
        granularity=['player'],
        column_names=column_names,
        prob_column="prob"
    )

    transformed_data = rolling_mean_transformation.fit_transform(historical_df)
    expected_df[rolling_mean_transformation.features_out[0]] = [None, None, None, None, 10, 10, None, None, None, None]
    expected_df[rolling_mean_transformation.features_out[1]] = [None, None, None, None, None, None, -10,
                                                                -10, -12.5, -15]
    expected_df[rolling_mean_transformation.features_out[2]] = [None, None, None, None, 10 * 0.6, 10 * 0.6, None, None,
                                                                None, None]
    expected_df[rolling_mean_transformation.features_out[3]] = [None, None, None, None, None, None, (1 - 0.3) * -10,
                                                                (1 - 0.3) * -10, (1 - 0.2) * -12.5, (1 - 0.2) * -15]
    pd.testing.assert_frame_equal(transformed_data, expected_df, check_like=True, check_dtype=False)


def test_binary_granularity_rolling_mean_fit_transform_transform(column_names):
    historical_df = pd.DataFrame(
        {
            'player': ['a', 'b', 'c', "d", 'a', 'b', 'c', "d", "c", "d"],
            "game": [1, 1, 1, 1, 2, 2, 2, 3, 4, 4],
            'score_difference': [10, 10, -10, -10, 15, 15, -15, -20, 2, 2],
            "won": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02"),
                           pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02")],
            "team": [1, 1, 2, 2, 1, 1, 2, 2, 2, 2],
        }
    )

    historical_df['team'] = historical_df['team'].astype('str')
    historical_df['game'] = historical_df['game'].astype('str')
    historical_df['player'] = historical_df['player'].astype('str')
    expected_historical_df = historical_df.copy()

    rolling_mean_transformation = BinaryOutcomeRollingMeanTransformer(
        features=['score_difference'],
        binary_column="won",
        window=1,
        min_periods=1,
        granularity=['player'],
        column_names=column_names
    )

    historical_df = rolling_mean_transformation.fit_transform(historical_df)
    expected_historical_df[rolling_mean_transformation.features_out[0]] = [None, None, None, None, 10, 10, None, None,
                                                                           None, None]
    expected_historical_df[rolling_mean_transformation.features_out[1]] = [None, None, None, None, None, None, -10,
                                                                           -10, -15, -20]
    pd.testing.assert_frame_equal(historical_df, expected_historical_df, check_like=True, check_dtype=False)

    future_df = pd.DataFrame(
        {
            'player': ['a', 'b'],
            "game": [5, 5],
            'score_difference': [None, None],
            "won": [None, None],
            "start_date": [pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-02")],
            "team": [1, 1],
        }
    )

    future_df['team'] = future_df['team'].astype('str')
    future_df['game'] = future_df['game'].astype('str')
    future_df['player'] = future_df['player'].astype('str')

    expected_future_df = future_df.copy()

    future_df = rolling_mean_transformation.transform(future_df)
    expected_future_df[rolling_mean_transformation.features_out[0]] = [15, 15]
    expected_future_df[rolling_mean_transformation.features_out[1]] = [None, None]

    pd.testing.assert_frame_equal(future_df, expected_future_df, check_like=True, check_dtype=False)


def test_binary_granularity_rolling_mean_fit_transform_opponent(column_names):
    df = pd.DataFrame(
        {
            'player': ['a', 'b', "a", "b", "a", "b"],
            "game": [1, 1, 2, 2, 3, 3],
            'score_difference': [10, -10, 5, -5, 3, 3],
            "won": [1, 0, 0, 1, 1,0],
            "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),
                           pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"),],
            "team": [1, 2, 1, 2, 1, 2],
            "prob": [0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
        }
    )

    df['team'] = df['team'].astype('str')
    df['game'] = df['game'].astype('str')
    df['player'] = df['player'].astype('str')
    expected_historical_df = df.copy()

    rolling_mean_transformation = BinaryOutcomeRollingMeanTransformer(
        features=['score_difference'],
        binary_column="won",
        window=2,
        min_periods=1,
        granularity=['player'],
        add_opponent=True,
        column_names=column_names,
        prob_column="prob"
    )

    df = rolling_mean_transformation.fit_transform(df)

    expected_historical_df[rolling_mean_transformation.features_out[0]] = [None, None, 10, None, 10, -5]
    expected_historical_df[rolling_mean_transformation.features_out[1]] = [None, None, None, -10, 5, -10 ]
    expected_historical_df[rolling_mean_transformation.features_out[2]] = [None, None, None, 10, -5, 10 ]
    expected_historical_df[rolling_mean_transformation.features_out[3]] = [None, None, -10, None, -10, 5]

    expected_historical_df[rolling_mean_transformation.features_out[4]] = [None, None, None, None, 10*0.6+0.4*5, -10*0.6+0.4*-5]
    expected_historical_df[rolling_mean_transformation.features_out[5]] = [None, None, None, None, -10*0.6+0.4*-5, 10*0.6+0.4*5]

    pd.testing.assert_frame_equal(df, expected_historical_df, check_like=True, check_dtype=False)



