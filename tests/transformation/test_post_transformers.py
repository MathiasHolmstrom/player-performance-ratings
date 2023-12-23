import pandas as pd

from player_performance_ratings.transformation import LagTransformation, RollingMeanTransformation


def test_lag_transformation():

    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a'],
            'game': [1, 1, 2],
            'points': [1, 2, 3]
        }
    )
    original_df = df.copy()

    lag_transformation = LagTransformation(
        feature_names=['points'],
        lag_length=1,
        granularity=['player']
    )

    df_with_lags = lag_transformation.transform(df)

    expected_df = original_df.assign(**{
        "lag_1_points":  [None, None, 1]
    })

    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True)


def test_lag_transformation_lag_length_2():
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a', "a"],
            'game': [1, 1, 2, 3],
            'points': [1, 2, 3, 4]
        }
    )
    original_df = df.copy()

    lag_transformation = LagTransformation(
        feature_names=['points'],
        lag_length=2,
        granularity=['player']
    )

    df_with_lags = lag_transformation.transform(df)

    expected_df = original_df.assign(**{
        "lag_1_points": [None, None, 1, 3],
        "lag_2_points": [None, None, None, 1]
    })

    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True)



def test_lag_transformation_with_game_id():
    df = pd.DataFrame(
        {
            'player': ['a',"a", 'b', 'a'],
            'points': [1, 2, 2, 1],
            "game": [1, 1, 1, 2]
        }
    )
    original_df = df.copy()

    lag_transformation = LagTransformation(
        feature_names=['points'],
        lag_length=1,
        granularity=['player'],
        game_id='game'
    )

    df_with_lags = lag_transformation.transform(df)

    expected_df = original_df.assign(**{
        "lag_1_points": [None, None, None, 1.5]
    })

    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True)


def test_lag_transformation_with_game_id_and_weights():
    df = pd.DataFrame(
        {
            'player': ['a',"a", 'b', 'a'],
            'points': [1, 2, 2, 1],
            "game": [1, 1, 1, 2],
            "weight": [0.9, 0.1, 1, 1]
        }
    )
    original_df = df.copy()

    lag_transformation = LagTransformation(
        feature_names=['points'],
        lag_length=1,
        granularity=['player'],
        game_id='game',
        weight_column='weight'
    )

    df_with_lags = lag_transformation.transform(df)

    expected_df = original_df.assign(**{
        "lag_1_points": [None, None, None, 0.9*1 + 2*0.1]
    })

    pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True)


def test_rolling_mean_transformation():
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a', "a", "a"],
            'points': [1, 2, 3, 4, 4]
        }
    )
    original_df = df.copy()

    rolling_mean_transformation = RollingMeanTransformation(
        feature_names=['points'],
        window=2,
        min_periods=1,
        granularity=['player']
    )

    df_with_rolling_mean = rolling_mean_transformation.transform(df)

    expected_df = original_df.assign(**{
        "rolling_mean_2_points":  [None, None, 1, 2, (4+3)/2]
    })

    pd.testing.assert_frame_equal(df_with_rolling_mean, expected_df, check_like=True)

def test_rolling_mean_transformation_with_game_id():
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a', "a", "a"],
            'points': [1, 2, 3, 4, 4],
            "game": [1, 1, 1, 2, 2]
        }
    )
    original_df = df.copy()

    rolling_mean_transformation = RollingMeanTransformation(
        feature_names=['points'],
        window=2,
        min_periods=1,
        granularity=['player'],
        game_id='game'
    )

    df_with_rolling_mean = rolling_mean_transformation.transform(df)

    expected_df = original_df.assign(**{
        "rolling_mean_2_points": [None, None, None, 2, 2]
    })

    pd.testing.assert_frame_equal(df_with_rolling_mean, expected_df, check_like=True)


def test_rolling_mean_transformation_with_game_id_and_weights():
    df = pd.DataFrame(
        {
            'player': ['a', 'b', 'a', "a", "a"],
            'points': [1, 2, 3, 4, 4],
            "game": [1, 1, 1, 2, 2],
            "weight": [0.9, 1, 0.1, 1, 1]
        }
    )
    original_df = df.copy()

    rolling_mean_transformation = RollingMeanTransformation(
        feature_names=['points'],
        window=2,
        min_periods=1,
        granularity=['player'],
        game_id='game',
        weight_column='weight'
    )

    df_with_rolling_mean = rolling_mean_transformation.transform(df)

    expected_df = original_df.assign(**{
        "rolling_mean_2_points": [None, None, None, 0.9*1+3*0.1, 0.9*1+3*0.1]
    })

    pd.testing.assert_frame_equal(df_with_rolling_mean, expected_df, check_like=True)

