import pandas as pd
import pytest
import polars as pl
from polars.testing import assert_frame_equal
from player_performance_ratings import ColumnNames
from player_performance_ratings.transformers import (
    LagTransformer,
    RollingMeanDaysTransformer,
    BinaryOutcomeRollingMeanTransformer,
)
from player_performance_ratings.transformers.lag_generators import (
    RollingMeanTransformerPolars,
)


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game",
        team_id="team",
        player_id="player",
        start_date="start_date",
    )


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_team_fit_transform(df, column_names):
    "Should calculate average point of prior game"

    data = df(
        {
            "player": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
            "game": [1, 1, 1, 1, 2, 2, 2, 2],
            "points": [1, 2, 3, 2, 4, 5, 6, 7],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
            ],
        }
    )

    try:
        original_df = data.copy()
    except:
        original_df = data.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=["team"],
    )

    df_with_lags = lag_transformation.generate_historical(data, column_names=column_names)
    if isinstance(data, pl.DataFrame):
        expected_df = original_df.with_columns([
            pl.Series("lag_1_points", [None, None, None, None, 1.5, 1.5, 2.5, 2.5]),
            pl.col("team").cast(pl.String),
            pl.col("game").cast(pl.String),
            pl.col("player").cast(pl.String)
        ]
        )


    elif isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{"lag_1_points": [None, None, None, None, 1.5, 1.5, 2.5, 2.5]}
        )
        expected_df["team"] = expected_df["team"].astype("str")
        expected_df["game"] = expected_df["game"].astype("str")
        expected_df["player"] = expected_df["player"].astype("str")
    expected_df = pl.DataFrame(expected_df).select(df_with_lags.columns)
    assert_frame_equal(
        df_with_lags, expected_df, check_dtypes=False
    )


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_fit_transform_2_features(df, column_names):
    data = df(
        {
            "player": ["a", "b", "a"],
            "game": [1, 1, 2],
            "team": [1, 2, 1],
            "points": [1, 2, 3],
            "points_per_minute": [0.5, 1, 1.5],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
            ],
        }
    )
    if isinstance(data, pl.DataFrame):
        original_df = data.clone()
    else:
        original_df = data.copy()

    lag_transformation = LagTransformer(
        features=["points", "points_per_minute"],
        lag_length=1,
        granularity=["player"],
    )

    df_with_lags = lag_transformation.generate_historical(data, column_names=column_names)
    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                "lag_1_points": [None, None, 1],
                "lag_1_points_per_minute": [None, None, 0.5],
            }
        )
        expected_df["team"] = expected_df["team"].astype("str")
        expected_df["game"] = expected_df["game"].astype("str")
        expected_df["player"] = expected_df["player"].astype("str")
        pd.testing.assert_frame_equal(
            df_with_lags, expected_df, check_like=True, check_dtype=False
        )

    else:
        expected_df = original_df.with_columns([
            pl.Series("lag_1_points", [None, None, 1]),
            pl.Series("lag_1_points_per_minute", [None, None, 0.5]),
            pl.col("team").cast(pl.String),
            pl.col("game").cast(pl.String),
            pl.col("player").cast(pl.String)
        ])
        expected_df = pl.DataFrame(expected_df).select(df_with_lags.columns)
        pl.testing.assert_frame_equal(df_with_lags, expected_df, check_dtype=False)


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_fit_transform_lag_length_2(df, column_names):
    data = df(
        {
            "player": ["a", "b", "a", "a"],
            "game": [1, 1, 2, 3],
            "points": [1, 2, 3, 4],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
            ],
            "team": [1, 2, 1, 1],
        }
    )
    try:
        original_df = data.copy()
    except:
        original_df = data.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=2,
        granularity=["player"],
    )

    df_with_lags = lag_transformation.generate_historical(data, column_names=column_names)

    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{"lag_1_points": [None, None, 1, 3], "lag_2_points": [None, None, None, 1]}
        )
        expected_df["team"] = expected_df["team"].astype("str")
        expected_df["game"] = expected_df["game"].astype("str")
        expected_df["player"] = expected_df["player"].astype("str")
        pd.testing.assert_frame_equal(
            df_with_lags, expected_df, check_like=True, check_dtype=False
        )

    else:
        expected_df = original_df.with_columns([
            pl.Series("lag_1_points", [None, None, 1, 3]),
            pl.Series("lag_2_points", [None, None, None, 1]),
            pl.col("team").cast(pl.String),
            pl.col("game").cast(pl.String),
            pl.col("player").cast(pl.String)
        ])

        pl.testing.assert_frame_equal(
            df_with_lags, expected_df.select(df_with_lags.columns), check_dtype=False
        )


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_fit_transform_and_transform(df, column_names):
    historical_df = df(
        {
            "player": ["a", "b", "a"],
            "game": [1, 1, 2],
            "points": [1, 2, 3],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
            ],
            "team": [1, 2, 1],
        }
    )

    future_df = df(
        {
            "player": ["a", "b", "a"],
            "game": [3, 3, 4],
            "start_date": [
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-05"),
            ],
            "team": [1, 2, 1],
        }
    )
    try:
        future_df_copy = future_df.copy()
    except:
        future_df_copy = future_df.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=["player"],
    )

    _ = lag_transformation.generate_historical(historical_df, column_names=column_names)
    future_transformed_df = lag_transformation.generate_future(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_df = future_df_copy.assign(
            **{lag_transformation.prefix + "1_points": [3, 2, 3]}
        )
        expected_df["team"] = expected_df["team"].astype("str")
        expected_df["game"] = expected_df["game"].astype("str")
        expected_df["player"] = expected_df["player"].astype("str")
        pd.testing.assert_frame_equal(
            future_transformed_df, expected_df, check_like=True, check_dtype=False
        )
    else:
        expected_df = future_df_copy.with_columns([
            pl.Series(lag_transformation.prefix + "1_points", [3, 2, 3]),
            pl.col("team").cast(pl.String),
            pl.col("game").cast(pl.String),
            pl.col("player").cast(pl.String)
        ])
        pl.testing.assert_frame_equal(future_transformed_df, expected_df.select(future_transformed_df.columns),
                                      check_dtype=False)


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_transformation_transform_2_lags(df, column_names):
    historical_df = df(
        {
            "player": ["a", "b", "a"],
            "game": [1, 1, 2],
            "points": [1, 2, 3],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
            ],
            "team": [1, 2, 1],
        }
    )

    future_df = df(
        {
            "player": ["a", "b", "a"],
            "game": [3, 3, 4],
            "start_date": [
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-05"),
            ],
            "team": [1, 2, 1],
        }
    )
    try:
        future_df_copy = future_df.copy()
    except:
        future_df_copy = future_df.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=2,
        granularity=["player"],
    )

    _ = lag_transformation.generate_historical(historical_df, column_names=column_names)
    future_transformed_df = lag_transformation.generate_future(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_df = future_df_copy.assign(
            **{lag_transformation.prefix + "1_points": [3, 2, 3]}
        )
        expected_df = expected_df.assign(
            **{lag_transformation.prefix + "2_points": [1, None, 1]}
        )
        expected_df["team"] = expected_df["team"].astype("str")
        expected_df["game"] = expected_df["game"].astype("str")
        expected_df["player"] = expected_df["player"].astype("str")
        pd.testing.assert_frame_equal(
            future_transformed_df, expected_df, check_like=True, check_dtype=False
        )
    else:
        expected_df = future_df_copy.with_columns([
            pl.Series(lag_transformation.prefix + "1_points", [3, 2, 3]),
            pl.Series(lag_transformation.prefix + "2_points", [1, None, 1]),
            pl.col("team").cast(pl.String),
            pl.col("game").cast(pl.String),
            pl.col("player").cast(pl.String)
        ])
        pl.testing.assert_frame_equal(future_transformed_df, expected_df.select(future_transformed_df.columns),
                                      check_dtype=False)


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_transformer_fit_transform_transform_multiple_teams(df, column_names):
    data = df(
        {
            "player": ["a", "b", "a", "c"],
            "game": [1, 1, 2, 2],
            "team": [1, 2, 1, 3],
            "points": [1, 2, 3, 5],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
            ],
        }
    )
    try:
        original_df = data.copy()
    except:
        original_df = data.clone()

    lag_transformation = LagTransformer(
        features=["points"], lag_length=2, granularity=["player"], add_opponent=True
    )

    df_with_lags = lag_transformation.generate_historical(data, column_names=column_names)

    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                lag_transformation.features_out[0]: [None, None, 1, None],
                lag_transformation.features_out[1]: [None, None, None, 1],
                lag_transformation.features_out[2]: [None, None, None, None],
                lag_transformation.features_out[3]: [None, None, None, None],
            }
        )
        expected_df["team"] = expected_df["team"].astype("str")
        expected_df["game"] = expected_df["game"].astype("str")
        expected_df["player"] = expected_df["player"].astype("str")
        pd.testing.assert_frame_equal(
            df_with_lags, expected_df, check_like=True, check_dtype=False
        )
    else:
        expected_df = original_df.with_columns([
            pl.Series(lag_transformation.features_out[0], [None, None, 1, None]),
            pl.Series(lag_transformation.features_out[1], [None, None, None, 1]),
            pl.Series(lag_transformation.features_out[2], [None, None, None, None]),
            pl.Series(lag_transformation.features_out[3], [None, None, None, None]),
            pl.col("team").cast(pl.String),
            pl.col("game").cast(pl.String),
            pl.col("player").cast(pl.String)
        ])

    future_df = pd.DataFrame(
        {
            "player": ["a", "b", "b", "c"],
            "game": [3, 3, 4, 4],
            "team": [1, 2, 2, 3],
            "start_date": [
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-05"),
                pd.to_datetime("2023-01-05"),
            ],
        }
    )
    try:
        expected_future_df = future_df.copy()
    except:
        expected_future_df = future_df.clone()

    future_df = lag_transformation.generate_future(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_future_df = expected_future_df.assign(
            **{
                lag_transformation.features_out[0]: [3, 2, 2, 5],
                lag_transformation.features_out[1]: [2, 3, 5, 2],
                lag_transformation.features_out[2]: [1, None, None, None],
                lag_transformation.features_out[3]: [None, 1, None, None],
            }
        )
        expected_future_df["team"] = expected_future_df["team"].astype("str")
        expected_future_df["game"] = expected_future_df["game"].astype("str")
        expected_future_df["player"] = expected_future_df["player"].astype("str")
        pd.testing.assert_frame_equal(
            future_df, expected_future_df, check_like=True, check_dtype=False
        )
    else:
        expected_future_df = expected_future_df.with_columns([
            pl.Series(lag_transformation.features_out[0], [3, 2, 2, 5]),
            pl.Series(lag_transformation.features_out[1], [2, 3, 5, 2]),
            pl.Series(lag_transformation.features_out[2], [1, None, None, None]),
            pl.Series(lag_transformation.features_out[3], [None, 1, None, None]),
            pl.col("team").cast(pl.String),
            pl.col("game").cast(pl.String),
            pl.col("player").cast(pl.String)
        ])
        pl.testing.assert_frame_equal(future_df, expected_future_df.select(future_df.columns), check_dtype=False)


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_transformer_parent_match_id(df, column_names: ColumnNames):
    column_names = column_names
    column_names.update_match_id = "series_id"
    historical_df = df(
        {
            "player": ["a", "a", "a", "a"],
            "game": [1, 2, 3, 4],
            "points": [1, 2, 3, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": [1, 1, 1, 1],
            "series_id": [1, 1, 2, 3],
        }
    )
    try:
        expected_df = historical_df.copy()
    except:
        expected_df = historical_df.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=2,
        granularity=["player"],
    )

    transformed_df = lag_transformation.generate_historical(
        historical_df, column_names=column_names
    )

    if isinstance(historical_df, pd.DataFrame):
        expected_df = expected_df.assign(
            **{lag_transformation.features_out[0]: [None, None, 1.5, 3]}
        )
        expected_df = expected_df.assign(
            **{lag_transformation.features_out[1]: [None, None, None, 1.5]}
        )
        expected_df["team"] = expected_df["team"].astype("str")
        expected_df["game"] = expected_df["game"].astype("str")
        expected_df["player"] = expected_df["player"].astype("str")
        expected_df["series_id"] = expected_df["series_id"].astype("str")
        pd.testing.assert_frame_equal(
            transformed_df, expected_df, check_like=True, check_dtype=False
        )
    else:
        expected_df = expected_df.with_columns([
            pl.Series(lag_transformation.features_out[0], [None, None, 1.5, 3]),
            pl.Series(lag_transformation.features_out[1], [None, None, None, 1.5]),
            pl.col("team").cast(pl.String),
            pl.col("game").cast(pl.String),
            pl.col("player").cast(pl.String),
            pl.col("series_id").cast(pl.String)
        ])
        pl.testing.assert_frame_equal(transformed_df, expected_df.select(transformed_df.columns), check_dtype=False)


def test_rolling_mean_fit_transform(column_names):
    df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "a"],
            "game": [1, 1, 2, 3],
            "points": [1, 2, 3, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": [1, 2, 1, 1],
        }
    )
    original_df = df.copy()

    rolling_mean_transformation = RollingMeanTransformerPolars(
        features=["points"],
        window=2,
        min_periods=1,
        granularity=["player"],
    )

    df_with_rolling_mean = rolling_mean_transformation.generate_historical(
        df, column_names=column_names
    )

    expected_df = original_df.assign(
        **{
            f"{rolling_mean_transformation.prefix}2_points": [
                None,
                None,
                1,
                (3 + 1) / 2,
            ]
        }
    )
    pd.testing.assert_frame_equal(
        df_with_rolling_mean, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_fit_transform_and_transform(column_names):
    historical_df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "a"],
            "game": [1, 1, 2, 3],
            "points": [1, 2, 3, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": [1, 2, 1, 1],
        }
    )

    future_df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "b"],
            "game": [4, 4, 5, 5],
            "start_date": [
                pd.to_datetime("2023-01-05"),
                pd.to_datetime("2023-01-05"),
                pd.to_datetime("2023-01-06"),
                pd.to_datetime("2023-01-06"),
            ],
            "team": [1, 2, 1, 2],
        }
    )

    original_future_df = future_df.copy()
    rolling_mean_transformation = RollingMeanTransformerPolars(
        features=["points"],
        window=2,
        min_periods=1,
        granularity=["player"],
        add_opponent=True,
    )

    _ = rolling_mean_transformation.generate_historical(
        df=historical_df, column_names=column_names
    )
    transformed_future_df = rolling_mean_transformation.generate_future(future_df)

    expected_df = original_future_df.assign(
        **{
            f"{rolling_mean_transformation.prefix}2_points": [2.5, 2, 2.5, 2],
            rolling_mean_transformation.features_out[1]: [2, 2.5, 2, 2.5],
        }
    )
    pd.testing.assert_frame_equal(transformed_future_df, expected_df, check_like=True)


def test_rolling_mean_transformer_fit_transformer_team_stat(column_names):
    historical_df = pd.DataFrame(
        {
            "player": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "game": [1, 1, 1, 1, 2, 2, 2, 2],
            "score_difference": [10, 10, -10, -10, 15, 15, -15, -15],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
            ],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )

    expected_df = historical_df.copy()

    rolling_mean_transformation = RollingMeanTransformerPolars(
        features=["score_difference"],
        window=2,
        min_periods=1,
        granularity=["team"],
    )

    transformed_data = rolling_mean_transformation.generate_historical(
        historical_df, column_names=column_names
    )
    expected_df[rolling_mean_transformation.prefix + "2_score_difference"] = [
        None,
        None,
        None,
        None,
        10,
        10,
        -10,
        -10,
    ]
    pd.testing.assert_frame_equal(
        transformed_data, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_days_fit_transform(column_names):
    df = pd.DataFrame(
        {
            "player": ["a", "a", "b", "a", "a"],
            "game": [1, 2, 2, 3, 4],
            "points": [1, 1, 2, 3, 2],
            "points2": [2, 2, 4, 6, 4],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-10"),
            ],
            "team": [1, 1, 2, 1, 1],
        }
    )

    original_df = df.copy()

    rolling_mean_transformation = RollingMeanDaysTransformer(
        features=["points", "points2"], days=2, granularity=["player"], add_count=True
    )

    transformed_df = rolling_mean_transformation.generate_historical(
        df, column_names=column_names
    )

    expected_df = original_df.assign(
        **{
            rolling_mean_transformation.features_out[0]: [None, None, None, 1, None],
            rolling_mean_transformation.features_out[1]: [None, None, None, 2, None],
            rolling_mean_transformation.features_out[2]: [0, 0, 0, 2, 0],
        }
    )

    pd.testing.assert_frame_equal(
        transformed_df, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_days_series_id(column_names: ColumnNames):
    column_names = column_names
    column_names.update_match_id = "series_id"
    historical_df = pd.DataFrame(
        {
            "player": ["a", "a", "a", "a"],
            "game": [1, 2, 3, 4],
            "points": [1, 2, 3, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": [1, 1, 1, 1],
            "series_id": [1, 1, 2, 3],
        }
    )
    expected_df = historical_df.copy()

    transformer = RollingMeanDaysTransformer(
        features=["points"],
        days=2,
    )

    transformed_df = transformer.generate_historical(
        df=historical_df, column_names=column_names
    )

    expected_df = expected_df.assign(
        **{transformer.features_out[0]: [None, None, 1.5, 3]}
    )

    pd.testing.assert_frame_equal(
        transformed_df, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_days_fit_transform_40_days(column_names):
    df = pd.DataFrame(
        {
            "player": ["a", "a", "a", "b", "a", "b"],
            "game": [1, 2, 3, 4, 5, 6],
            "points": [1, 1.5, 2, 3, 2, 4],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-10"),
                pd.to_datetime("2023-01-10"),
                pd.to_datetime("2023-02-15"),
            ],
            "team": [1, 1, 1, 2, 1, 2],
        }
    )

    original_df = df.copy()

    rolling_mean_transformation = RollingMeanDaysTransformer(
        features=["points"],
        days=40,
        granularity=["player"],
    )

    transformed_df = rolling_mean_transformation.generate_historical(
        df, column_names=column_names
    )

    expected_df = original_df.assign(
        **{
            rolling_mean_transformation.features_out[0]: [None, 1, 1, None, 1.5, 3],
        }
    )

    pd.testing.assert_frame_equal(
        transformed_df, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_days_fit_transform_opponent(column_names):
    df = pd.DataFrame(
        {
            "player": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "game": [1, 1, 1, 1, 2, 2, 2, 2],
            "points": [1, 1.5, 2, 3, 2, 4, 1, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
            ],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )

    original_df = df.copy()

    rolling_mean_transformation = RollingMeanDaysTransformer(
        features=["points"], days=10, granularity=["player"], add_opponent=True
    )

    transformed_df = rolling_mean_transformation.generate_historical(
        df, column_names=column_names
    )

    expected_df = original_df.assign(
        **{
            rolling_mean_transformation.features_out[0]: [
                None,
                None,
                None,
                None,
                1,
                1.5,
                2,
                3,
            ],
            rolling_mean_transformation.features_out[1]: [
                None,
                None,
                None,
                None,
                2.5,
                2.5,
                1.25,
                1.25,
            ],
        }
    )

    pd.testing.assert_frame_equal(
        transformed_df, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_days_transformer_transform(column_names):
    historical_df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "b"],
            "game": [1, 1, 2, 2],
            "points": [1, 2, 3, 4],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
            ],
            "team": [1, 2, 1, 2],
        }
    )

    future_df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "b"],
            "game": [3, 3, 4, 4],
            "start_date": [
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-25"),
                pd.to_datetime("2023-01-25"),
            ],
            "team": [1, 2, 1, 2],
        }
    )

    transformer = RollingMeanDaysTransformer(
        features=["points"],
        days=10,
        granularity=["player"],
        add_opponent=True,
        add_count=True,
    )
    expected_historical_df = historical_df.copy()
    historical_df = transformer.generate_historical(
        historical_df, column_names=column_names
    )
    expected_historical_df = expected_historical_df.assign(
        **{
            transformer.features_out[0]: [None, None, 1, 2],
            transformer.features_out[1]: [None, None, 2, 1],
            f"{transformer.prefix}10_count": [0, 0, 1, 1],
            f"{transformer.prefix}10_count_opponent": [0, 0, 1, 1],
        }
    )

    pd.testing.assert_frame_equal(
        historical_df, expected_historical_df, check_like=True, check_dtype=False
    )

    expected_df = future_df.copy()

    transformed_future_df = transformer.generate_future(df=future_df)

    expected_df = expected_df.assign(
        **{
            transformer.features_out[0]: [2, 3, 2, 3],
            transformer.features_out[1]: [3, 2, 3, 2],
            f"{transformer.prefix}10_count": [2, 2, 2, 2],
            f"{transformer.prefix}10_count_opponent": [2, 2, 2, 2],
        }
    )

    pd.testing.assert_frame_equal(
        transformed_future_df, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_days_tranformer_transform_first_future_beyond_window(
        column_names,
):
    historical_df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "b"],
            "game": [1, 1, 2, 2],
            "points": [1, 2, 3, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-12"),
                pd.to_datetime("2023-01-12"),
            ],
            "team": [1, 2, 1, 2],
        }
    )

    transformer = RollingMeanDaysTransformer(
        features=["points"],
        days=10,
        granularity=["player"],
        add_opponent=True,
        add_count=True,
    )
    expected_historical_df = historical_df.copy()
    historical_df = transformer.generate_historical(
        historical_df, column_names=column_names
    )

    expected_historical_df = expected_historical_df.assign(
        **{
            transformer.features_out[0]: [None, None, None, None],
            transformer.features_out[1]: [None, None, None, None],
            f"{transformer.prefix}10_count": [0, 0, 0, 0],
            f"{transformer.prefix}10_count_opponent": [0, 0, 0, 0],
        }
    )

    pd.testing.assert_frame_equal(
        historical_df, expected_historical_df, check_like=True, check_dtype=False
    )

    future_df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "b"],
            "game": [3, 3, 4, 4],
            "start_date": [
                pd.to_datetime("2023-01-16"),
                pd.to_datetime("2023-01-16"),
                pd.to_datetime("2023-01-25"),
                pd.to_datetime("2023-01-25"),
            ],
            "team": [1, 2, 1, 2],
        }
    )

    expected_df = future_df.copy()

    transformed_future_df = transformer.generate_future(df=future_df)

    expected_df = expected_df.assign(
        **{
            transformer.features_out[0]: [3, 2, 3, 2],
            transformer.features_out[1]: [2, 3, 2, 3],
            f"{transformer.prefix}10_count": [1, 1, 1, 1],
            f"{transformer.prefix}10_count_opponent": [1, 1, 1, 1],
        }
    )

    pd.testing.assert_frame_equal(
        transformed_future_df, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_transform_parent_match_id(column_names: ColumnNames):
    column_names = column_names
    column_names.update_match_id = "series_id"
    historical_df = pd.DataFrame(
        {
            "player": ["a", "a", "a", "a"],
            "game": [1, 2, 3, 4],
            "points": [1, 2, 3, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": [1, 1, 1, 1],
            "series_id": [1, 1, 2, 3],
        }
    )
    expected_df = historical_df.copy()

    transformer = RollingMeanTransformerPolars(
        features=["points"],
        window=2,
    )

    transformed_df = transformer.generate_historical(
        historical_df, column_names=column_names
    )

    expected_df = expected_df.assign(
        **{transformer.features_out[0]: [None, None, 1.5, (1.5 + 3) / 2]}
    )
    pd.testing.assert_frame_equal(
        transformed_df, expected_df, check_like=True, check_dtype=False
    )


@pytest.mark.parametrize("min_periods", [1, 10])
def test_binary_granularity_rolling_mean_transformer(column_names, min_periods):
    historical_df = pd.DataFrame(
        {
            "player": ["a", "b", "c", "d", "a", "b", "c", "d", "c", "d"],
            "game": [1, 1, 1, 1, 2, 2, 2, 3, 4, 4],
            "score_difference": [10, 10, -10, -10, 15, 15, -15, -20, 2, 2],
            "won": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
            ],
            "team": [1, 1, 2, 2, 1, 1, 2, 2, 2, 2],
            "prob": [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.3, 0.3, 0.2, 0.2],
        }
    )

    historical_df["team"] = historical_df["team"].astype("str")
    historical_df["game"] = historical_df["game"].astype("str")
    historical_df["player"] = historical_df["player"].astype("str")

    expected_df = historical_df.copy()

    transformer = BinaryOutcomeRollingMeanTransformer(
        features=["score_difference"],
        binary_column="won",
        window=10,
        min_periods=min_periods,
        granularity=["player"],
        prob_column="prob",
    )

    transformed_data = transformer.generate_historical(
        df=historical_df, column_names=column_names
    )

    if min_periods == 1:
        expected_df[transformer.features_out[0]] = [
            None,
            None,
            None,
            None,
            10,
            10,
            None,
            None,
            None,
            None,
        ]
        expected_df[transformer.features_out[1]] = [
            None,
            None,
            None,
            None,
            None,
            None,
            -10,
            -10,
            -12.5,
            -15,
        ]
        expected_df[transformer.features_out[2]] = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
    else:
        for i in range(3):
            expected_df[transformer.features_out[i]] = [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]

    pd.testing.assert_frame_equal(
        transformed_data, expected_df, check_like=True, check_dtype=False
    )


def test_binary_granularity_rolling_mean_fit_transform_transform(column_names):
    historical_df = pd.DataFrame(
        {
            "player": ["a", "b", "c", "d", "a", "b", "c", "d", "a", "b", "c", "d"],
            "game": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "score_difference": [10, 10, -10, -10, 15, 15, -15, -20, -2, -2, 2, 2],
            "won": [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
            ],
            "team": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            "prob": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )

    historical_df["team"] = historical_df["team"].astype("str")
    historical_df["game"] = historical_df["game"].astype("str")
    historical_df["player"] = historical_df["player"].astype("str")
    expected_historical_df = historical_df.copy()

    transformer = BinaryOutcomeRollingMeanTransformer(
        features=["score_difference"],
        binary_column="won",
        window=3,
        min_periods=1,
        granularity=["player"],
        add_opponent=True,
        prob_column="prob",
    )

    historical_df = transformer.generate_historical(
        historical_df, column_names=column_names
    )
    expected_historical_df[transformer.features_out[0]] = [
        None,
        None,
        None,
        None,
        10,
        10,
        None,
        None,
        12.5,
        12.5,
        None,
        None,
    ]
    expected_historical_df[transformer.features_out[1]] = [
        None,
        None,
        None,
        None,
        None,
        None,
        -10,
        -10,
        None,
        None,
        -12.5,
        -15,
    ]

    expected_historical_df[transformer.features_out[2]] = [
        None,
        None,
        None,
        None,
        None,
        None,
        10,
        10,
        None,
        None,
        12.5,
        12.5,
    ]

    expected_historical_df[transformer.features_out[3]] = [
        None,
        None,
        None,
        None,
        -10,
        -10,
        None,
        None,
        -13.75,
        -13.75,
        None,
        None,
    ]

    expected_historical_df[transformer.features_out[4]] = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    pd.testing.assert_frame_equal(
        historical_df, expected_historical_df, check_like=True, check_dtype=False
    )

    future_df = pd.DataFrame(
        {
            "player": ["a", "d", "a", "d"],
            "game": [5, 5, 6, 6],
            "score_difference": [None, None, None, None],
            "won": [None, None, None, None],
            "start_date": [
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": [1, 2, 1, 2],
            "prob": [0.6, 0.4, 0.7, 0.3],
        }
    )

    future_df["team"] = future_df["team"].astype("str")
    future_df["game"] = future_df["game"].astype("str")
    future_df["player"] = future_df["player"].astype("str")

    expected_future_df = future_df.copy()

    future_df = transformer.generate_future(future_df)
    expected_future_df[transformer.features_out[0]] = [12.5, 2, 12.5, 2]
    expected_future_df[transformer.features_out[1]] = [-2, -15, -2, -15]
    expected_future_df[transformer.features_out[2]] = [2, 12.5, 2, 12.5]
    expected_future_df[transformer.features_out[3]] = [-15, -2, -15, -2]
    expected_future_df[transformer.features_out[4]] = [
        12.5 * 0.6 + 0.4 * -2,
        2 * 0.4 - 15 * 0.6,
        12.5 * 0.7 - 2 * 0.3,
        2 * 0.3 - 15 * 0.7,
    ]

    pd.testing.assert_frame_equal(
        future_df, expected_future_df, check_like=True, check_dtype=False
    )


def test_binary_granularity_rolling_mean_fit_transform_opponent(column_names):
    df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "b", "a", "b"],
            "game": [1, 1, 2, 2, 3, 3],
            "score_difference": [10, -10, 5, -5, 3, 3],
            "won": [1, 0, 0, 1, 1, 0],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
            ],
            "team": [1, 2, 1, 2, 1, 2],
            "prob": [0.6, 0.4, 0.6, 0.4, 0.6, 0.4],
        }
    )

    df["team"] = df["team"].astype("str")
    df["game"] = df["game"].astype("str")
    df["player"] = df["player"].astype("str")
    expected_historical_df = df.copy()

    rolling_mean_transformation = BinaryOutcomeRollingMeanTransformer(
        features=["score_difference"],
        binary_column="won",
        window=2,
        min_periods=1,
        granularity=["player"],
        add_opponent=True,
        prob_column="prob",
    )

    df = rolling_mean_transformation.generate_historical(df, column_names=column_names)

    expected_historical_df[rolling_mean_transformation.features_out[0]] = [
        None,
        None,
        10,
        None,
        10,
        -5,
    ]
    expected_historical_df[rolling_mean_transformation.features_out[1]] = [
        None,
        None,
        None,
        -10,
        5,
        -10,
    ]
    expected_historical_df[rolling_mean_transformation.features_out[2]] = [
        None,
        None,
        None,
        10,
        -5,
        10,
    ]
    expected_historical_df[rolling_mean_transformation.features_out[3]] = [
        None,
        None,
        -10,
        None,
        -10,
        5,
    ]

    expected_historical_df[rolling_mean_transformation.features_out[4]] = [
        None,
        None,
        None,
        None,
        10 * 0.6 + 0.4 * 5,
        -10 * 0.6 + 0.4 * -5,
    ]

    pd.testing.assert_frame_equal(
        df, expected_historical_df, check_like=True, check_dtype=False
    )
