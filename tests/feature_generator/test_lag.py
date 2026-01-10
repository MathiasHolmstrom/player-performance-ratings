import copy

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spforge import ColumnNames
from spforge.feature_generator import (
    LagTransformer,
)


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game",
        team_id="team",
        player_id="player",
        start_date="start_date",
    )


@pytest.mark.parametrize("use_column_names", [True, False])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_team_lag_transform_historical_group_to_team_granularity(
    df, column_names, use_column_names
):
    "Should calculate average point of prior game"

    data = df(
        {
            "player": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
            "game": [1, 1, 1, 1, 2, 2, 2, 2],
            "points": [1, 2, 3, 2, 4, 5, 6, 7],
            "points2": [1, 2, 3, 2, 4, 5, 6, 7],
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
    except Exception:
        original_df = data.clone()

    if use_column_names:
        lag_transformation = LagTransformer(
            features=["points"],
            lag_length=1,
            granularity=["team"],
            group_to_granularity=["team", "game"],
        )

        df_with_lags = lag_transformation.fit_transform(data, column_names=column_names)
    else:

        lag_transformation = LagTransformer(
            features=["points"],
            lag_length=1,
            granularity=["team"],
            group_to_granularity=["team", "game"],
            match_id_column="game",
        )

        df_with_lags = lag_transformation.fit_transform(data, column_names=None)

    if isinstance(data, pl.DataFrame):
        expected_df = original_df.with_columns(
            [
                pl.Series("lag_points1", [None, None, None, None, 1.5, 1.5, 2.5, 2.5]),
                pl.col("team"),
                pl.col("game"),
                pl.col("player"),
            ]
        )
        pl.testing.assert_frame_equal(
            df_with_lags, expected_df.select(df_with_lags.columns), check_dtype=False
        )

    elif isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{"lag_points1": [None, None, None, None, 1.5, 1.5, 2.5, 2.5]}
        )

        pd.testing.assert_frame_equal(
            df_with_lags, expected_df[df_with_lags.columns], check_dtype=False
        )


@pytest.mark.parametrize("use_column_names", [True, False])
def test_lag_fit_tranform_group_to_team_granularity(column_names, use_column_names):
    data = pd.DataFrame(
        {
            column_names.start_date: ["2023-01-01"] * 4 + ["2023-01-02"] * 4 + ["2023-01-03"] * 4,
            column_names.player_id: [1, 2, 3, 4] * 3,
            column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            column_names.team_id: [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            "team_points": [1, 1, 2, 2, 3, 3, 4, 4, 9, 10, 11, 12],
        }
    )

    if use_column_names:
        transformer = LagTransformer(
            features=["team_points"],
            lag_length=2,
            granularity=[column_names.team_id],
            group_to_granularity=[column_names.team_id, column_names.match_id],
        )
    else:
        transformer = LagTransformer(
            features=["team_points"],
            lag_length=2,
            granularity=[column_names.team_id],
            group_to_granularity=[column_names.team_id, column_names.match_id],
            match_id_column="game",
        )
        column_names = None

    expected_df = data.copy()
    transformed_df = transformer.fit_transform(df=data, column_names=column_names)
    expected_df[transformer.features_out[0]] = [None] * 4 + [1, 1, 2, 2, 3, 3, 4, 4]
    expected_df[transformer.features_out[1]] = [None] * 8 + [1, 1, 2, 2]

    pd.testing.assert_frame_equal(
        expected_df, transformed_df[expected_df.columns], check_dtype=False
    )


@pytest.mark.parametrize("use_column_names", [True, False])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_fit_transform_group_to_team_granularity_update_match_id(
    df, column_names, use_column_names
):
    "Should calculate average point of prior game"
    column_names = copy.deepcopy(column_names)
    column_names.update_match_id = "update_match_id"
    data = df(
        {
            "player": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "update_match_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "team": [1, 2, 1, 2, 1, 2, 1, 2],
            "game": [1, 1, 2, 2, 3, 3, 4, 4],
            "points": [1, 2, 3, 2, 4, 5, 6, 7],
            "points2": [1, 2, 3, 4, 5, 6, 7, 8],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
        }
    )

    try:
        original_df = data.copy()
    except Exception:
        original_df = data.clone()

    if not use_column_names:
        lag_transformation = LagTransformer(
            features=["points"],
            lag_length=1,
            granularity=["team"],
            group_to_granularity=["team", "game"],
            match_id_column=column_names.match_id,
            update_column=column_names.update_match_id,
        )
        column_names = None
    else:
        lag_transformation = LagTransformer(
            features=["points"],
            lag_length=1,
            granularity=["team"],
            group_to_granularity=["team", "game"],
            column_names=column_names,
            add_opponent=True,
        )

    df_with_lags = lag_transformation.fit_transform(data, column_names=column_names)
    if isinstance(data, pl.DataFrame):
        expected_df = original_df.with_columns(
            [
                pl.Series("lag_points1", [None, None, None, None, 3, 2, 3, 2]),
                pl.col("team"),
                pl.col("game"),
                pl.col("player"),
            ]
        )
        if use_column_names:
            expected_df = expected_df.with_columns(
                pl.Series(
                    lag_transformation.features_out[1],
                    [None, None, None, None, 2, 3, 2, 3],
                )
            )
        pl.testing.assert_frame_equal(
            df_with_lags, expected_df.select(df_with_lags.columns), check_dtype=False
        )

    elif isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                "lag_points1": [None, None, None, None, 3, 2, 3, 2],
            }
        )
        if use_column_names:
            expected_df = expected_df.assign(
                **{
                    lag_transformation.features_out[1]: [
                        None,
                        None,
                        None,
                        None,
                        2,
                        3,
                        2,
                        3,
                    ]
                }
            )

        pd.testing.assert_frame_equal(
            df_with_lags, expected_df[df_with_lags.columns], check_dtype=False
        )


@pytest.mark.parametrize("use_column_names", [True, False])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_transform_historical_2_features_update_match_id(df, column_names, use_column_names):
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
    original_df = data.clone() if isinstance(data, pl.DataFrame) else data.copy()

    if use_column_names:
        lag_transformation = LagTransformer(
            features=["points", "points_per_minute"],
            lag_length=1,
            granularity=["player"],
        )
    else:
        lag_transformation = LagTransformer(
            features=["points", "points_per_minute"],
            lag_length=1,
            granularity=["player"],
            match_id_column=column_names.match_id,
        )
        column_names = None

    df_with_lags = lag_transformation.fit_transform(data, column_names=column_names)
    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                "lag_points1": [None, None, 1],
                "lag_points_per_minute1": [None, None, 0.5],
            }
        )

        pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True, check_dtype=False)

    else:
        expected_df = original_df.with_columns(
            [
                pl.Series("lag_points1", [None, None, 1]),
                pl.Series("lag_points_per_minute1", [None, None, 0.5]),
                pl.col("team"),
                pl.col("game"),
                pl.col("player"),
            ]
        )
        expected_df = pl.DataFrame(expected_df).select(df_with_lags.columns)
        pl.testing.assert_frame_equal(df_with_lags, expected_df, check_dtype=False)


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_transform_historical_lag_length_2(df, column_names):
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
    except Exception:
        original_df = data.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=2,
        granularity=["player"],
    )

    df_with_lags = lag_transformation.fit_transform(data, column_names=column_names)

    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                "lag_points1": [None, None, 1, 3],
                "lag_points2": [None, None, None, 1],
            }
        )

        pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True, check_dtype=False)

    else:
        expected_df = original_df.with_columns(
            [
                pl.Series("lag_points1", [None, None, 1, 3]),
                pl.Series("lag_points2", [None, None, None, 1]),
                pl.col("team"),
                pl.col("game"),
                pl.col("player"),
            ]
        )

        pl.testing.assert_frame_equal(
            df_with_lags, expected_df.select(df_with_lags.columns), check_dtype=False
        )


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_transform_historical_and_transform_future(df, column_names):
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
    except Exception:
        future_df_copy = future_df.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=["player"],
    )

    _ = lag_transformation.fit_transform(historical_df, column_names=column_names)
    future_transformed_df = lag_transformation.future_transform(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_df = future_df_copy.assign(**{lag_transformation.prefix + "_points1": [3, 2, 3]})

        pd.testing.assert_frame_equal(
            future_transformed_df, expected_df, check_like=True, check_dtype=False
        )
    else:
        expected_df = future_df_copy.with_columns(
            [
                pl.Series(lag_transformation.prefix + "_points1", [3, 2, 3]),
                pl.col("team"),
                pl.col("game"),
                pl.col("player"),
            ]
        )
        pl.testing.assert_frame_equal(
            future_transformed_df,
            expected_df.select(future_transformed_df.columns),
            check_dtype=False,
        )


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
    except Exception:
        future_df_copy = future_df.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=2,
        granularity=["player"],
    )

    _ = lag_transformation.fit_transform(historical_df, column_names=column_names)
    future_transformed_df = lag_transformation.future_transform(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_df = future_df_copy.assign(**{lag_transformation.prefix + "_points1": [3, 2, 3]})
        expected_df = expected_df.assign(**{lag_transformation.prefix + "_points2": [1, None, 1]})

        pd.testing.assert_frame_equal(
            future_transformed_df, expected_df, check_like=True, check_dtype=False
        )
    else:
        expected_df = future_df_copy.with_columns(
            [
                pl.Series(lag_transformation.prefix + "_points1", [3, 2, 3]),
                pl.Series(lag_transformation.prefix + "_points2", [1, None, 1]),
                pl.col("team"),
                pl.col("game"),
                pl.col("player"),
            ]
        )
        pl.testing.assert_frame_equal(
            future_transformed_df,
            expected_df.select(future_transformed_df.columns),
            check_dtype=False,
        )


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
    except Exception:
        original_df = data.clone()

    lag_transformation = LagTransformer(
        features=["points"], lag_length=2, granularity=["player"], add_opponent=True
    )

    df_with_lags = lag_transformation.fit_transform(data, column_names=column_names)

    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                lag_transformation.features_out[0]: [None, None, 1, None],
                lag_transformation.features_out[1]: [None, None, None, 1],
                lag_transformation.features_out[2]: [None, None, None, None],
                lag_transformation.features_out[3]: [None, None, None, None],
            }
        )

        pd.testing.assert_frame_equal(df_with_lags, expected_df, check_like=True, check_dtype=False)
    else:
        expected_df = original_df.with_columns(
            [
                pl.Series(lag_transformation.features_out[0], [None, None, 1, None]),
                pl.Series(lag_transformation.features_out[1], [None, None, None, 1]),
                pl.Series(lag_transformation.features_out[2], [None, None, None, None]),
                pl.Series(lag_transformation.features_out[3], [None, None, None, None]),
                pl.col("team"),
                pl.col("game"),
                pl.col("player"),
            ]
        )
        assert_frame_equal(
            df_with_lags, expected_df.select(df_with_lags.columns), check_dtype=False
        )

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
    except Exception:
        expected_future_df = future_df.clone()

    future_df = lag_transformation.future_transform(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_future_df = expected_future_df.assign(
            **{
                lag_transformation.features_out[0]: [3, 2, 2, 5],
                lag_transformation.features_out[1]: [2, 3, 5, 2],
                lag_transformation.features_out[2]: [1, None, None, None],
                lag_transformation.features_out[3]: [None, 1, None, None],
            }
        )

        pd.testing.assert_frame_equal(
            future_df, expected_future_df, check_like=True, check_dtype=False
        )
    else:
        expected_future_df = expected_future_df.with_columns(
            [
                pl.Series(lag_transformation.features_out[0], [3, 2, 2, 5]),
                pl.Series(lag_transformation.features_out[1], [2, 3, 5, 2]),
                pl.Series(lag_transformation.features_out[2], [1, None, None, None]),
                pl.Series(lag_transformation.features_out[3], [None, 1, None, None]),
                pl.col("team"),
                pl.col("game"),
                pl.col("player"),
            ]
        )
        pl.testing.assert_frame_equal(
            future_df, expected_future_df.select(future_df.columns), check_dtype=False
        )
