import copy

import pandas as pd
import pytest
import polars as pl
from polars.testing import assert_frame_equal
from player_performance_ratings import ColumnNames
from player_performance_ratings.transformers import (
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
def test_lag_team_transform_historical(df, column_names, use_column_names):
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
    except:
        original_df = data.clone()

    if use_column_names:
        lag_transformation = LagTransformer(
            features=["points"],
            lag_length=1,
            granularity=["team"],
        )

        df_with_lags = lag_transformation.transform_historical(
            data, column_names=column_names
        )
    else:

        lag_transformation = LagTransformer(
            features=["points"],
            lag_length=1,
            granularity=["team"],
            update_match_id_column="game",
        )



        df_with_lags = lag_transformation.transform_historical(
            data, column_names=None
        )

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
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_lag_fit_transform_update_match_id(df, column_names, use_column_names):
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


    if not use_column_names:
        lag_transformation = LagTransformer(
            features=["points"],
            lag_length=1,
            granularity=["team"],
            update_match_id_column=column_names.update_match_id,
        )
        column_names = None
    else:
        lag_transformation = LagTransformer(
            features=["points"],
            lag_length=1,
            granularity=["team"],
        )

    df_with_lags = lag_transformation.transform_historical(
        data, column_names=column_names
    )
    if isinstance(data, pl.DataFrame):
        expected_df = original_df.with_columns(
            [
                pl.Series("lag_points1", [None, None, None, None, 2, 2, 2, 2]),
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
            **{"lag_points1": [None, None, None, None, 2, 2, 2, 2]}
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
    if isinstance(data, pl.DataFrame):
        original_df = data.clone()
    else:
        original_df = data.copy()

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
            update_match_id_column=column_names.update_match_id
        )
        column_names = None

    df_with_lags = lag_transformation.transform_historical(
        data, column_names=column_names
    )
    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                "lag_points1": [None, None, 1],
                "lag_points_per_minute1": [None, None, 0.5],
            }
        )

        pd.testing.assert_frame_equal(
            df_with_lags, expected_df, check_like=True, check_dtype=False
        )

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
    except:
        original_df = data.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=2,
        granularity=["player"],
    )

    df_with_lags = lag_transformation.transform_historical(
        data, column_names=column_names
    )

    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                "lag_points1": [None, None, 1, 3],
                "lag_points2": [None, None, None, 1],
            }
        )

        pd.testing.assert_frame_equal(
            df_with_lags, expected_df, check_like=True, check_dtype=False
        )

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
    except:
        future_df_copy = future_df.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=["player"],
    )

    _ = lag_transformation.transform_historical(historical_df, column_names=column_names)
    future_transformed_df = lag_transformation.transform_future(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_df = future_df_copy.assign(
            **{lag_transformation.prefix + "_points1": [3, 2, 3]}
        )

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
    except:
        future_df_copy = future_df.clone()

    lag_transformation = LagTransformer(
        features=["points"],
        lag_length=2,
        granularity=["player"],
    )

    _ = lag_transformation.transform_historical(historical_df, column_names=column_names)
    future_transformed_df = lag_transformation.transform_future(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_df = future_df_copy.assign(
            **{lag_transformation.prefix + "_points1": [3, 2, 3]}
        )
        expected_df = expected_df.assign(
            **{lag_transformation.prefix + "_points2": [1, None, 1]}
        )

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
    except:
        original_df = data.clone()

    lag_transformation = LagTransformer(
        features=["points"], lag_length=2, granularity=["player"], add_opponent=True
    )

    df_with_lags = lag_transformation.transform_historical(
        data, column_names=column_names
    )

    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                lag_transformation.features_out[0]: [None, None, 1, None],
                lag_transformation.features_out[1]: [None, None, None, 1],
                lag_transformation.features_out[2]: [None, None, None, None],
                lag_transformation.features_out[3]: [None, None, None, None],
            }
        )

        pd.testing.assert_frame_equal(
            df_with_lags, expected_df, check_like=True, check_dtype=False
        )
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
    except:
        expected_future_df = future_df.clone()

    future_df = lag_transformation.transform_future(future_df)

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



@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("min_periods", [10, 1])
def test_binary_granularity_rolling_mean_transformer(df, column_names, min_periods):
    historical_df = df(
        {
            "player": ["a", "b", "c", "d", "a", "b", "c", "d", "c", "d"],
            "game": ["1", "1", "1", "1", "2", "2", "2", "3", "4", "4"],
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
            "team": ["1", "1", "2", "2", "1", "1", "2", "2", "2", "2"],
            "prob": [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.3, 0.3, 0.2, 0.2],
        }
    )

    try:
        expected_df = historical_df.copy()
    except:
        expected_df = historical_df.clone()

    transformer = BinaryOutcomeRollingMeanTransformer(
        features=["score_difference"],
        binary_column="won",
        window=10,
        min_periods=min_periods,
        granularity=["player"],
        prob_column="prob",
    )

    transformed_data = transformer.transform_historical(
        df=historical_df, column_names=column_names
    )
    if isinstance(historical_df, pd.DataFrame):
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
    else:
        if min_periods == 1:
            expected_df = expected_df.with_columns(
                [
                    pl.Series(
                        transformer.features_out[0],
                        [None, None, None, None, 10.0, 10.0, None, None, None, None],
                    ),
                    pl.Series(
                        transformer.features_out[1],
                        [
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            -10.0,
                            -10.0,
                            -12.5,
                            -15.0,
                        ],
                    ),
                    pl.Series(
                        transformer.features_out[2],
                        [None, None, None, None, None, None, None, None, None, None],
                    ),
                ]
            )

        else:
            for i in range(3):
                expected_df = expected_df.with_columns(
                    [
                        pl.Series(
                            transformer.features_out[i],
                            [
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
                            ],
                        )
                    ]
                )
        pl.testing.assert_frame_equal(
            transformed_data,
            expected_df.select(transformed_data.columns),
            check_dtype=False,
        )


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
@pytest.mark.parametrize("min_periods", [10, 1])
def test_binary_granularity_rolling_mean_fit_transform_transform(
    df, column_names, min_periods
):
    historical_df = df(
        {
            "player": ["a", "b", "c", "d", "a", "b", "c", "d", "a", "b", "c", "d"],
            "game": ["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3"],
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
            "team": ["1", "1", "2", "2", "1", "1", "2", "2", "1", "1", "2", "2"],
            "prob": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )

    try:
        expected_historical_df = historical_df.copy()
    except:
        expected_historical_df = historical_df.to_pandas()

    transformer = BinaryOutcomeRollingMeanTransformer(
        features=["score_difference"],
        binary_column="won",
        window=3,
        min_periods=1,
        granularity=["player"],
        add_opponent=True,
        prob_column="prob",
    )

    historical_df = transformer.transform_historical(
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
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ]
    if isinstance(historical_df, pd.DataFrame):
        pd.testing.assert_frame_equal(
            historical_df, expected_historical_df, check_like=True, check_dtype=False
        )
    else:
        pl.testing.assert_frame_equal(
            historical_df,
            pl.DataFrame(expected_historical_df).select(historical_df.columns),
            check_dtype=False,
        )

    future_df = pd.DataFrame(
        {
            "player": ["a", "d", "a", "d"],
            "game": ["5", "5", "6", "6"],
            "score_difference": [None, None, None, None],
            "won": [None, None, None, None],
            "start_date": [
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": ["1", "2", "1", "2"],
            "prob": [0.6, 0.4, 0.7, 0.3],
        }
    )
    try:
        expected_future_df = future_df.copy()
    except:
        expected_future_df = future_df.to_pandas()

    future_df = transformer.transform_future(future_df)
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
    if isinstance(future_df, pd.DataFrame):
        pd.testing.assert_frame_equal(
            future_df, expected_future_df, check_like=True, check_dtype=False
        )
    else:
        pl.testing.assert_frame_equal(
            future_df,
            pl.DataFrame(expected_future_df).select(future_df.columns),
            check_dtype=False,
        )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_binary_granularity_rolling_mean_fit_transform_opponent(df, column_names):
    df = df(
        {
            "player": ["a", "b", "a", "b", "a", "b"],
            "game": ["1", "1", "2", "2", "3", "3"],
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
            "team": ["1", "2", "1", "2", "1", "2"],
            "prob": [0.6, 0.4, 0.6, 0.4, 0.6, 0.4],
        }
    )

    try:
        expected_historical_df = df.copy()
    except:
        expected_historical_df = df.to_pandas()

    rolling_mean_transformation = BinaryOutcomeRollingMeanTransformer(
        features=["score_difference"],
        binary_column="won",
        window=2,
        min_periods=1,
        granularity=["player"],
        add_opponent=True,
        prob_column="prob",
    )

    df = rolling_mean_transformation.transform_historical(df, column_names=column_names)

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
    if isinstance(df, pd.DataFrame):
        pd.testing.assert_frame_equal(
            df, expected_historical_df, check_like=True, check_dtype=False
        )
    else:
        pl.testing.assert_frame_equal(
            df,
            pl.DataFrame(expected_historical_df).select(df.columns),
            check_dtype=False,
        )
