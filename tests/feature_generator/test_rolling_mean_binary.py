import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spforge import ColumnNames
from spforge.feature_generator import BinaryOutcomeRollingMeanTransformer


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game",
        team_id="team",
        player_id="player",
        start_date="start_date",
    )


@pytest.mark.parametrize("use_column_names", [True, False])
@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("min_periods", [10, 1])
def test_binary_granularity_rolling_mean_transformer(
    df, column_names, min_periods, use_column_names
):
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
    except Exception:
        expected_df = historical_df.clone()

    if use_column_names:
        transformer = BinaryOutcomeRollingMeanTransformer(
            features=["score_difference"],
            binary_column="won",
            window=10,
            min_periods=min_periods,
            granularity=["player"],
            prob_column="prob",
        )
        transformed_data = transformer.fit_transform(df=historical_df, column_names=column_names)
    else:
        transformer = BinaryOutcomeRollingMeanTransformer(
            features=["score_difference"],
            binary_column="won",
            window=10,
            min_periods=min_periods,
            granularity=["player"],
            prob_column="prob",
            match_id_column=column_names.match_id,
            update_column=column_names.update_match_id,
        )
        transformed_data = transformer.fit_transform(df=historical_df, column_names=None)

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
        assert_frame_equal(
            transformed_data,
            expected_df.select(transformed_data.columns),
            check_dtype=False,
        )


@pytest.mark.parametrize("use_column_names", [True, False])
def test_binary_granularity_rolling_mean_transformer_update_id_differ_from_game_id(
    column_names, use_column_names
):
    column_names.update_match_id = "series_id"
    historical_df = pd.DataFrame(
        {
            "player": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "game": ["1", "1", "2", "2", "3", "3", "4", "4"],
            "series_id": ["1", "1", "1", "1", "1", "1", "2", "2"],
            "score_difference": [10, 0, -15, 15, 20, -20, 2, 2],
            "won": [1, 0, 0, 1, 1, 0, 0, 1],
            "start_date": [
                pd.to_datetime("2023-01-01 15:00:00"),
                pd.to_datetime("2023-01-01 15:00:00"),
                pd.to_datetime("2023-01-01 16:00:00"),
                pd.to_datetime("2023-01-01 16:00:00"),
                pd.to_datetime("2023-01-01 17:00:00"),
                pd.to_datetime("2023-01-01 17:00:00"),
                pd.to_datetime("2023-01-02 17:00:00"),
                pd.to_datetime("2023-01-02 17:00:00"),
            ],
            "team": ["1", "2", "1", "2", "1", "2", "1", "2"],
            "prob": [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.8, 0.2],
        }
    )

    expected_df = historical_df.copy()

    if use_column_names:
        transformer = BinaryOutcomeRollingMeanTransformer(
            features=["score_difference"],
            binary_column="won",
            window=10,
            min_periods=1,
            granularity=["player"],
            prob_column="prob",
        )
        transformed_data = transformer.fit_transform(df=historical_df, column_names=column_names)
    else:
        transformer = BinaryOutcomeRollingMeanTransformer(
            features=["score_difference"],
            binary_column="won",
            window=10,
            min_periods=1,
            granularity=["player"],
            prob_column="prob",
            match_id_column=column_names.match_id,
            update_column=column_names.update_match_id,
        )
        transformed_data = transformer.fit_transform(df=historical_df, column_names=None)

    expected_df[transformer.features_out[0]] = [
        None,
        None,
        None,
        None,
        None,
        None,
        15,
        15,
    ]
    expected_df[transformer.features_out[1]] = [
        None,
        None,
        None,
        None,
        None,
        None,
        -15,
        -10,
    ]
    expected_df[transformer.features_out[2]] = [
        None,
        None,
        None,
        None,
        None,
        None,
        0.8 * 15 + 0.2 * -15,
        0.2 * 15 + 0.8 * -10,
    ]

    pd.testing.assert_frame_equal(
        expected_df, transformed_data[expected_df.columns], check_dtype=False
    )


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
@pytest.mark.parametrize("min_periods", [10, 1])
def test_binary_granularity_rolling_mean_generate_future(df, column_names, min_periods):
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
    except Exception:
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

    historical_df = transformer.fit_transform(historical_df, column_names=column_names)
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
    expected_historical_df[transformer.features_out[1]] = (
        [None] * 6 + [-10] * 2 + [None] * 2 + [-12.5, -15]
    )
    expected_historical_df[transformer.features_out[2]] = (
        [None] * 6 + [10, 10] + [None] * 2 + [12.5, 12.5]
    )

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

    expected_historical_df[transformer.features_out[4]] = [float("nan")] * 12
    expected_historical_df[transformer.features_out[5]] = [float("nan")] * 12

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

    future_df = df(
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
    except Exception:
        expected_future_df = future_df.to_pandas()

    future_df = transformer.future_transform(future_df)
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
    expected_future_df[transformer.features_out[5]] = [
        2 * 0.4 - 15 * 0.6,
        12.5 * 0.6 + 0.4 * -2,
        2 * 0.3 - 15 * 0.7,
        12.5 * 0.7 - 2 * 0.3,
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
def test_binary_granularity_rolling_mean_generate_historical_opponent(df, column_names):
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
    except Exception:
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

    df = rolling_mean_transformation.fit_transform(df, column_names=column_names)

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
    expected_historical_df[rolling_mean_transformation.features_out[5]] = [
        None,
        None,
        None,
        None,
        -10 * 0.6 + 0.4 * -5,
        10 * 0.6 + 0.4 * 5,
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
