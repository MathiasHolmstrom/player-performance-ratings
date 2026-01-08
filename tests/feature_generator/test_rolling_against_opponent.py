import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spforge import ColumnNames
from spforge.feature_generator._rolling_against_opponent import (
    RollingAgainstOpponentTransformer,
)


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game",
        team_id="team",
        player_id="player",
        start_date="start_date",
        participation_weight="participation_weight",
    )


@pytest.mark.parametrize("use_column_names", [True, False])
@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_rolling_mean_transform_historical(column_names, df, use_column_names):
    data = df(
        {
            "game": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3],
            "points": [1, 2, 2, 4, 3, 2, 4, 6, 6, 2, 6, 1, 2, 0],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": [1, 1, 1, 2, 2, 2, 1, 3, 1, 1, 1, 2, 2, 2],
            "player": [1, 2, 3, 4, 5, 6, 1, 7, 1, 2, 3, 4, 5, 6],
            "position": [
                "G",
                "G",
                "F",
                "G",
                "G",
                "F",
                "G",
                "F",
                "G",
                "G",
                "F",
                "G",
                "G",
                "F",
            ],
        }
    )

    if use_column_names:
        rolling_mean = RollingAgainstOpponentTransformer(
            granularity=["position"],
            features=["points"],
            window=20,
        )
        transformed_data = rolling_mean.fit_transform(data, column_names=column_names)

    else:
        rolling_mean = RollingAgainstOpponentTransformer(
            granularity=["position"],
            features=["points"],
            window=20,
            match_id_column="game",
            team_column="team",
        )
        transformed_data = rolling_mean.fit_transform(data)
    expected_values = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        2.0,
        1.5,
        1.5,
        2.0,
        3.5,
        3.5,
        4.0,
    ]
    if isinstance(transformed_data, pd.DataFrame):
        expected_data = data.copy()
        expected_data = expected_data.assign(**{rolling_mean.features_out[0]: expected_values})
        pd.testing.assert_frame_equal(
            transformed_data, expected_data[transformed_data.columns], check_dtype=False
        )
    else:
        expected_data = data.clone()
        expected_data = expected_data.with_columns(
            pl.Series(
                rolling_mean.features_out[0],
                expected_values,
            )
        )

        assert_frame_equal(
            transformed_data,
            expected_data.select(transformed_data.columns),
            check_dtype=False,
        )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_rolling_mean_transform_future(column_names, df):
    historical_df = df(
        {
            "game": [1, 1, 1, 1, 1, 1, 2, 2],
            "points": [1, 2, 2, 4, 3, 2, 4, 5],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
            ],
            "team": [1, 1, 1, 2, 2, 2, 1, 3],
            "player": [1, 2, 5, 7, 3, 4, 1, 6],
            "position": ["G", "G", "F", "G", "G", "F", "G", "F"],
        }
    )

    future_df = df(
        {
            "game": [3, 3],
            "points": [None, None],
            "start_date": [pd.to_datetime("2023-01-04"), pd.to_datetime("2023-01-04")],
            "team": [1, 2],
            "player": [2, 4],
            "position": ["G", "F"],
        }
    )

    rolling_mean = RollingAgainstOpponentTransformer(
        granularity=["position"],
        features=["points"],
        window=20,
    )
    _ = rolling_mean.fit_transform(historical_df, column_names=column_names)

    transformed_future_data = rolling_mean.future_transform(future_df)

    if isinstance(transformed_future_data, pd.DataFrame):
        expected_data = transformed_future_data.copy()
        expected_data = expected_data.assign(**{rolling_mean.features_out[0]: [1.5, 3.5]})
        pd.testing.assert_frame_equal(
            transformed_future_data,
            expected_data[transformed_future_data.columns],
            check_dtype=False,
        )
    else:
        expected_data = transformed_future_data.clone()
        expected_data = expected_data.with_columns(
            pl.Series(rolling_mean.features_out[0], [1.5, 3.5])
        )

        assert_frame_equal(
            transformed_future_data,
            expected_data.select(transformed_future_data.columns),
            check_dtype=False,
        )
