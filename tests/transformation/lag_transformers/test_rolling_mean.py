import pandas as pd
import pytest
import polars as pl
from polars.testing import assert_frame_equal
from spforge import ColumnNames
from spforge.transformers import RollingMeanTransformer


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game",
        team_id="team",
        player_id="player",
        start_date="start_date",
        participation_weight="participation_weight",
    )


@pytest.mark.parametrize("add_opponent", [True, False])
@pytest.mark.parametrize("use_column_names", [True, False])
@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_rolling_mean_transform_historical_game_team(
    df, column_names, use_column_names, add_opponent
):
    column_names.player_id = None
    data = df(
        {
            "game": [1, 1, 2, 2, 3, 3],
            "points": [1, 2, 3, 2, 4, 5],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
            ],
            "team": [1, 2, 1, 2, 1, 3],
        }
    )
    try:
        original_df = data.copy()
    except:
        original_df = data.clone()

    if not use_column_names:
        if add_opponent:
            return
        rolling_mean_transformation = RollingMeanTransformer(
            features=["points"],
            window=3,
            min_periods=1,
            granularity=["team"],
            add_opponent=add_opponent,
            match_id_update_column=column_names.update_match_id,
        )
        column_names = None

    else:
        rolling_mean_transformation = RollingMeanTransformer(
            features=["points"],
            window=3,
            min_periods=1,
            granularity=["team"],
            add_opponent=add_opponent,
        )

    df_with_rolling_mean = rolling_mean_transformation.transform_historical(
        data, column_names=column_names
    )
    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                f"{rolling_mean_transformation.prefix}_points3": [
                    None,
                    None,
                    1,
                    2,
                    2,
                    None,
                ],
                f"{rolling_mean_transformation.prefix}_points3_opponent": [
                    None,
                    None,
                    2,
                    1,
                    None,
                    2,
                ],
            }
        )
        if not add_opponent:
            expected_df = expected_df.drop(
                columns=[f"{rolling_mean_transformation.prefix}_points3_opponent"]
            )

        pd.testing.assert_frame_equal(
            df_with_rolling_mean, expected_df, check_like=True, check_dtype=False
        )
    else:
        expected_df = original_df.with_columns(
            [
                pl.Series(
                    f"{rolling_mean_transformation.prefix}_points3",
                    [None, None, 1, 2, 2, None],
                    strict=False,
                ),
                pl.Series(
                    f"{rolling_mean_transformation.prefix}_points3_opponent",
                    [None, None, 2, 1, None, 2],
                    strict=False,
                ),
            ]
        )
        if not add_opponent:
            expected_df = expected_df.drop(
                f"{rolling_mean_transformation.prefix}_points3_opponent"
            )

        assert_frame_equal(
            df_with_rolling_mean,
            expected_df.select(df_with_rolling_mean.columns),
            check_dtype=False,
        )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_rolling_mean_transform_historical_player(df, column_names):
    data = df(
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
    try:
        original_df = data.copy()
    except:
        original_df = data.clone()

    rolling_mean_transformation = RollingMeanTransformer(
        features=["points"],
        window=2,
        min_periods=1,
        granularity=["player"],
    )

    df_with_rolling_mean = rolling_mean_transformation.transform_historical(
        data, column_names=column_names
    )
    if isinstance(data, pd.DataFrame):
        expected_df = original_df.assign(
            **{
                f"{rolling_mean_transformation.prefix}_points2": [
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
    else:
        expected_df = original_df.with_columns(
            [
                pl.Series(
                    f"{rolling_mean_transformation.prefix}_points2",
                    [None, None, 1, (3 + 1) / 2],
                    strict=False,
                )
            ]
        )
        pl.testing.assert_frame_equal(
            df_with_rolling_mean,
            expected_df.select(df_with_rolling_mean.columns),
            check_dtype=False,
        )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_rolling_mean_historical_participation_weight(df, column_names):
    historical_df = df(
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
            "participation_weight": [1.0, 0.5, 0.5, 0.5],
        }
    )
    rolling_mean_transformation = RollingMeanTransformer(
        features=["points"],
        window=2,
        min_periods=1,
        granularity=["player"],
        scale_by_participation_weight=True,
    )

    transformed_df = rolling_mean_transformation.transform_historical(
        df=historical_df, column_names=column_names
    )

    if isinstance(historical_df, pl.DataFrame):
        expected_df = historical_df.with_columns(
            pl.Series(
                rolling_mean_transformation.features_out[0],
                [None, None, 1.0, (1 * 1 + 3 * 0.5) / 1.5],
            )
        )
        pl.testing.assert_frame_equal(
            transformed_df,
            expected_df.select(transformed_df.columns),
            check_dtype=False,
        )
    else:
        expected_df = historical_df.copy()
        expected_df[rolling_mean_transformation.features_out[0]] = [
            None,
            None,
            1.0,
            (1 * 1 + 3 * 0.5) / 1.5,
        ]
        pd.testing.assert_frame_equal(
            transformed_df, expected_df, check_like=True, check_dtype=False
        )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_rolling_mean_transform_historical_and_transform_future(df, column_names):
    historical_df = df(
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
    try:
        original_future_df = future_df.copy()
    except:
        original_future_df = future_df.clone()
    rolling_mean_transformation = RollingMeanTransformer(
        features=["points"],
        window=2,
        min_periods=1,
        granularity=["player"],
        add_opponent=True,
    )

    _ = rolling_mean_transformation.transform_historical(
        df=historical_df, column_names=column_names
    )
    transformed_future_df = rolling_mean_transformation.transform_future(future_df)

    if isinstance(future_df, pd.DataFrame):
        expected_df = original_future_df.assign(
            **{
                f"{rolling_mean_transformation.prefix}_points2": [2.5, 2, 2.5, 2],
                rolling_mean_transformation.features_out[1]: [2, 2.5, 2, 2.5],
            }
        )
        pd.testing.assert_frame_equal(
            transformed_future_df, expected_df, check_like=True
        )
    else:
        expected_df = original_future_df.with_columns(
            [
                pl.Series(
                    f"{rolling_mean_transformation.prefix}_points2", [2.5, 2, 2.5, 2]
                ),
                pl.Series(
                    rolling_mean_transformation.features_out[1], [2, 2.5, 2, 2.5]
                ),
            ]
        )
        pl.testing.assert_frame_equal(
            transformed_future_df,
            expected_df.select(transformed_future_df.columns),
            check_dtype=False,
        )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_rolling_mean_historical_transform_team_stat(df, column_names):
    historical_df = df(
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
    try:
        expected_df = historical_df.copy()
    except:
        expected_df = historical_df.clone()

    rolling_mean_transformation = RollingMeanTransformer(
        features=["score_difference"],
        window=2,
        min_periods=1,
        granularity=["team"],
    )

    transformed_data = rolling_mean_transformation.transform_historical(
        historical_df, column_names=column_names
    )
    if isinstance(historical_df, pd.DataFrame):
        expected_df[rolling_mean_transformation.prefix + "_score_difference2"] = [
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
    else:
        expected_df = expected_df.with_columns(
            [
                pl.Series(
                    rolling_mean_transformation.prefix + "_score_difference2",
                    [None, None, None, None, 10, 10, -10, -10],
                )
            ]
        )
        pl.testing.assert_frame_equal(
            transformed_data,
            expected_df.select(transformed_data.columns),
            check_dtype=False,
        )


@pytest.mark.parametrize("use_column_names", [True, False])
def test_rolling_mean_transform_parent_match_id(
    column_names: ColumnNames, use_column_names
):
    column_names = column_names
    column_names.update_match_id = "series_id"
    historical_df = pd.DataFrame(
        {
            "player": ["a", "a", "a", "a"],
            "game": [1, 2, 3, 4],
            "points": [1, 2, 3, 2],
            "points2": [1, 2, 3, 4],
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
    if use_column_names:
        transformer = RollingMeanTransformer(
            features=["points"],
            window=2,
            granularity=["player"],
        )
    else:
        transformer = RollingMeanTransformer(
            features=["points"],
            window=2,
            granularity=["player"],
            match_id_update_column="series_id",
        )
        column_names = None

    transformed_df = transformer.transform_historical(
        historical_df, column_names=column_names
    )

    expected_df = expected_df.assign(
        **{transformer.features_out[0]: [None, None, 1.5, (1.5 + 3) / 2]}
    )
    pd.testing.assert_frame_equal(
        transformed_df, expected_df, check_like=True, check_dtype=False
    )


@pytest.mark.parametrize("use_column_names", [True, False])
def test_rolling_mean_transform_historical_granularity_differs_from_input_granularity(
    column_names: ColumnNames, use_column_names
):
    column_names.player_id = None
    data = pd.DataFrame(
        {
            column_names.start_date: [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
            ],
            column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            column_names.team_id: [1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 5, 5],
            "points": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "position": [
                "pg",
                "sg",
                "pg",
                "sg",
                "pg",
                "sg",
                "pg",
                "sg",
                "pg",
                "sg",
                "pg",
                "sg",
            ],
            "league": ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a"],
        }
    )

    if use_column_names:
        transformer = RollingMeanTransformer(
            features=["points"],
            window=10,
            granularity=["league", "position"],
            unique_constraint=[column_names.match_id, column_names.team_id, "position"],
        )
    else:
        transformer = RollingMeanTransformer(
            features=["points"],
            window=10,
            granularity=["league", "position"],
            match_id_update_column=column_names.match_id,
        )
        column_names = None

    expected_df = data.copy()
    transformed_df = transformer.transform_historical(
        df=data, column_names=column_names
    )

    expected_df = expected_df.assign(
        **{
            transformer.features_out[0]: [
                None,
                None,
                None,
                None,
                2,
                3,
                2,
                3,
                (1 + 3 + 5 + 7) / 4,
                (2 + 4 + 6 + 8) / 4,
                (1 + 3 + 5 + 7) / 4,
                (2 + 4 + 6 + 8) / 4,
            ]
        }
    )
    pd.testing.assert_frame_equal(
        transformed_df, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_transform_future_granularity_differs_from_input_granularity(
    column_names: ColumnNames,
):

    column_names.player_id = None
    historical_df = pd.DataFrame(
        {
            column_names.start_date: [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
            ],
            column_names.match_id: [1, 1, 1, 1, 2, 2, 2, 2],
            column_names.team_id: [1, 1, 2, 2, 3, 3, 4, 4],
            "points": [1, 2, 3, 4, 5, 6, 7, 8],
            "position": ["pg", "sg", "pg", "sg", "pg", "sg", "pg", "sg"],
            "league": ["a", "a", "a", "a", "a", "a", "a", "a"],
        }
    )
    future_df = pd.DataFrame(
        {
            column_names.start_date: [
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
            ],
            column_names.match_id: [3, 3, 3, 3],
            column_names.team_id: [1, 1, 5, 5],
            "points": [9, 10, 11, 12],
            "position": ["pg", "sg", "pg", "sg"],
            "league": ["a", "a", "a", "a"],
        }
    )

    transformer = RollingMeanTransformer(
        features=["points"],
        window=10,
        granularity=["league", "position"],
        unique_constraint=[column_names.match_id, column_names.team_id, "position"],
    )

    expected_df = future_df.copy()
    _ = transformer.transform_historical(df=historical_df, column_names=column_names)

    transformed_future_df = transformer.transform_future(future_df)

    expected_df = expected_df.assign(
        **{
            transformer.features_out[0]: [
                (1 + 3 + 5 + 7) / 4,
                (2 + 4 + 6 + 8) / 4,
                (1 + 3 + 5 + 7) / 4,
                (2 + 4 + 6 + 8) / 4,
            ]
        }
    )
    pd.testing.assert_frame_equal(
        transformed_future_df, expected_df, check_like=True, check_dtype=False
    )
