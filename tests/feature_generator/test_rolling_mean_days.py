import pandas as pd
import pytest

from spforge import ColumnNames
from spforge.feature_generator import RollingMeanDaysTransformer


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
def test_rolling_mean_days_transform_historical(column_names, use_column_names):
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

    if use_column_names:
        rolling_mean_transformation = RollingMeanDaysTransformer(
            features=["points", "points2"],
            days=2,
            granularity=["player"],
            add_count=True,
        )
        transformed_df = rolling_mean_transformation.fit_transform(df, column_names=column_names)
    else:
        rolling_mean_transformation = RollingMeanDaysTransformer(
            features=["points", "points2"],
            days=2,
            granularity=["player"],
            add_count=True,
            date_column=column_names.start_date,
            update_column=column_names.update_match_id,
        )
        transformed_df = rolling_mean_transformation.fit_transform(df, column_names=None)

    expected_df = original_df.assign(
        **{
            rolling_mean_transformation.features_out[0]: [None, None, None, 1, None],
            rolling_mean_transformation.features_out[1]: [None, None, None, 2, None],
            rolling_mean_transformation.features_out[2]: [0, 0, 0, 2, 0],
        }
    )

    pd.testing.assert_frame_equal(transformed_df, expected_df, check_like=True, check_dtype=False)


@pytest.mark.parametrize("use_column_names", [True, False])
def test_rolling_mean_days_update_id_different_from_game_id(
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
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
            ],
            "team": [1, 1, 1, 1],
            "series_id": [1, 1, 2, 3],
        }
    )
    expected_df = historical_df.copy()
    if column_names:
        transformer = RollingMeanDaysTransformer(
            add_count=True,
            features=["points"],
            days=2,
            granularity=["player"],
            column_names=column_names,
        )

    else:
        transformer = RollingMeanDaysTransformer(
            add_count=True,
            features=["points"],
            days=2,
            granularity=["player"],
            update_column=column_names.update_match_id,
        )

    transformed_df = transformer.fit_transform(historical_df)

    expected_df = expected_df.assign(
        **{
            transformer.features_out[0]: [None, None, 1.5, 3],
            transformer.features_out[1]: [0, 0, 2, 1],
        }
    )

    pd.testing.assert_frame_equal(transformed_df, expected_df, check_like=True, check_dtype=False)


@pytest.mark.parametrize("use_column_names", [True, False])
def test_rolling_mean_days_transform_historical_40_days(use_column_names, column_names):
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
    if use_column_names:
        rolling_mean_transformation = RollingMeanDaysTransformer(
            features=["points"],
            days=40,
            granularity=["player"],
        )

        transformed_df = rolling_mean_transformation.fit_transform(df, column_names=column_names)
    else:
        rolling_mean_transformation = RollingMeanDaysTransformer(
            features=["points"],
            days=40,
            granularity=["player"],
            update_column=column_names.update_match_id,
        )

        transformed_df = rolling_mean_transformation.fit_transform(df, column_names=column_names)

    expected_df = original_df.assign(
        **{
            rolling_mean_transformation.features_out[0]: [None, 1, 1, None, 1.5, 3],
        }
    )

    pd.testing.assert_frame_equal(transformed_df, expected_df, check_like=True, check_dtype=False)


def test_rolling_mean_days_transform_future_40_days_update_id_differs_from_match_id(
    column_names,
):
    column_names.update_match_id = "series_id"
    historical_df = pd.DataFrame(
        {
            "player": ["a", "b", "c", "d"] * 2,
            "game": [1, 1, 1, 1, 2, 2, 2, 2],
            "points": [1, 2, 3, 4, 5, 6, 7, 8],
            "series_id": [1, 1, 1, 1, 1, 1, 1, 1],
            "start_date": [
                pd.to_datetime("2023-01-01 15:00:00"),
                pd.to_datetime("2023-01-01 15:00:00"),
                pd.to_datetime("2023-01-01 15:00:00"),
                pd.to_datetime("2023-01-01 15:00:00"),
                pd.to_datetime("2023-01-01 16:00:00"),
                pd.to_datetime("2023-01-01 16:00:00"),
                pd.to_datetime("2023-01-01 16:00:00"),
                pd.to_datetime("2023-01-01 16:00:00"),
            ],
            "team": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )

    future_df = pd.DataFrame(
        {
            "player": ["a", "b", "c", "d"] * 2,
            "game": [3, 3, 3, 3, 4, 4, 4, 4],
            "points": [9, 10, 11, 12] * 2,
            "series_id": [3] * 8,
            "start_date": [
                pd.to_datetime("2023-01-02 17:00:00"),
                pd.to_datetime("2023-01-02 17:00:00"),
                pd.to_datetime("2023-01-02 17:00:00"),
                pd.to_datetime("2023-01-02 17:00:00"),
                pd.to_datetime("2023-01-02 18:00:00"),
                pd.to_datetime("2023-01-02 18:00:00"),
                pd.to_datetime("2023-01-02 18:00:00"),
                pd.to_datetime("2023-01-02 18:00:00"),
            ],
            "team": [1, 1, 2, 2] * 2,
        }
    )

    original_df = future_df.copy()

    rolling_mean_transformation = RollingMeanDaysTransformer(
        features=["points"],
        days=40,
        granularity=["player"],
    )

    _ = rolling_mean_transformation.fit_transform(historical_df, column_names=column_names)
    transformed_future_df = rolling_mean_transformation.future_transform(future_df)

    expected_df = original_df.assign(
        **{
            rolling_mean_transformation.features_out[0]: [3, 4, 5, 6] * 2,
        }
    )

    pd.testing.assert_frame_equal(
        transformed_future_df, expected_df, check_like=True, check_dtype=False
    )


def test_rolling_mean_days_generate_historical_opponent(column_names):
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

    transformed_df = rolling_mean_transformation.fit_transform(df, column_names=column_names)

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

    pd.testing.assert_frame_equal(transformed_df, expected_df, check_like=True, check_dtype=False)


def test_rolling_mean_days_transformer_future_transform(column_names):
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
    historical_df = transformer.fit_transform(historical_df, column_names=column_names)
    expected_historical_df = expected_historical_df.assign(
        **{
            transformer.features_out[0]: [None, None, 1, 2],
            transformer.features_out[1]: [None, None, 2, 1],
            f"{transformer.prefix}_count10": [0, 0, 1, 1],
            f"{transformer.prefix}_count10_opponent": [0, 0, 1, 1],
        }
    )

    pd.testing.assert_frame_equal(
        historical_df, expected_historical_df, check_like=True, check_dtype=False
    )

    expected_df = future_df.copy()

    transformed_future_df = transformer.future_transform(df=future_df)

    expected_df = expected_df.assign(
        **{
            transformer.features_out[0]: [2, 3, 2, 3],
            transformer.features_out[1]: [3, 2, 3, 2],
            f"{transformer.prefix}_count10": [2, 2, 2, 2],
            f"{transformer.prefix}_count10_opponent": [2, 2, 2, 2],
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
    historical_df = transformer.fit_transform(historical_df, column_names=column_names)

    expected_historical_df = expected_historical_df.assign(
        **{
            transformer.features_out[0]: [None, None, None, None],
            transformer.features_out[1]: [None, None, None, None],
            f"{transformer.prefix}_count10": [0, 0, 0, 0],
            f"{transformer.prefix}_count10_opponent": [0, 0, 0, 0],
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

    transformed_future_df = transformer.future_transform(df=future_df)

    expected_df = expected_df.assign(
        **{
            transformer.features_out[0]: [3, 2, 3, 2],
            transformer.features_out[1]: [2, 3, 2, 3],
            f"{transformer.prefix}_count10": [1, 1, 1, 1],
            f"{transformer.prefix}_count10_opponent": [1, 1, 1, 1],
        }
    )

    pd.testing.assert_frame_equal(
        transformed_future_df, expected_df, check_like=True, check_dtype=False
    )


@pytest.mark.parametrize("use_column_names", [True, False])
def test_rolling_mean_days_transform_historical_granularity_differs_from_input_granularity(
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
        transformer = RollingMeanDaysTransformer(
            features=["points"],
            days=10,
            granularity=["league", "position"],
            unique_constraint=[column_names.match_id, column_names.team_id, "position"],
        )
    else:
        transformer = RollingMeanDaysTransformer(
            features=["points"],
            days=10,
            granularity=["league", "position"],
            update_column=column_names.match_id,
            date_column=column_names.start_date,
        )
        column_names = None

    expected_df = data.copy()
    transformed_df = transformer.fit_transform(df=data, column_names=column_names)

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
    pd.testing.assert_frame_equal(transformed_df, expected_df, check_like=True, check_dtype=False)


def test_rolling_mean__days_transform_future_granularity_differs_from_input_granularity(
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

    transformer = RollingMeanDaysTransformer(
        features=["points"],
        days=10,
        granularity=["league", "position"],
        unique_constraint=[column_names.match_id, column_names.team_id, "position"],
    )

    expected_df = future_df.copy()
    _ = transformer.fit_transform(df=historical_df, column_names=column_names)

    transformed_future_df = transformer.future_transform(future_df)

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
