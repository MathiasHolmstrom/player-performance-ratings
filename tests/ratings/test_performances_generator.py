import math

import pandas as pd
import polars as pl
import pytest
from deepdiff import DeepDiff

from spforge import ColumnNames

from spforge.transformers.fit_transformers import (
    SymmetricDistributionTransformer,
    PartialStandardScaler,
    MinMaxTransformer,
)
from spforge.transformers.fit_transformers._performance_manager import (
    ColumnWeight,
    create_performance_scalers_transformers,
    PerformanceWeightsManager,
    PerformanceManager,
)


def test_auto_create_pre_transformers():
    pre_transformations = create_performance_scalers_transformers(
        transformer_names=["symmetric", "partial_standard_scaler", "min_max"],
        pre_transformers=[],
        features=["kills", "deaths"],
    )

    expected_pre_transformations = [
        SymmetricDistributionTransformer(features=["kills", "deaths"], prefix=""),
        PartialStandardScaler(
            features=["kills", "deaths"],
            ratio=1,
            max_value=9999,
            target_mean=0,
            prefix="",
        ),
        MinMaxTransformer(features=["kills", "deaths"]),
    ]

    diff = DeepDiff(pre_transformations, expected_pre_transformations)
    assert diff == {}


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_performance_weights_manager_fit_transform(df):
    column_names = [
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        ),
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        ),
    ]

    data = df(
        {
            column_names[0].match_id: [1, 1, 2, 2],
            column_names[0].team_id: [1, 2, 1, 2],
            column_names[0].player_id: [1, 2, 1, 2],
            column_names[0].start_date: [
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
            "points_difference": [5, 1, 3, 3],
            "won": [1, 0, 1, 0],
        }
    )

    weights = [
        ColumnWeight(name="won", weight=0.5),
        ColumnWeight(name="points_difference", weight=0.5),
    ]

    performances_generator = PerformanceWeightsManager(weights=weights, prefix="")

    df_with_performances = performances_generator.fit_transform(data)
    if isinstance(data, pd.DataFrame):
        expected_df_with_performances = data.copy()
    else:
        expected_df_with_performances = data.to_pandas()
        df_with_performances = df_with_performances.to_pandas()

    expected_df_with_performances[performances_generator.features_out[0]] = [
        1,
        0,
        0.75,
        0.25,
    ]

    pd.testing.assert_frame_equal(
        df_with_performances,
        expected_df_with_performances[df_with_performances.columns],
        check_dtype=False,
        check_like=True,
    )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_performance_manager_fit_transform_and_transform(df):
    column_names = [
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        )
    ]

    data = df(
        {
            column_names[0].match_id: [1, 1, 2, 2],
            column_names[0].team_id: [1, 2, 1, 2],
            column_names[0].player_id: [1, 2, 1, 2],
            column_names[0].start_date: [
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
            "points_difference": [5, 1, 3, 3],
        }
    )

    performances_generator = PerformanceManager(
        prefix="", features=["points_difference"], max_value=1, min_value=0
    )

    df_with_performances = performances_generator.fit_transform(data)
    if isinstance(data, pd.DataFrame):
        expected_df_with_performances = data.copy()
    else:
        expected_df_with_performances = data.to_pandas()
        df_with_performances = df_with_performances.to_pandas()

    expected_df_with_performances[performances_generator.features_out[0]] = [
        1,
        0,
        0.5,
        0.5,
    ]

    pd.testing.assert_frame_equal(
        df_with_performances,
        expected_df_with_performances[df_with_performances.columns],
        check_dtype=False,
        check_like=True,
    )

    transform_data = df(
        {
            column_names[0].match_id: [3, 3],
            column_names[0].team_id: [1, 2],
            column_names[0].player_id: [1, 2],
            column_names[0].start_date: [
                pd.to_datetime("2021-01-03"),
                pd.to_datetime("2021-01-3"),
            ],
            "points_difference": [4, 3],
        }
    )

    transformed_data_with_performances = performances_generator.transform(
        transform_data
    )
    if isinstance(data, pl.DataFrame):
        transformed_data_with_performances = (
            transformed_data_with_performances.to_pandas()
        )

    assert math.isclose(
        transformed_data_with_performances[
            performances_generator.features_out[0]
        ].tolist()[1],
        0.5,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    assert (
        transformed_data_with_performances[
            performances_generator.features_out[0]
        ].tolist()[0]
        > 0.7
    )
    assert (
        transformed_data_with_performances[
            performances_generator.features_out[0]
        ].tolist()[0]
        < 0.8
    )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_performances_manager_with_only_partial_standard_scaler(df):
    column_names = [
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        )
    ]

    data = df(
        {
            column_names[0].match_id: [1, 1, 2, 2],
            column_names[0].team_id: [1, 2, 1, 2],
            column_names[0].player_id: [1, 2, 1, 2],
            column_names[0].start_date: [
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
            "points_difference": [5, 1, 3, 3],
        }
    )

    performances_generator = PerformanceManager(
        min_value=-4.3,
        max_value=4.3,
        prefix="",
        features=["points_difference"],
        transformer_names=["partial_standard_scaler"],
    )

    df_with_performances = performances_generator.fit_transform(data)
    if isinstance(data, pl.DataFrame):

        df_with_performances = df_with_performances.to_pandas()

    assert df_with_performances[performances_generator.features_out[0]].tolist()[0] > 0
    assert df_with_performances[performances_generator.features_out[0]].tolist()[1] < 0
    assert (
        df_with_performances[performances_generator.features_out[0]].tolist()[2] == 0.0
    )
    assert (
        df_with_performances[performances_generator.features_out[0]].tolist()[3] == 0.0
    )

    transform_data = df(
        {
            column_names[0].match_id: [3, 3],
            column_names[0].team_id: [1, 2],
            column_names[0].player_id: [1, 2],
            column_names[0].start_date: [
                pd.to_datetime("2021-01-03"),
                pd.to_datetime("2021-01-3"),
            ],
            "points_difference": [4, 3],
        }
    )

    transformed_data_with_performances = performances_generator.transform(
        transform_data
    )
    if isinstance(data, pl.DataFrame):
        transformed_data_with_performances = (
            transformed_data_with_performances.to_pandas()
        )

    assert (
        transformed_data_with_performances[
            performances_generator.features_out[0]
        ].tolist()[1]
        == 0
    )
    assert (
        transformed_data_with_performances[
            performances_generator.features_out[0]
        ].tolist()[0]
        > 0.3
    )
    assert (
        transformed_data_with_performances[
            performances_generator.features_out[0]
        ].tolist()[0]
        < 0.8
    )
