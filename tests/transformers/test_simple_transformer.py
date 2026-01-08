import pandas as pd
import polars as pl
import pytest

from spforge.transformers._simple_transformer import (
    AggregatorTransformer,
    NormalizerToColumnTransformer,
)


@pytest.mark.parametrize("df_type", ["pd", "pl"])
def test_aggregator_transformer_granularity_sum(df_type):
    df = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "team_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "player_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "minutes": [10, 20, 30, 40, 40, 30, 30, 40],
        }
    )
    expected_df = df.to_pandas()
    if df_type == "pd":
        df = df.to_pandas()

    transformer = AggregatorTransformer(
        columns=["minutes"],
        granularity=["game_id", "team_id"],
        column_to_alias={"minutes": "minutes_sum"},
        aggregator="sum",
    )
    transformed_df = transformer.transform(df)
    # Convert to pandas for comparison
    if isinstance(transformed_df, pl.DataFrame):
        transformed_df = transformed_df.to_pandas()

    expected_df["minutes_sum"] = [30, 30, 70, 70, 70, 70, 70, 70]

    pd.testing.assert_frame_equal(transformed_df, expected_df, check_dtype=False)


@pytest.mark.parametrize("df_type", ["pd", "pl"])
def test_aggregator_transformer_sum(df_type):
    df = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "team_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "player_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "minutes": [10, 20, 30, 40, 40, 30, 30, 40],
        }
    )
    expected_df = df.to_pandas()
    if df_type == "pd":
        df = df.to_pandas()

    transformer = AggregatorTransformer(
        columns=["minutes"],
        column_to_alias={"minutes": "minutes_sum"},
        aggregator="sum",
    )
    transformed_df = transformer.transform(df)
    # Convert to pandas for comparison
    if isinstance(transformed_df, pl.DataFrame):
        transformed_df = transformed_df.to_pandas()

    expected_df["minutes_sum"] = sum([10, 20, 30, 40, 40, 30, 30, 40])

    pd.testing.assert_frame_equal(transformed_df, expected_df, check_dtype=False)


@pytest.mark.parametrize("df_type", ["pd", "pl"])
def test_aggregator_transformer_mean(df_type):
    df = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "team_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "player_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "minutes": [10, 20, 30, 40, 40, 30, 30, 40],
            "minutes_sum": [30, 30, 70, 70, 70, 70, 70, 70],
        }
    )
    expected_df = df.to_pandas()

    if df_type == "pd":
        df = df.to_pandas()

    transformer = AggregatorTransformer(
        columns=["minutes", "minutes_sum"],
        column_to_alias={"minutes": "minutes_mean", "minutes_sum": "minutes_sum_mean"},
        aggregator="mean",
    )
    transformed_df = transformer.transform(df)
    # Convert to pandas for comparison
    if isinstance(transformed_df, pl.DataFrame):
        transformed_df = transformed_df.to_pandas()
    expected_df["minutes_mean"] = expected_df["minutes"].mean()
    expected_df["minutes_sum_mean"] = expected_df["minutes_sum"].mean()

    pd.testing.assert_frame_equal(transformed_df, expected_df, check_dtype=False)


@pytest.mark.parametrize("df_type", ["pd", "pl"])
def test_normalizer_to_column_transformer(df_type):
    df = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "team_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "player_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "predicted_minutes": [10, 20, 30, 40, 40, 30, 30, 40],
            "minutes_sum_mean": [50, 50, 50, 50, 50, 50, 50, 50],
        }
    )
    expected_df = df.to_pandas()
    if df_type == "pd":
        df = df.to_pandas()

    transformer = NormalizerToColumnTransformer(
        column="predicted_minutes",
        granularity=["game_id", "team_id"],
        normalize_to_column="minutes_sum_mean",
    )

    transformed_df = transformer.transform(df)
    # Convert to pandas for comparison
    if isinstance(transformed_df, pl.DataFrame):
        transformed_df = transformed_df.to_pandas()
    game_id_1_team_1_ratio = 50 / 30
    game_id_1_team_2_ratio = 50 / 70
    game_id_2_team_1_ratio = 50 / 70
    game_id_2_team_2_ratio = 50 / 70

    expected_df["predicted_minutes"] = [
        10 * game_id_1_team_1_ratio,
        20 * game_id_1_team_1_ratio,
        30 * game_id_1_team_2_ratio,
        40 * game_id_1_team_2_ratio,
        40 * game_id_2_team_1_ratio,
        30 * game_id_2_team_1_ratio,
        30 * game_id_2_team_2_ratio,
        40 * game_id_2_team_2_ratio,
    ]

    pd.testing.assert_frame_equal(transformed_df, expected_df, check_dtype=False)
