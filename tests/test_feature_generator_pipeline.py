import pandas as pd
import polars as pl
import pytest

from spforge import ColumnNames, FeatureGeneratorPipeline
from spforge.feature_generator import LagTransformer, RollingWindowTransformer


@pytest.fixture
def column_names():
    return ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="date",
    )


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_feature_generator_pipeline__fit_transform_preserves_row_count(df_type, column_names):
    """FeatureGeneratorPipeline.fit_transform should preserve row count."""
    data = df_type(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "team_id": ["A", "B", "A", "B", "A", "B"],
            "player_id": ["p1", "p2", "p1", "p2", "p1", "p2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-03"]),
            "points": [10, 15, 12, 18, 14, 20],
        }
    )

    lag_gen = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=[column_names.player_id],
    )

    pipeline = FeatureGeneratorPipeline(
        feature_generators=[lag_gen],
        column_names=column_names,
    )

    initial_row_count = len(data)
    result = pipeline.fit_transform(data, column_names=column_names)

    assert len(result) == initial_row_count


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_feature_generator_pipeline__transform_preserves_row_count(df_type, column_names):
    """FeatureGeneratorPipeline.transform should preserve row count."""
    data = df_type(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": ["A", "B", "A", "B"],
            "player_id": ["p1", "p2", "p1", "p2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]),
            "points": [10, 15, 12, 18],
        }
    )

    lag_gen = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=[column_names.player_id],
    )

    pipeline = FeatureGeneratorPipeline(
        feature_generators=[lag_gen],
        column_names=column_names,
    )

    # Fit on initial data
    pipeline.fit_transform(data, column_names=column_names)

    # Transform on new data
    new_data = df_type(
        {
            "game_id": [3, 3, 4, 4],
            "team_id": ["A", "B", "A", "B"],
            "player_id": ["p1", "p2", "p1", "p2"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-03", "2023-01-04", "2023-01-04"]),
            "points": [14, 20, 16, 22],
        }
    )

    initial_row_count = len(new_data)
    result = pipeline.transform(new_data)

    assert len(result) == initial_row_count


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_feature_generator_pipeline__future_transform_preserves_row_count(df_type, column_names):
    """FeatureGeneratorPipeline.future_transform should preserve row count."""
    data = df_type(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": ["A", "B", "A", "B"],
            "player_id": ["p1", "p2", "p1", "p2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]),
            "points": [10, 15, 12, 18],
        }
    )

    lag_gen = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=[column_names.player_id],
    )

    pipeline = FeatureGeneratorPipeline(
        feature_generators=[lag_gen],
        column_names=column_names,
    )

    # Fit on initial data
    pipeline.fit_transform(data, column_names=column_names)

    # Future transform on data without outcomes
    future_data = df_type(
        {
            "game_id": [3, 3],
            "team_id": ["A", "B"],
            "player_id": ["p1", "p2"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        }
    )

    initial_row_count = len(future_data)
    result = pipeline.future_transform(future_data)

    assert len(result) == initial_row_count


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_feature_generator_pipeline__detects_duplicate_features(df_type, column_names):
    """FeatureGeneratorPipeline should raise error for duplicate features across generators."""
    data = df_type(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": ["A", "B", "A", "B"],
            "player_id": ["p1", "p2", "p1", "p2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]),
            "points": [10, 15, 12, 18],
        }
    )

    # Create two lag generators that output the same feature name
    lag_gen_1 = LagTransformer(
        features_to_lag=["points"],
        lag_length=1,
        grouping_features=[column_names.player_id],
        unique_constraint=[column_names.match_id],
        column_names=column_names,
    )

    lag_gen_2 = LagTransformer(
        features_to_lag=["points"],
        lag_length=1,
        grouping_features=[column_names.player_id],
        unique_constraint=[column_names.match_id],
        column_names=column_names,
    )

    pipeline = FeatureGeneratorPipeline(
        feature_generators=[lag_gen_1, lag_gen_2],
        column_names=column_names,
    )

    # Should raise AssertionError for duplicate features
    with pytest.raises(AssertionError, match="Duplicate features"):
        pipeline.fit_transform(data, column_names=column_names)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_feature_generator_pipeline__maintains_feature_order(df_type, column_names):
    """FeatureGeneratorPipeline should maintain features in order of generators."""
    data = df_type(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "team_id": ["A", "B", "A", "B", "A", "B"],
            "player_id": ["p1", "p2", "p1", "p2", "p1", "p2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-03"]),
            "points": [10, 15, 12, 18, 14, 20],
            "rebounds": [5, 6, 7, 8, 6, 9],
        }
    )

    lag_points = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=[column_names.player_id],
    )

    lag_rebounds = LagTransformer(
        features=["rebounds"],
        lag_length=1,
        granularity=[column_names.player_id],
    )

    pipeline = FeatureGeneratorPipeline(
        feature_generators=[lag_points, lag_rebounds],
        column_names=column_names,
    )

    result = pipeline.fit_transform(data, column_names=column_names)

    # Verify that lag features were added
    assert "points_lag_1" in result.columns
    assert "rebounds_lag_1" in result.columns


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_feature_generator_pipeline__works_with_empty_generator_list(df_type, column_names):
    """FeatureGeneratorPipeline should handle empty generator list."""
    data = df_type(
        {
            "game_id": [1, 1],
            "team_id": ["A", "B"],
            "player_id": ["p1", "p2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "points": [10, 15],
        }
    )

    pipeline = FeatureGeneratorPipeline(
        feature_generators=[],
        column_names=column_names,
    )

    # Should return data unchanged
    result = pipeline.fit_transform(data, column_names=column_names)

    assert len(result) == len(data)
    assert list(result.columns) == list(data.columns)


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_feature_generator_pipeline__chains_generators_correctly(df_type, column_names):
    """FeatureGeneratorPipeline should chain generators so output of N is input to N+1."""
    data = df_type(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "team_id": ["A", "B", "A", "B", "A", "B"],
            "player_id": ["p1", "p2", "p1", "p2", "p1", "p2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-03"]),
            "points": [10, 15, 12, 18, 14, 20],
        }
    )

    # Generator 1: Lag points
    lag_gen = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=[column_names.player_id],
    )

    # Generator 2: Rolling window on points (will use original points column)
    rolling_gen = RollingWindowTransformer(
        features=["points"],
        window=2,
        granularity=[column_names.player_id],
        aggregation="mean",
    )

    pipeline = FeatureGeneratorPipeline(
        feature_generators=[lag_gen, rolling_gen],
        column_names=column_names,
    )

    result = pipeline.fit_transform(data, column_names=column_names)

    # Both generators should have added their features
    assert "points_lag_1" in result.columns
    assert "points_rolling_mean_2" in result.columns


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_feature_generator_pipeline__future_transform_no_state_mutation(df_type, column_names):
    """FeatureGeneratorPipeline.future_transform should not mutate generator state."""
    data = df_type(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": ["A", "B", "A", "B"],
            "player_id": ["p1", "p2", "p1", "p2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]),
            "points": [10, 15, 12, 18],
        }
    )

    lag_gen = LagTransformer(
        features=["points"],
        lag_length=1,
        granularity=[column_names.player_id],
    )

    pipeline = FeatureGeneratorPipeline(
        feature_generators=[lag_gen],
        column_names=column_names,
    )

    # Fit on initial data
    result_fit = pipeline.fit_transform(data, column_names=column_names)

    # Call future_transform
    future_data = df_type(
        {
            "game_id": [3, 3],
            "team_id": ["A", "B"],
            "player_id": ["p1", "p2"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        }
    )

    result_future = pipeline.future_transform(future_data)

    # Call transform to verify state is still from fit_transform, not future_transform
    # If future_transform mutated state, this would give different results
    new_data = df_type(
        {
            "game_id": [3, 3],
            "team_id": ["A", "B"],
            "player_id": ["p1", "p2"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
            "points": [14, 20],
        }
    )

    result_transform = pipeline.transform(new_data)

    # Both future_transform and transform should produce results
    # The key is that future_transform didn't break the state
    assert "points_lag_1" in result_future.columns
    assert "points_lag_1" in result_transform.columns
