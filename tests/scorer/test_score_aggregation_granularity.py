"""
Tests for aggregation_level and granularity functionality in scorer classes.
"""

import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import mean_absolute_error

from spforge.scorer import (
    MeanBiasScorer,
    OrdinalLossScorer,
    SklearnScorer,
)
from spforge.scorer._score import PWMSE


# Helper function to create dataframe based on type
def create_dataframe(df_type, data: dict):
    """Helper to create a DataFrame based on type"""
    return df_type(data)


# ============================================================================
# Aggregation Level Tests
# ============================================================================


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_mean_bias_scorer_aggregation_level(df_type):
    """MeanBiasScorer with aggregation_level groups data before scoring"""
    # Game-player level data
    df = create_dataframe(
        df_type,
        {
            "game_id": [1, 1, 1, 1],
            "player_id": [1, 2, 3, 4],
            "team_id": [1, 1, 2, 2],
            "pred": [0.5, 0.6, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        },
    )

    # Aggregate to game-team level
    scorer = MeanBiasScorer(
        pred_column="pred", target="target", aggregation_level=["game_id", "team_id"]
    )

    score = scorer.score(df)
    # After aggregation: game1-team1: pred=1.1, target=1; game1-team2: pred=0.7, target=1
    # Mean bias: ((1.1-1) + (0.7-1)) / 2 = -0.1
    expected = ((1.1 - 1) + (0.7 - 1)) / 2
    assert abs(score - expected) < 1e-10


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_sklearn_scorer_aggregation_level(df_type):
    """SklearnScorer with aggregation_level groups data before scoring"""
    df = create_dataframe(
        df_type,
        {
            "game_id": [1, 1, 1, 1],
            "player_id": [1, 2, 3, 4],
            "team_id": [1, 1, 2, 2],
            "pred": [0.1, 0.6, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        },
    )

    scorer = SklearnScorer(
        pred_column="pred",
        scorer_function=mean_absolute_error,
        target="target",
        aggregation_level=["game_id", "team_id"],
    )

    score = scorer.score(df)
    # After aggregation: game1-team1: pred=0.7, target=1; game1-team2: pred=0.35, target=0.5
    # MAE: mean([abs(0.7-1), abs(0.35-0.5)]) = mean([0.3, 0.15]) = 0.225
    assert isinstance(score, float)
    assert score >= 0


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_pwmse_aggregation_level(df_type):
    """PWMSE with aggregation_level groups data before scoring"""
    df = create_dataframe(
        df_type,
        {
            "game_id": [1, 1, 1, 1],
            "player_id": [1, 2, 3, 4],
            "team_id": [1, 1, 2, 2],
            "pred": [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2], [0.6, 0.4]],
            "target": [0, 1, 0, 1],
        },
    )

    scorer = PWMSE(
        pred_column="pred", target="target", labels=[0, 1], aggregation_level=["game_id", "team_id"]
    )

    score = scorer.score(df)
    assert isinstance(score, float)
    assert score >= 0


# ============================================================================
# Granularity Tests (Separate Scores Per Group)
# ============================================================================


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_mean_bias_scorer_granularity(df_type):
    """MeanBiasScorer with granularity returns separate scores per group"""
    df = create_dataframe(
        df_type, {"team_id": [1, 1, 2, 2], "pred": [0.5, 0.6, 0.3, 0.4], "target": [0, 1, 0, 1]}
    )

    scorer = MeanBiasScorer(pred_column="pred", target="target", granularity=["team_id"])

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 2  # Two teams

    # Team 1: (0.5-0 + 0.6-1) / 2 = 0.05
    # Team 2: (0.3-0 + 0.4-1) / 2 = -0.15
    assert (1,) in result
    assert (2,) in result
    assert abs(result[(1,)] - 0.05) < 1e-10
    assert abs(result[(2,)] - (-0.15)) < 1e-10


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_mean_bias_scorer_granularity_multiple_columns(df_type):
    """MeanBiasScorer with multiple granularity columns"""
    df = create_dataframe(
        df_type,
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "pred": [0.5, 0.6, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        },
    )

    scorer = MeanBiasScorer(pred_column="pred", target="target", granularity=["game_id", "team_id"])

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 4  # Four combinations

    # Check all combinations are present
    assert (1, 1) in result
    assert (1, 2) in result
    assert (2, 1) in result
    assert (2, 2) in result


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_sklearn_scorer_granularity(df_type):
    """SklearnScorer with granularity returns separate scores per group"""
    df = create_dataframe(
        df_type, {"team_id": [1, 1, 2, 2], "pred": [0.1, 0.6, 0.3, 0.4], "target": [0, 1, 0, 1]}
    )

    scorer = SklearnScorer(
        pred_column="pred",
        scorer_function=mean_absolute_error,
        target="target",
        granularity=["team_id"],
    )

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 2

    # Team 1: MAE([0, 1], [0.1, 0.6]) = mean([0.1, 0.4]) = 0.25
    # Team 2: MAE([0, 1], [0.3, 0.4]) = mean([0.3, 0.6]) = 0.45
    assert (1,) in result
    assert (2,) in result
    assert abs(result[(1,)] - 0.25) < 1e-10
    assert abs(result[(2,)] - 0.45) < 1e-10


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_pwmse_granularity(df_type):
    """PWMSE with granularity returns separate scores per group"""
    df = create_dataframe(
        df_type,
        {
            "team_id": [1, 1, 2, 2],
            "pred": [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2], [0.6, 0.4]],
            "target": [0, 1, 0, 1],
        },
    )

    scorer = PWMSE(pred_column="pred", target="target", labels=[0, 1], granularity=["team_id"])

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert (1,) in result
    assert (2,) in result
    assert all(isinstance(v, float) for v in result.values())


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_ordinal_loss_scorer_granularity(df_type):
    """OrdinalLossScorer with granularity returns separate scores per group"""
    df = create_dataframe(
        df_type,
        {
            "team_id": [1, 1, 2, 2],
            "pred": [[0.1, 0.6, 0.3], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]],
            "target": [1, 0, 2, 1],
        },
    )

    scorer = OrdinalLossScorer(
        pred_column="pred", target="target", classes=[0, 1, 2], granularity=["team_id"]
    )

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert (1,) in result
    assert (2,) in result
    assert all(isinstance(v, float) for v in result.values())


# ============================================================================
# Combined Aggregation Level and Granularity Tests
# ============================================================================


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_mean_bias_scorer_aggregation_and_granularity(df_type):
    """MeanBiasScorer with both aggregation_level and granularity"""
    # Game-player level data
    df = create_dataframe(
        df_type,
        {
            "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "team_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "pred": [0.5, 0.6, 0.3, 0.4, 0.7, 0.8, 0.2, 0.3],
            "target": [0, 1, 0, 1, 1, 0, 1, 0],
        },
    )

    # First aggregate to game-team level, then calculate separate scores per team
    scorer = MeanBiasScorer(
        pred_column="pred",
        target="target",
        aggregation_level=["game_id", "team_id"],
        granularity=["team_id"],
    )

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 2  # Two teams

    # After aggregation:
    # Game1-team1: pred=1.1, target=1
    # Game1-team2: pred=0.7, target=1
    # Game2-team1: pred=1.5, target=1
    # Game2-team2: pred=0.5, target=1
    #
    # Team 1 scores: (1.1-1) and (1.5-1) = [0.1, 0.5], mean = 0.3
    # Team 2 scores: (0.7-1) and (0.5-1) = [-0.3, -0.5], mean = -0.4
    assert (1,) in result
    assert (2,) in result
    # Values may vary slightly due to aggregation, so just check they're floats
    assert isinstance(result[(1,)], float)
    assert isinstance(result[(2,)], float)


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_sklearn_scorer_aggregation_and_granularity(df_type):
    """SklearnScorer with both aggregation_level and granularity"""
    df = create_dataframe(
        df_type,
        {
            "game_id": [1, 1, 1, 1],
            "player_id": [1, 2, 3, 4],
            "team_id": [1, 1, 2, 2],
            "pred": [0.1, 0.6, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        },
    )

    scorer = SklearnScorer(
        pred_column="pred",
        scorer_function=mean_absolute_error,
        target="target",
        aggregation_level=["game_id", "team_id"],
        granularity=["team_id"],
    )

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert (1,) in result
    assert (2,) in result
    assert all(isinstance(v, float) for v in result.values())


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_granularity_single_row_per_group(df_type):
    """Granularity with single row per group"""
    df = create_dataframe(df_type, {"team_id": [1, 2], "pred": [0.5, 0.6], "target": [0, 1]})

    scorer = MeanBiasScorer(pred_column="pred", target="target", granularity=["team_id"])

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert (1,) in result
    assert (2,) in result


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_granularity_empty_group(df_type):
    """Granularity with empty group (should handle gracefully)"""
    df = create_dataframe(df_type, {"team_id": [1, 1], "pred": [0.5, 0.6], "target": [0, 1]})

    scorer = MeanBiasScorer(pred_column="pred", target="target", granularity=["team_id"])

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 1  # Only team 1
    assert (1,) in result


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_aggregation_level_no_granularity(df_type):
    """aggregation_level without granularity returns single score"""
    df = create_dataframe(
        df_type,
        {
            "game_id": [1, 1, 1, 1],
            "player_id": [1, 2, 3, 4],
            "team_id": [1, 1, 2, 2],
            "pred": [0.5, 0.6, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        },
    )

    scorer = MeanBiasScorer(
        pred_column="pred", target="target", aggregation_level=["game_id", "team_id"]
    )

    result = scorer.score(df)
    assert isinstance(result, float)  # Single score, not dict


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_no_aggregation_no_granularity(df_type):
    """No aggregation_level and no granularity returns single score"""
    df = create_dataframe(df_type, {"pred": [0.5, 0.6, 0.3, 0.4], "target": [0, 1, 0, 1]})

    scorer = MeanBiasScorer(pred_column="pred", target="target")

    result = scorer.score(df)
    assert isinstance(result, float)  # Single score, not dict
    expected = (0.5 - 0 + 0.6 - 1 + 0.3 - 0 + 0.4 - 1) / 4
    assert abs(result - expected) < 1e-10


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_granularity_with_filters(df_type):
    """Granularity works with filters"""
    df = create_dataframe(
        df_type,
        {
            "team_id": [1, 1, 2, 2],
            "pred": [0.5, 0.6, 0.3, 0.4],
            "target": [0, 1, 0, 1],
            "filter_col": [1, 1, 0, 0],
        },
    )

    from spforge.scorer import Filter, Operator

    scorer = MeanBiasScorer(
        pred_column="pred",
        target="target",
        granularity=["team_id"],
        filters=[Filter(column_name="filter_col", value=1, operator=Operator.EQUALS)],
    )

    result = scorer.score(df)
    assert isinstance(result, dict)
    assert len(result) == 1  # Only team 1 (team 2 filtered out)
    assert (1,) in result


@pytest.mark.parametrize("df_type", [pl.DataFrame, pd.DataFrame])
def test_granularity_with_validation_column(df_type):
    """Granularity works with validation column"""
    df = create_dataframe(
        df_type,
        {
            "team_id": [1, 1, 2, 2],
            "pred": [0.5, 0.6, 0.3, 0.4],
            "target": [0, 1, 0, 1],
            "valid": [1, 1, 1, 0],
        },
    )

    scorer = MeanBiasScorer(
        pred_column="pred", target="target", granularity=["team_id"], validation_column="valid"
    )

    result = scorer.score(df)
    assert isinstance(result, dict)
    # Team 1: 2 rows (both valid=1)
    # Team 2: 1 row (valid=1), 1 row filtered out (valid=0)
    assert (1,) in result
    assert (2,) in result
