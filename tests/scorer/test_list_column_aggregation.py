"""Tests for list column aggregation and RankedProbabilityScorer."""

import numpy as np
import polars as pl

from spforge.scorer import MeanBiasScorer
from spforge.scorer._score import RankedProbabilityScorer


class TestListColumnAggregation:
    """Aggregation on list/array prediction columns (element-wise mean/sum)."""

    def test_mean_bias_list_pred_mean_aggregation(self):
        """MeanBiasScorer with list pred column uses element-wise mean."""
        df = pl.DataFrame(
            {
                "game_id": [1, 1, 2, 2],
                "pred": [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]],
                "target": [1, 0, 1, 0],
            }
        )
        scorer = MeanBiasScorer(
            pred_column="pred",
            target="target",
            aggregation_level=["game_id"],
            aggregation_method={"pred": "mean", "target": "mean"},
        )
        result = scorer.score(df)
        assert isinstance(result, float)

    def test_mean_bias_list_pred_sum_aggregation(self):
        """MeanBiasScorer with list pred column uses element-wise sum."""
        df = pl.DataFrame(
            {
                "game_id": [1, 1],
                "pred": [[0.2, 0.8], [0.4, 0.6]],
                "target": [1, 0],
            }
        )
        scorer = MeanBiasScorer(
            pred_column="pred",
            target="target",
            aggregation_level=["game_id"],
            aggregation_method={"pred": "sum", "target": "sum"},
        )
        result = scorer.score(df)
        assert isinstance(result, float)

    def test_list_mean_aggregation_values_correct(self):
        """Element-wise mean produces correct averaged probability vectors."""
        df = pl.DataFrame(
            {
                "group": ["a", "a"],
                "pred": [[0.2, 0.8], [0.4, 0.6]],
                "target": [1, 0],
            }
        )
        scorer = MeanBiasScorer(
            pred_column="pred",
            target="target",
            aggregation_level=["group"],
            aggregation_method={"pred": "mean", "target": "mean"},
        )
        agg = scorer.aggregate(df)
        if hasattr(agg, "to_native"):
            agg = agg.to_native()
        pred_values = agg["pred"].to_list()[0]
        np.testing.assert_allclose(pred_values, [0.3, 0.7])

    def test_list_sum_aggregation_values_correct(self):
        """Element-wise sum produces correct summed probability vectors."""
        df = pl.DataFrame(
            {
                "group": ["a", "a"],
                "pred": [[0.2, 0.8], [0.4, 0.6]],
                "target": [1, 0],
            }
        )
        scorer = MeanBiasScorer(
            pred_column="pred",
            target="target",
            aggregation_level=["group"],
            aggregation_method={"pred": "sum", "target": "sum"},
        )
        agg = scorer.aggregate(df)
        if hasattr(agg, "to_native"):
            agg = agg.to_native()
        pred_values = agg["pred"].to_list()[0]
        np.testing.assert_allclose(pred_values, [0.6, 1.4])

    def test_list_aggregation_with_scalar_target(self):
        """List pred + scalar target: pred gets element-wise agg, target gets scalar agg."""
        df = pl.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "pred": [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]],
                "target": [1, 0, 1, 0],
            }
        )
        scorer = MeanBiasScorer(
            pred_column="pred",
            target="target",
            aggregation_level=["group"],
            aggregation_method={"pred": "mean", "target": "mean"},
        )
        agg = scorer.aggregate(df)
        if hasattr(agg, "to_native"):
            agg = agg.to_native()
        assert len(agg) == 2

    def test_list_aggregation_multiple_groups(self):
        """Multiple group keys with list columns."""
        df = pl.DataFrame(
            {
                "game_id": [1, 1, 1, 1],
                "team_id": [1, 1, 2, 2],
                "pred": [[0.1, 0.9], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]],
                "target": [1, 0, 0, 1],
            }
        )
        scorer = MeanBiasScorer(
            pred_column="pred",
            target="target",
            aggregation_level=["game_id", "team_id"],
            aggregation_method={"pred": "mean", "target": "mean"},
        )
        agg = scorer.aggregate(df)
        if hasattr(agg, "to_native"):
            agg = agg.to_native()
        assert len(agg) == 2


class TestRankedProbabilityScorer:
    """Tests for RankedProbabilityScorer."""

    def test_perfect_prediction_scores_zero(self):
        """Perfect one-hot predictions should give RPS = 0."""
        df = pl.DataFrame(
            {
                "pred": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "target": [0, 1, 2],
            }
        )
        scorer = RankedProbabilityScorer(pred_column="pred", target="target", num_classes=3)
        score = scorer.score(df)
        assert abs(score) < 1e-10

    def test_worst_prediction_high_score(self):
        """Maximally wrong predictions should give high RPS."""
        df = pl.DataFrame(
            {
                "pred": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                "target": [0, 2],
            }
        )
        scorer = RankedProbabilityScorer(pred_column="pred", target="target", num_classes=3)
        score = scorer.score(df)
        assert score > 0.4

    def test_rps_range(self):
        """RPS should be in [0, 1]."""
        df = pl.DataFrame(
            {
                "pred": [[0.3, 0.4, 0.3], [0.1, 0.8, 0.1], [0.5, 0.3, 0.2]],
                "target": [0, 1, 2],
            }
        )
        scorer = RankedProbabilityScorer(pred_column="pred", target="target", num_classes=3)
        score = scorer.score(df)
        assert 0.0 <= score <= 1.0

    def test_granularity(self):
        """RPS with granularity returns dict of per-group scores."""
        df = pl.DataFrame(
            {
                "group": [1, 1, 2, 2],
                "pred": [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5], [0.1, 0.9]],
                "target": [0, 1, 0, 1],
            }
        )
        scorer = RankedProbabilityScorer(
            pred_column="pred",
            target="target",
            num_classes=2,
            granularity=["group"],
        )
        result = scorer.score(df)
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_compare_to_naive(self):
        """compare_to_naive returns positive value when model beats naive."""
        df = pl.DataFrame(
            {
                "pred": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * 10,
                "target": [0, 1, 2] * 10,
            }
        )
        scorer = RankedProbabilityScorer(
            pred_column="pred",
            target="target",
            num_classes=3,
            compare_to_naive=True,
        )
        score = scorer.score(df)
        assert score > 0  # Perfect model beats naive

    def test_validation_column(self):
        """Only validation rows are scored."""
        df = pl.DataFrame(
            {
                "pred": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]],
                "target": [0, 1, 0, 1],
                "is_validation": [1, 1, 0, 0],
            }
        )
        scorer = RankedProbabilityScorer(
            pred_column="pred",
            target="target",
            num_classes=2,
            validation_column="is_validation",
        )
        score = scorer.score(df)
        assert abs(score) < 1e-10  # Only perfect predictions are in validation

    def test_aggregation_level_with_list_columns(self):
        """RPS with aggregation_level uses base class list aggregation."""
        df = pl.DataFrame(
            {
                "game_id": [1, 1, 2, 2],
                "pred": [[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]],
                "target": [0, 0, 1, 1],
            }
        )
        scorer = RankedProbabilityScorer(
            pred_column="pred",
            target="target",
            num_classes=2,
            aggregation_level=["game_id"],
            aggregation_method={"pred": "mean", "target": "mean"},
        )
        score = scorer.score(df)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_dataframe(self):
        """Empty DataFrame returns 0.0."""
        df = pl.DataFrame(
            {
                "pred": pl.Series([], dtype=pl.List(pl.Float64)),
                "target": pl.Series([], dtype=pl.Int64),
            }
        )
        scorer = RankedProbabilityScorer(pred_column="pred", target="target", num_classes=3)
        assert scorer.score(df) == 0.0
