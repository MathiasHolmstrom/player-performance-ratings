"""Baseline tests pinning exact numeric outputs for scorers across pd/pl backends.

These tests use hand-computed expected values (not scorer outputs) to lock in
behavior before refactoring scorers to pure narwhals.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from spforge.scorer import MeanBiasScorer, OrdinalLossScorer
from spforge.scorer._score import RankedProbabilityScorer

# ---------------------------------------------------------------------------
# OrdinalLossScorer
# ---------------------------------------------------------------------------


class TestOrdinalLossScorerParity:
    """Pin OrdinalLossScorer exact outputs across backends."""

    CLASSES = [0, 1, 2]
    PREDS = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]
    TARGETS = [0, 1, 2, 1]

    # Hand-computed:
    # threshold 0/1: log_loss_mean = (log(.7)+log(.9)+log(.8)+log(.9))/4, weight=1/3
    # threshold 1/2: log_loss_mean = (log(.9)+log(.7)+log(.5)+log(.3))/4, weight=2/3
    # sum_lr = -(ll1*w1) - (ll2*w2)
    EXPECTED_SCORE = 0.4590708680

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_exact_3class_score(self, df_type):
        if df_type == pd.DataFrame:
            df = pd.DataFrame({"pred": self.PREDS, "target": self.TARGETS})
        else:
            df = pl.DataFrame({"pred": self.PREDS, "target": self.TARGETS})
        scorer = OrdinalLossScorer(pred_column="pred", target="target", classes=self.CLASSES)
        result = scorer.score(df)
        assert abs(result - self.EXPECTED_SCORE) < 1e-8

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_granularity(self, df_type):
        data = {
            "pred": self.PREDS,
            "target": self.TARGETS,
            "group": [1, 1, 2, 2],
        }
        df = pd.DataFrame(data) if df_type == pd.DataFrame else pl.DataFrame(data)
        scorer = OrdinalLossScorer(
            pred_column="pred", target="target", classes=self.CLASSES, granularity=["group"]
        )
        result = scorer.score(df)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert pytest.approx(result[(1,)], abs=1e-8) == 0.2310177298
        assert pytest.approx(result[(2,)], abs=1e-8) == 0.9485599924

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_compare_to_naive(self, df_type):
        preds = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * 10
        targets = [0, 1, 2] * 10
        if df_type == pd.DataFrame:
            df = pd.DataFrame({"pred": preds, "target": targets})
        else:
            df = pl.DataFrame({"pred": preds, "target": targets})
        scorer = OrdinalLossScorer(
            pred_column="pred", target="target", classes=self.CLASSES, compare_to_naive=True
        )
        result = scorer.score(df)
        assert pytest.approx(result, abs=1e-8) == 0.6364141633

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_zero_weight_class(self, df_type):
        """No target==2, so class_=2 threshold effectively has zero weight for target==2."""
        preds = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]
        targets = [0, 1, 1]
        if df_type == pd.DataFrame:
            df = pd.DataFrame({"pred": preds, "target": targets})
        else:
            df = pl.DataFrame({"pred": preds, "target": targets})
        scorer = OrdinalLossScorer(pred_column="pred", target="target", classes=self.CLASSES)
        result = scorer.score(df)
        assert pytest.approx(result, abs=1e-8) == 0.3328382546


# ---------------------------------------------------------------------------
# RankedProbabilityScorer
# ---------------------------------------------------------------------------


class TestRankedProbabilityScorerParity:
    """Pin RankedProbabilityScorer exact outputs across backends."""

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_perfect_preds_zero(self, df_type):
        preds = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        targets = [0, 1, 2]
        if df_type == pd.DataFrame:
            df = pd.DataFrame({"pred": preds, "target": targets})
        else:
            df = pl.DataFrame({"pred": preds, "target": targets})
        scorer = RankedProbabilityScorer(pred_column="pred", target="target", num_classes=3)
        assert abs(scorer.score(df)) < 1e-10

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_known_data_exact(self, df_type):
        # Hand-computed: row0 RPS=0.145, row1 RPS=0.25, mean=0.1975
        preds = [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]
        targets = [0, 2]
        if df_type == pd.DataFrame:
            df = pd.DataFrame({"pred": preds, "target": targets})
        else:
            df = pl.DataFrame({"pred": preds, "target": targets})
        scorer = RankedProbabilityScorer(pred_column="pred", target="target", num_classes=3)
        assert pytest.approx(scorer.score(df), abs=1e-10) == 0.1975

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_granularity(self, df_type):
        data = {
            "pred": [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5], [0.1, 0.9]],
            "target": [0, 1, 0, 1],
            "group": [1, 1, 2, 2],
        }
        df = pd.DataFrame(data) if df_type == pd.DataFrame else pl.DataFrame(data)
        scorer = RankedProbabilityScorer(
            pred_column="pred", target="target", num_classes=2, granularity=["group"]
        )
        result = scorer.score(df)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert pytest.approx(result[(1,)], abs=1e-8) == 0.065
        assert pytest.approx(result[(2,)], abs=1e-8) == 0.13

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_compare_to_naive(self, df_type):
        preds = [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]
        targets = [0, 2]
        if df_type == pd.DataFrame:
            df = pd.DataFrame({"pred": preds, "target": targets})
        else:
            df = pl.DataFrame({"pred": preds, "target": targets})
        scorer = RankedProbabilityScorer(
            pred_column="pred", target="target", num_classes=3, compare_to_naive=True
        )
        assert pytest.approx(scorer.score(df), abs=1e-8) == 0.0525


# ---------------------------------------------------------------------------
# MeanBiasScorer
# ---------------------------------------------------------------------------


class TestMeanBiasScorerParity:
    """Pin MeanBiasScorer exact outputs across backends."""

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_basic_exact(self, df_type):
        data = {"pred": [2.5, 3.0, 4.0, 3.5], "target": [2.0, 3.0, 3.5, 4.0]}
        df = df_type(data)
        scorer = MeanBiasScorer(pred_column="pred", target="target")
        assert pytest.approx(scorer.score(df), abs=1e-10) == 0.125

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_relative_bias_exact(self, df_type):
        data = {
            "pred": [10.0, 12.0, 8.0, 9.0],
            "target": [9.0, 11.0, 8.5, 9.5],
            "is_home": [1, 1, 0, 0],
        }
        df = df_type(data)
        scorer = MeanBiasScorer(pred_column="pred", target="target", relative_bias_column="is_home")
        assert pytest.approx(scorer.score(df), abs=1e-10) == 1.5

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_granularity(self, df_type):
        data = {
            "pred": [2.0, 3.0, 4.0, 5.0],
            "target": [1.0, 2.0, 3.0, 4.0],
            "group": [1, 1, 2, 2],
        }
        df = df_type(data)
        scorer = MeanBiasScorer(
            pred_column="pred",
            target="target",
            granularity=["group"],
        )
        result = scorer.score(df)
        assert isinstance(result, dict)
        assert len(result) == 2
        # group 1: bias = mean([2-1, 3-2]) = 1.0; group 2: bias = mean([4-3, 5-4]) = 1.0
        for key in result:
            assert pytest.approx(result[key], abs=1e-8) == 1.0


# ---------------------------------------------------------------------------
# List column aggregation through BaseScorer (pd.DataFrame input)
# ---------------------------------------------------------------------------


class TestListColumnAggregationParity:
    """Test list column aggregation through BaseScorer."""

    def test_polars_list_pred_mean_aggregation(self):
        """Polars list column element-wise mean aggregation."""
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
