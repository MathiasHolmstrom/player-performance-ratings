import numpy as np
import pandas as pd
import polars as pl
import pytest

from spforge.transformers import GroupAggregatePredictorTransformer


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_group_aggregate_predictor_transformer__group_total_and_share(df_type):
    data = df_type(
        {
            "match_id": ["m1", "m1", "m2", "m2"],
            "pred": [2.0, 1.0, 3.0, 3.0],
        }
    )
    transformer = GroupAggregatePredictorTransformer(pred_col="pred", group_cols=["match_id"])

    transformer.fit(data)
    out = transformer.transform(data)

    assert list(out.columns) == ["agg_pred_total", "agg_pred_share_group"]

    if isinstance(out, pd.DataFrame):
        totals = out["agg_pred_total"].to_numpy()
        shares = out["agg_pred_share_group"].to_numpy()
    else:
        totals = out["agg_pred_total"].to_numpy()
        shares = out["agg_pred_share_group"].to_numpy()

    assert np.allclose(totals, [3.0, 3.0, 6.0, 6.0])
    assert np.allclose(shares, [2.0 / 3.0, 1.0 / 3.0, 0.5, 0.5])


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_group_aggregate_predictor_transformer__role_totals_and_shares(df_type):
    data = df_type(
        {
            "match_id": ["m1", "m1", "m1", "m1"],
            "is_starting": [True, True, False, False],
            "pred": [4.0, 2.0, 3.0, 1.0],
        }
    )
    transformer = GroupAggregatePredictorTransformer(
        pred_col="pred",
        group_cols=["match_id"],
        role_col="is_starting",
        role_true_label=True,
        prefix="team",
    )

    transformer.fit(data)
    out = transformer.transform(data)

    expected_cols = [
        "team_pred_total",
        "team_pred_share_group",
        "team_role_total",
        "team_non_role_total",
        "team_pred_share_role",
    ]
    assert list(out.columns) == expected_cols

    if isinstance(out, pd.DataFrame):
        shares = out["team_pred_share_role"].to_numpy()
        role_total = out["team_role_total"].to_numpy()
        non_role_total = out["team_non_role_total"].to_numpy()
    else:
        shares = out["team_pred_share_role"].to_numpy()
        role_total = out["team_role_total"].to_numpy()
        non_role_total = out["team_non_role_total"].to_numpy()

    assert np.allclose(role_total, [6.0, 6.0, 6.0, 6.0])
    assert np.allclose(non_role_total, [4.0, 4.0, 4.0, 4.0])
    assert np.allclose(shares, [4.0 / 6.0, 2.0 / 6.0, 3.0 / 4.0, 1.0 / 4.0])


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_group_aggregate_predictor_transformer__segment_totals_and_share(df_type):
    data = df_type(
        {
            "match_id": ["m1", "m1", "m1", "m1"],
            "unit": ["start", "start", "bench", "bench"],
            "pred": [4.0, 2.0, 3.0, 1.0],
        }
    )
    transformer = GroupAggregatePredictorTransformer(
        pred_col="pred",
        group_cols=["match_id"],
        segment_col="unit",
        segment_values=["start", "bench"],
        prefix="team",
    )

    transformer.fit(data)
    out = transformer.transform(data)

    expected_cols = [
        "team_pred_total",
        "team_pred_share_group",
        "team_segment_start_total",
        "team_segment_bench_total",
        "team_pred_share_segment",
    ]
    assert list(out.columns) == expected_cols

    share_segment = out["team_pred_share_segment"].to_numpy()
    start_total = out["team_segment_start_total"].to_numpy()
    bench_total = out["team_segment_bench_total"].to_numpy()
    assert np.allclose(start_total, [6.0, 6.0, 6.0, 6.0])
    assert np.allclose(bench_total, [4.0, 4.0, 4.0, 4.0])
    assert np.allclose(share_segment, [4.0 / 6.0, 2.0 / 6.0, 3.0 / 4.0, 1.0 / 4.0])


@pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
def test_group_aggregate_predictor_transformer__missing_columns_raise(df_type):
    data = df_type({"match_id": ["m1"], "pred": [1.0]})
    transformer = GroupAggregatePredictorTransformer(
        pred_col="pred",
        group_cols=["match_id"],
        segment_col="is_starting",
        segment_values=[True, False],
    )
    if df_type is pd.DataFrame:
        fit_data = data.assign(is_starting=True)
    else:
        fit_data = data.with_columns(pl.lit(True).alias("is_starting"))
    transformer.fit(fit_data)

    with pytest.raises(ValueError, match="Missing required columns for transform"):
        transformer.transform(data)


def test_group_aggregate_predictor_transformer__context_and_feature_names():
    transformer = GroupAggregatePredictorTransformer(
        pred_col="pred",
        group_cols=["match_id", "team_id"],
        segment_col="is_starting",
        segment_values=[True, False],
        prefix="agg",
    )
    assert transformer.context_features == ["match_id", "team_id", "is_starting"]
    assert transformer.get_feature_names_out() == [
        "agg_pred_total",
        "agg_pred_share_group",
        "agg_segment_true_total",
        "agg_segment_false_total",
        "agg_pred_share_segment",
    ]
