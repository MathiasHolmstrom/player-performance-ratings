import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from spforge.performance_transformers import PerformanceWeightsManager
from spforge.performance_transformers._performance_manager import (
    ColumnWeight,
    PerformanceManager,
    create_performance_scalers_transformers,
)
from spforge.performance_transformers._performances_transformers import (
    MinMaxTransformer,
    PartialStandardScaler,
    SymmetricDistributionTransformer,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "feat_a": [1.0, 0.5, np.nan, 0.0],
            "feat_b": [0.0, 0.5, 1.0, np.nan],
            "other": [1, 2, 3, 4],
        }
    )


def test_column_weight_validation():
    # Valid weights
    ColumnWeight(name="test", weight=0.5)

    # Invalid weights
    with pytest.raises(ValueError, match="Weight must be positive"):
        ColumnWeight(name="test", weight=-0.1)
    with pytest.raises(ValueError, match="Weight must be less than 1"):
        ColumnWeight(name="test", weight=1.1)


def test_performance_weights_manager_basic_flow(sample_data):
    weights = [ColumnWeight(name="feat_a", weight=0.6), ColumnWeight(name="feat_b", weight=0.4)]

    # Initialize with no transformers to test pure weighting logic
    manager = PerformanceWeightsManager(weights=weights, transformer_names=[], prefix="")

    # Mocking fit/transform because we passed empty transformers
    manager.fit(sample_data)
    output = manager.transform(sample_data)
    output_df = nw.from_native(output).to_pandas()

    assert "weighted_performance" in output_df.columns
    # Row 0: feat_a=1.0 (w=0.6), feat_b=0.0 (w=0.4). Sum = 0.6*1 + 0.4*0 = 0.6
    # Note: the code divides by sum_weight (0.6+0.4=1.0)
    assert output_df["weighted_performance"].iloc[0] == pytest.approx(0.6)


def test_performance_weights_manager_keeps_mean_when_weights_not_normalized():
    df = pd.DataFrame(
        {
            "feat_a": [0.0, 1.0, 2.0, 3.0],
            "feat_b": [3.0, 2.0, 1.0, 0.0],
        }
    )
    weights = [ColumnWeight(name="feat_a", weight=0.9), ColumnWeight(name="feat_b", weight=0.5)]

    manager = PerformanceWeightsManager(weights=weights, transformer_names=["min_max"], prefix="")
    output_df = nw.from_native(manager.fit_transform(df)).to_pandas()

    assert output_df["weighted_performance"].mean() == pytest.approx(0.5, abs=1e-6)


def test_lower_is_better_logic():
    df = pd.DataFrame({"feat_a": [1.0, 0.0]})
    weights = [ColumnWeight(name="feat_a", weight=1.0, lower_is_better=True)]

    manager = PerformanceWeightsManager(weights=weights, transformer_names=[], prefix="")
    output = manager.fit_transform(df)
    output_df = nw.from_native(output).to_pandas()

    # For lower_is_better, 1.0 should become 0.0, and 0.0 should become 1.0
    assert output_df["weighted_performance"].iloc[0] == 0.0
    assert output_df["weighted_performance"].iloc[1] == 1.0


def test_null_redistribution(sample_data):
    # If one column is null, the other column should take the full proportional weight
    weights = [ColumnWeight(name="feat_a", weight=0.5), ColumnWeight(name="feat_b", weight=0.5)]
    manager = PerformanceWeightsManager(weights=weights, transformer_names=[], prefix="")

    output = manager.fit_transform(sample_data)
    output_df = nw.from_native(output).to_pandas()

    # Row 2: feat_a is NaN, feat_b is 1.0.
    # Logic: weight_a becomes 0. sum_cols_weights becomes 0.5.
    # normalized weight_b = 0.5 / 0.5 = 1.0.
    # Result = 1.0 * feat_b = 1.0
    assert output_df["weighted_performance"].iloc[2] == 1.0


def test_clipping():
    # Test that values are clipped to min_value and max_value
    df = pd.DataFrame({"feat_a": [10.0, -10.0]})
    weights = [ColumnWeight(name="feat_a", weight=1.0)]

    manager = PerformanceWeightsManager(
        weights=weights, transformer_names=[], prefix="", min_value=0.0, max_value=1.0
    )

    output_df = nw.from_native(manager.fit_transform(df)).to_pandas()
    assert output_df["weighted_performance"].iloc[0] == 1.0
    assert output_df["weighted_performance"].iloc[1] == 0.0


def test_prefix_resolution():
    # Data has 'feat_a', Manager expects 'performance__feat_a'
    df = pd.DataFrame({"feat_a": [1.0, 0.0]})
    weights = [ColumnWeight(name="feat_a", weight=1.0)]

    # We set prefix but don't provide the prefixed column in input
    manager = PerformanceWeightsManager(
        weights=weights, transformer_names=[], prefix="performance__"
    )

    # transform should internally create performance__feat_a from feat_a
    output = manager.fit_transform(df)
    output_df = nw.from_native(output).to_pandas()

    assert "performance__weighted_performance" in output_df.columns
    assert output_df["performance__weighted_performance"].iloc[0] == 1.0


def test_all_nulls_raises_error():
    # If all weighted columns are null, the code should log error and raise ValueError
    df = pd.DataFrame({"feat_a": [np.nan], "feat_b": [np.nan]})
    weights = [ColumnWeight(name="feat_a", weight=0.5), ColumnWeight(name="feat_b", weight=0.5)]
    manager = PerformanceWeightsManager(weights=weights, transformer_names=[], prefix="")

    with pytest.raises(ValueError, match="performance contains nan values"):
        manager.fit_transform(df)


def test_return_all_features_property():
    weights = [ColumnWeight(name="feat_a", weight=1.0)]
    manager_true = PerformanceWeightsManager(
        weights=weights, return_all_features=True, prefix="p__"
    )
    manager_false = PerformanceWeightsManager(
        weights=weights, return_all_features=False, prefix="p__"
    )

    assert "p__feat_a" in manager_true.features_out
    assert "p__weighted_performance" in manager_true.features_out
    assert manager_false.features_out == ["p__weighted_performance"]


class _ConstantOutTransformer:
    def __init__(self, features: list[str], value: float, out_name: str = "out"):
        self.features = features
        self.value = float(value)
        self._out_name = out_name
        self.features_out: list[str] = []

    def fit(self, df):
        self.features_out = [f"{self._out_name}__{self.features[0]}"]
        return self

    def transform(self, df):
        return df.with_columns(nw.lit(self.value).alias(self.features_out[0]))


def _make_native_df(frame: str, data: dict):
    if frame == "pd":
        if pd is None:
            pytest.skip("pandas not available")
        return pd.DataFrame(data)
    if frame == "pl":
        if pl is None:
            pytest.skip("polars not available")
        return pl.DataFrame(data)
    raise ValueError(frame)


@pytest.mark.parametrize("bad_weight", [-0.1, -1.0])
def test_column_weight_rejects_negative_weights(bad_weight):
    with pytest.raises(ValueError):
        ColumnWeight(name="a", weight=bad_weight)


@pytest.mark.parametrize("bad_weight", [1.00001, 2.0])
def test_column_weight_rejects_weights_above_one(bad_weight):
    with pytest.raises(ValueError):
        ColumnWeight(name="a", weight=bad_weight)


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_performance_manager_aliases_unprefixed_input_when_transformer_expects_prefixed(frame):
    df = _make_native_df(frame, {"x": [0.1, 0.2], "other": [1, 2]})

    pm = PerformanceManager(
        features=["x"],
        transformer_names=[],
        prefix="performance__",
        performance_column="weighted_performance",
        min_value=-0.02,
        max_value=1.02,
    )

    pm.fit(df)
    out = pm.transform(df)

    out_nw = nw.from_native(out)
    cols = set(out_nw.columns)

    assert "x" in cols
    assert "other" in cols
    assert pm.performance_column in cols
    assert pm.performance_column == "performance__weighted_performance"


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_performance_manager_returns_only_input_cols_plus_performance(frame):
    df = _make_native_df(frame, {"x": [0.1, 0.2], "y": [10, 20]})

    pm = PerformanceManager(
        features=["x"],
        transformer_names=[],
        prefix="performance__",
        performance_column="weighted_performance",
    )
    pm.fit(df)
    out = pm.transform(df)

    out_nw = nw.from_native(out)
    assert set(out_nw.columns) == {"x", "y", pm.performance_column}
    assert pm.features_out == [pm.performance_column]


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_performance_manager_creates_transformers(frame):
    df = _make_native_df(frame, {"x": [0.1, 0.2]})

    pm = PerformanceManager(
        features=["x"], transformer_names=[], prefix="performance__", performance_column="x2"
    )

    pm.fit(df)
    assert len(pm.transformers) > 0


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_performance_manager_does_not_change_original_feature_performance_column_differ(frame):
    df = _make_native_df(frame, {"x": [0.1, 0.2, 1.2]})
    ori_x = df["x"].to_list()
    pm = PerformanceManager(features=["x"], transformer_names=[], performance_column="x2")

    new_df = nw.from_native(pm.fit_transform(df))
    assert new_df["x"].to_list() == ori_x


@pytest.mark.parametrize("frame", ["pd", "pl"])
def test_performance_manager_does_not_change_original_feature(frame):
    df = _make_native_df(frame, {"x": [0.1, 0.2, 1.2]})
    ori_x = df["x"].to_list()
    pm = PerformanceManager(features=["x"], transformer_names=[], performance_column="x")

    new_df = nw.from_native(pm.fit_transform(df))
    assert new_df["x"].to_list() == ori_x


@pytest.mark.parametrize("frame", ["pd", "pl"])
@pytest.mark.parametrize(
    "transformer_names, expected_types",
    [
        (["symmetric"], [SymmetricDistributionTransformer]),
        (["partial_standard_scaler"], [PartialStandardScaler]),
        (["partial_standard_scaler_mean0.5"], [PartialStandardScaler]),
        (["min_max"], [MinMaxTransformer]),
        (
            ["symmetric", "partial_standard_scaler", "min_max"],
            [SymmetricDistributionTransformer, PartialStandardScaler, MinMaxTransformer],
        ),
    ],
)
def test_factory_sets_transformer_features_to_prefixed_inputs_and_features_out_to_prefixed_outputs(
    frame,
    transformer_names,
    expected_types,
):
    prefix = "performance__"
    features = ["x", "y"]
    expected_in = [prefix + f for f in features]

    pre = []
    ts = create_performance_scalers_transformers(
        transformer_names=transformer_names,
        pre_transformers=pre,
        features=features,
        prefix=prefix,
    )

    assert [type(t) for t in ts] == expected_types

    for idx, t in enumerate(ts):
        if idx + 1 < len(ts):
            assert t.features_out == ts[idx + 1].features
        assert t.features == expected_in
