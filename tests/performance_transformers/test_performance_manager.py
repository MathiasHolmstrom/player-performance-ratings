import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from spforge.performance_transformers import PerformanceWeightsManager, QuantilePerformanceScaler
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
        min_value=0.0,
        max_value=1.0,
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


class TestZeroInflationHandling:
    @pytest.fixture
    def zero_inflated_data(self):
        """Create zero-inflated data with ~37.7% zeros."""
        np.random.seed(42)
        n = 1000
        zeros = np.zeros(377)
        nonzeros = np.random.exponential(scale=2, size=n - 377)
        raw = np.concatenate([zeros, nonzeros])
        np.random.shuffle(raw)
        return raw

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_performance_manager_detects_zero_inflation(self, frame, zero_inflated_data):
        """Test that PerformanceManager auto-detects zero-inflated distributions."""
        df = _make_native_df(frame, {"x": zero_inflated_data})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=["symmetric", "partial_standard_scaler", "min_max"],
            prefix="performance__",
            performance_column="perf",
            zero_inflation_threshold=0.15,
        )

        pm.fit(df)

        # Should have switched to quantile scaler
        assert pm._using_quantile_scaler is True
        assert isinstance(pm.transformers[-1], QuantilePerformanceScaler)

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_performance_manager_uses_standard_pipeline_for_normal_data(self, frame):
        """Test that PerformanceManager uses standard pipeline for non-zero-inflated data."""
        np.random.seed(42)
        # Normal distribution - no zero inflation
        data = np.random.normal(loc=0.5, scale=0.1, size=1000)
        df = _make_native_df(frame, {"x": data})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=["symmetric", "partial_standard_scaler", "min_max"],
            prefix="performance__",
            performance_column="perf",
            zero_inflation_threshold=0.15,
        )

        pm.fit(df)

        # Should NOT have switched to quantile scaler
        assert pm._using_quantile_scaler is False
        assert isinstance(pm.transformers[-1], MinMaxTransformer)

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_zero_inflation_output_properties(self, frame, zero_inflated_data):
        """Test that zero-inflated output has correct properties."""
        df = _make_native_df(frame, {"x": zero_inflated_data})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=["symmetric", "partial_standard_scaler", "min_max"],
            prefix="performance__",
            performance_column="perf",
            zero_inflation_threshold=0.15,
        )

        result = pm.fit_transform(df)
        result_nw = nw.from_native(result)
        scaled = result_nw["performance__perf"].to_numpy()

        # 1. All zeros should have the same scaled value (the midpoint of zero mass)
        is_zero = np.abs(zero_inflated_data) < 1e-10
        zero_scaled_values = scaled[is_zero]
        assert np.allclose(zero_scaled_values, zero_scaled_values[0], atol=1e-10)

        # 2. Zeros should have lower values than non-zeros (on average)
        is_nonzero = ~is_zero
        assert np.mean(scaled[is_zero]) < np.mean(scaled[is_nonzero])

        # 3. Mean should be approximately 0.5
        assert abs(np.mean(scaled) - 0.5) < 0.02

        # 4. Monotonicity preserved
        order = np.argsort(zero_inflated_data)
        sorted_scaled = scaled[order]
        assert np.all(np.diff(sorted_scaled) >= -1e-10)

        # 5. Bounded [0, 1] (with clipping tolerance)
        assert np.all((scaled >= pm.min_value) & (scaled <= pm.max_value))

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_disable_zero_inflation_detection(self, frame, zero_inflated_data):
        """Test that zero_inflation_threshold=0 disables detection."""
        df = _make_native_df(frame, {"x": zero_inflated_data})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=["symmetric", "partial_standard_scaler", "min_max"],
            prefix="performance__",
            performance_column="perf",
            zero_inflation_threshold=0,  # Disable detection
        )

        pm.fit(df)

        # Should NOT have switched to quantile scaler
        assert pm._using_quantile_scaler is False

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_performance_weights_manager_zero_inflation(self, frame, zero_inflated_data):
        """Test that PerformanceWeightsManager also handles zero inflation."""
        df = _make_native_df(frame, {"feat_a": zero_inflated_data})

        weights = [ColumnWeight(name="feat_a", weight=1.0)]
        manager = PerformanceWeightsManager(
            weights=weights,
            # Use default transformers (None) to enable zero inflation detection
            transformer_names=None,
            prefix="",
            zero_inflation_threshold=0.15,
        )

        manager.fit(df)

        # Should have switched to quantile scaler
        assert manager._using_quantile_scaler is True


class TestWeightedQuantileScaling:
    """Tests for weighted quantile scaling in PerformanceManager."""

    @pytest.fixture
    def weighted_zero_inflated_data(self):
        """Create zero-inflated data where high-weight rows have higher non-zero rate."""
        np.random.seed(42)
        n = 1000

        # Create weights (e.g., minutes played)
        weights = np.random.exponential(scale=20, size=n) + 1

        # High-weight rows have lower zero probability
        values = []
        for w in weights:
            zero_prob = 0.6 - 0.4 * (w / weights.max())
            if np.random.random() < zero_prob:
                values.append(0.0)
            else:
                values.append(np.random.exponential(scale=2))

        return np.array(values), weights

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_performance_manager_with_weight_column(self, frame, weighted_zero_inflated_data):
        """Test that PerformanceManager passes weight column to QuantilePerformanceScaler."""
        values, weights = weighted_zero_inflated_data
        df = _make_native_df(frame, {"x": values, "minutes": weights})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=None,  # Use defaults, auto-detect zero inflation
            prefix="performance__",
            performance_column="perf",
            zero_inflation_threshold=0.15,
            quantile_weight_column="minutes",
        )

        pm.fit(df)

        # Should have switched to quantile scaler
        assert pm._using_quantile_scaler is True
        assert isinstance(pm.transformers[-1], QuantilePerformanceScaler)
        # And should have the weight column set
        assert pm.transformers[-1].weight_column == "minutes"

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_weighted_scaling_reduces_weighted_bias(self, frame, weighted_zero_inflated_data):
        """Test that weighted scaling produces weighted mean closer to 0.5."""
        values, weights = weighted_zero_inflated_data
        df = _make_native_df(frame, {"x": values, "minutes": weights})

        # With weighted scaling
        pm_weighted = PerformanceManager(
            features=["x"],
            transformer_names=None,
            prefix="performance__",
            performance_column="perf",
            zero_inflation_threshold=0.15,
            quantile_weight_column="minutes",
        )

        result_weighted = pm_weighted.fit_transform(df)
        result_weighted_nw = nw.from_native(result_weighted)
        scaled_weighted = result_weighted_nw["performance__perf"].to_numpy()

        # Without weighted scaling
        pm_unweighted = PerformanceManager(
            features=["x"],
            transformer_names=None,
            prefix="performance__",
            performance_column="perf",
            zero_inflation_threshold=0.15,
            quantile_weight_column=None,  # No weighting
        )

        result_unweighted = pm_unweighted.fit_transform(df)
        result_unweighted_nw = nw.from_native(result_unweighted)
        scaled_unweighted = result_unweighted_nw["performance__perf"].to_numpy()

        # Compute weighted means
        weighted_mean_of_weighted = np.average(scaled_weighted, weights=weights)
        weighted_mean_of_unweighted = np.average(scaled_unweighted, weights=weights)

        # Weighted scaling should have weighted mean closer to 0.5
        assert abs(weighted_mean_of_weighted - 0.5) < abs(weighted_mean_of_unweighted - 0.5), (
            f"Weighted mean with weighted scaling ({weighted_mean_of_weighted:.4f}) "
            f"should be closer to 0.5 than without ({weighted_mean_of_unweighted:.4f})"
        )

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_performance_weights_manager_with_quantile_weight_column(
        self, frame, weighted_zero_inflated_data
    ):
        """Test that PerformanceWeightsManager also supports quantile_weight_column."""
        from spforge.performance_transformers._performance_manager import ColumnWeight

        values, weights = weighted_zero_inflated_data
        df = _make_native_df(frame, {"feat_a": values, "minutes": weights})

        column_weights = [ColumnWeight(name="feat_a", weight=1.0)]
        manager = PerformanceWeightsManager(
            weights=column_weights,
            transformer_names=None,
            prefix="",
            zero_inflation_threshold=0.15,
            quantile_weight_column="minutes",
        )

        manager.fit(df)

        # Should have switched to quantile scaler with weight column
        assert manager._using_quantile_scaler is True
        assert manager.transformers[-1].weight_column == "minutes"

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_weight_column_not_used_when_no_zero_inflation(self, frame):
        """Test that weight column is not needed when zero inflation is not detected."""
        np.random.seed(42)
        # Normal distribution - no zero inflation
        data = np.random.normal(loc=0.5, scale=0.1, size=1000)
        weights = np.random.exponential(scale=20, size=1000) + 1

        df = _make_native_df(frame, {"x": data, "minutes": weights})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=None,
            prefix="performance__",
            performance_column="perf",
            zero_inflation_threshold=0.15,
            quantile_weight_column="minutes",
        )

        pm.fit(df)

        # Should NOT have switched to quantile scaler
        assert pm._using_quantile_scaler is False


class TestAutoScalePerformanceBounds:
    """Tests for ensuring scaled performance stays within [0, 1] bounds."""

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_auto_scale_performance_preserves_non_negative(self, frame):
        """Scaled performance should be non-negative when input is non-negative."""
        np.random.seed(42)
        # Create data similar to free throw % - centered around 0.77 with some zeros
        n = 400
        data = []
        for _ in range(n):
            if np.random.random() < 0.25:  # 25% zeros (missed all free throws)
                data.append(0.0)
            else:
                # Values between 0.6 and 1.0, centered around 0.77
                data.append(np.clip(np.random.normal(0.77, 0.15), 0.0, 1.0))

        df = _make_native_df(frame, {"x": data})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=["symmetric", "partial_standard_scaler", "min_max"],
            prefix="performance__",
            performance_column="perf",
        )

        result = pm.fit_transform(df)
        result_nw = nw.from_native(result)
        scaled = result_nw["performance__perf"].to_numpy()

        assert np.all(scaled >= 0), f"Scaled performance should not be negative, min was {scaled.min()}"

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_auto_scale_performance_output_range(self, frame):
        """Scaled performance should be in [0, 1] when input is in [0, 1]."""
        np.random.seed(42)
        # Create data with performance in [0, 1] but skewed distribution
        n = 400
        data = []
        for _ in range(n):
            if np.random.random() < 0.25:
                data.append(0.0)
            else:
                data.append(np.clip(np.random.normal(0.77, 0.15), 0.0, 1.0))

        df = _make_native_df(frame, {"x": data})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=["symmetric", "partial_standard_scaler", "min_max"],
            prefix="performance__",
            performance_column="perf",
        )

        result = pm.fit_transform(df)
        result_nw = nw.from_native(result)
        scaled = result_nw["performance__perf"].to_numpy()

        assert np.all(scaled >= 0.0), f"Scaled min should be >= 0, got {scaled.min()}"
        assert np.all(scaled <= 1.0), f"Scaled max should be <= 1, got {scaled.max()}"

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_default_bounds_are_unit_interval(self, frame):
        """Test that default bounds are [0, 1]."""
        pm = PerformanceManager(
            features=["x"],
            transformer_names=[],
            prefix="",
            performance_column="x",
        )

        assert pm.min_value == 0.0
        assert pm.max_value == 1.0

    @pytest.mark.parametrize("frame", ["pd", "pl"])
    def test_custom_bounds_still_work(self, frame):
        """Test that custom bounds can still be specified."""
        df = _make_native_df(frame, {"x": [-10.0, 0.5, 10.0]})

        pm = PerformanceManager(
            features=["x"],
            transformer_names=[],
            prefix="",
            performance_column="x",
            min_value=-0.5,
            max_value=1.5,
        )

        result = pm.fit_transform(df)
        result_nw = nw.from_native(result)
        scaled = result_nw["x"].to_numpy()

        assert scaled.min() >= -0.5
        assert scaled.max() <= 1.5
