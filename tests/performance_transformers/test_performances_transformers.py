import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression

from spforge.performance_transformers import (
    DiminishingValueTransformer,
    QuantilePerformanceScaler,
    SymmetricDistributionTransformer,
)
from spforge.performance_transformers._performances_transformers import (
    GroupByTransformer,
    SklearnEstimatorImputer,
)

dfs = [
    pl.DataFrame(
        {
            "a": [1.0, 2.2, 3.0, 2.3],
            "b": [4.0, 5.0, 6.4, 20.8],
            "target": [2.1, float("nan"), None, 4.2],
        }
    ),
    pd.DataFrame(
        {
            "a": [1.0, 2.2, 3.0, 2.3],
            "b": [4.0, 5.0, 6.4, 20.8],
            "target": [2.1, float("nan"), None, 4.2],
        }
    ),
]


@pytest.mark.parametrize("df", dfs)
def test_sklearn_estimator_imputer(df):
    imputer = SklearnEstimatorImputer(
        estimator=LinearRegression(),
        target_name="target",
        features=["a", "b"],
    )
    transformed = imputer.fit_transform(df)

    if isinstance(df, pd.DataFrame):
        assert transformed["target"].isnull().sum() == 0
    elif isinstance(df, pl.DataFrame):
        assert transformed["target"].is_null().sum() == 0
        assert transformed["target"].is_nan().sum() == 0

    assert len(transformed) == len(df)


@pytest.mark.parametrize("df", dfs)
def test_diminishing_value_transformer(df):
    transformer = DiminishingValueTransformer(
        features=["a", "b"],
    )

    transformed = transformer.fit_transform(df)

    assert transformed["a"].max() < df["a"].max()
    assert transformed["b"].max() < df["b"].max()

    assert transformed["a"].min() == df["a"].min()
    assert transformed["b"].min() == df["b"].min()

    assert len(transformed) == len(df)


@pytest.mark.parametrize("df_type", ["pl", "pd"])
@pytest.mark.parametrize("granularity", [["position"], None])
def test_symmetric_distribution_transformer(df_type, granularity):
    import numpy as np

    positions = np.random.choice(["Forward", "Midfielder", "Defender"], size=3500)
    a = np.random.normal(loc=0, scale=10, size=3500)
    b = np.random.exponential(scale=2, size=3500)
    if df_type == "pd":
        df = pd.DataFrame({"position": positions, "a": a, "b": b}).sort_values(by=["position", "a"])
    elif df_type == "pl":
        df = pl.DataFrame({"position": positions, "a": a, "b": b}).sort(["position", "a"])
    else:
        raise ValueError("df_type must be 'pd' or 'pl'")

    transformer = SymmetricDistributionTransformer(
        features=["a", "b"], skewness_allowed=0.3, granularity=granularity, min_rows=1
    )

    transformed = transformer.fit_transform(df)
    if df_type == "pd":
        transformed = transformed.sort_values(by=["position", "a"])
        assert transformed["a"].tolist() == df["a"].tolist()
        assert transformed["b"].skew() < df["b"].skew()
    elif df_type == "pl":
        transformed = transformed.sort(["position", "a"])
        assert transformed["a"].to_list() == df["a"].to_list()
        assert transformed["b"].skew() < df["b"].skew()


@pytest.mark.parametrize("df_type", ["pd", "pl"])
def test_group_by_transformer(df_type):
    import numpy as np

    transformer = GroupByTransformer(
        features=["a", "b"], aggregation="mean", granularity=["position"], prefix="v"
    )
    positions = np.random.choice(["Forward", "Midfielder", "Defender"], size=3500)
    a = np.random.normal(loc=0, scale=10, size=3500)
    b = np.random.exponential(scale=2, size=3500)
    if df_type == "pd":
        df = pd.DataFrame({"position": positions, "a": a, "b": b}).sort_values(by=["position", "a"])
    elif df_type == "pl":
        df = pl.DataFrame({"position": positions, "a": a, "b": b}).sort(["position", "a"])
    else:
        raise ValueError("df_type must be 'pd' or 'pl'")

    transformed = transformer.fit_transform(df)
    if isinstance(df, pd.DataFrame):
        df["va"] = df.groupby(transformer.granularity)[["a"]].transform("mean")
        df["vb"] = df.groupby(transformer.granularity)[["b"]].transform("mean")
        pd.testing.assert_frame_equal(transformed.reset_index(drop=True), df.reset_index(drop=True))
    elif isinstance(df, pl.DataFrame):
        expected_transformed = df.with_columns(
            [
                pl.col("a").mean().over(["position"]).alias("va"),
                pl.col("b").mean().over(["position"]).alias("vb"),
            ]
        )
        assert_frame_equal(transformed, expected_transformed)


def test_groupby_transformer_fit_transform():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2, 2],
            "performance": [0.2, 0.3, 0.4, 0.5, 0.6, 0.2],
            "player_id": [1, 2, 3, 1, 2, 3],
        }
    )

    transformer = GroupByTransformer(features=["performance"], granularity=["player_id"])

    expected_df = df.copy()
    expected_df[transformer.prefix + "performance"] = [0.35, 0.45, 0.3, 0.35, 0.45, 0.3]

    transformed_df = transformer.fit_transform(df)
    pd.testing.assert_frame_equal(expected_df, transformed_df)


def test_diminshing_value_transformer():
    df = pd.DataFrame(
        {
            "performance": [0.2, 0.2, 0.2, 0.2, 0.9],
            "player_id": [1, 2, 3, 1, 2],
        }
    )

    transformer = DiminishingValueTransformer(features=["performance"])

    ori_df = df.copy()
    transformed_df = transformer.fit_transform(df)

    assert transformed_df["performance"].iloc[4] < ori_df["performance"].iloc[4]
    assert transformed_df["performance"].iloc[0] == ori_df["performance"].iloc[0]


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_reverse_diminshing_value_transformer(df):
    data = df(
        {
            "performance": [0.1, 0.8, 0.8, 0.8, 0.8],
            "player_id": [1, 2, 3, 1, 2],
        }
    )

    transformer = DiminishingValueTransformer(features=["performance"], reverse=True)
    try:
        ori_df = data.copy()
    except Exception:
        ori_df = data.clone()
    transformed_df = transformer.fit_transform(data)

    if isinstance(ori_df, pd.DataFrame):
        assert transformed_df["performance"].iloc[0] > ori_df["performance"].iloc[0]
        assert transformed_df["performance"].iloc[3] == ori_df["performance"].iloc[3]
    else:
        assert (
            transformed_df.row(0, named=True)["performance"]
            > ori_df.row(0, named=True)["performance"]
        )
        assert (
            transformed_df.row(3, named=True)["performance"]
            == ori_df.row(3, named=True)["performance"]
        )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_symmetric_distribution_transformery_fit_transform(df):
    data = df(
        {
            "performance": [
                0.1,
                0.2,
                0.15,
                0.2,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.5,
                0.15,
                0.45,
                0.5,
            ],
            "player_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 4, 3, 2, 2],
            "position": [
                "PG",
                "PG",
                "PG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
            ],
        }
    )

    transformer = SymmetricDistributionTransformer(features=["performance"], max_iterations=40)
    transformed_df = transformer.fit_transform(data)
    if isinstance(transformed_df, pd.DataFrame):
        transformed_df = pl.DataFrame(transformed_df)
    assert abs(data["performance"].skew()) > transformer.skewness_allowed

    assert abs(transformed_df["performance"].skew()) < transformer.skewness_allowed


def test_symmetric_distribution_transformer_transform():
    ori_fit_transform_df = pd.DataFrame(
        {
            "performance": [
                0.1,
                0.2,
                0.15,
                0.2,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.5,
                0.15,
                0.45,
                0.5,
            ],
            "player_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 4, 3, 2, 2],
            "position": [
                "PG",
                "PG",
                "PG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
            ],
        }
    )

    to_transform_df = pd.DataFrame(
        {
            "performance": [0.1, 0.4, 0.8],
            "player_id": [1, 1, 1],
            "position": ["PG", "SG", "SG"],
        }
    )

    transformer = SymmetricDistributionTransformer(features=["performance"], max_iterations=40)
    fit_transformed_df = transformer.fit_transform(ori_fit_transform_df)
    expected_value_1 = fit_transformed_df.iloc[
        ori_fit_transform_df[ori_fit_transform_df["performance"] == 0.8].index.tolist()[0]
    ]["performance"]
    expected_value_2 = fit_transformed_df.iloc[
        ori_fit_transform_df[ori_fit_transform_df["performance"] == 0.1].index.tolist()[0]
    ]["performance"]

    transformed_df = transformer.transform(to_transform_df)

    assert transformed_df["performance"].iloc[2] == expected_value_1
    assert transformed_df["performance"].iloc[0] == expected_value_2
    assert transformed_df["performance"].iloc[0] > 0.1


def test_symmetric_distribution_transformer_with_granularity_fit_transform():
    df = pd.DataFrame(
        {
            "performance": [
                0.1,
                0.2,
                0.15,
                0.2,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.5,
                0.15,
                0.45,
                0.5,
            ],
            "player_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 4, 3, 2, 2],
            "position": [
                "PG",
                "PG",
                "PG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
                "SG",
            ],
        }
    )

    transformer = SymmetricDistributionTransformer(
        features=["performance"], granularity=["position"], max_iterations=40, prefix=""
    )
    transformed_df = transformer.fit_transform(df)
    assert (
        abs(df[lambda x: x.position == "SG"]["performance"].skew()) > transformer.skewness_allowed
    )
    assert (
        abs(transformed_df.loc[lambda x: x.position == "SG"]["performance"].skew())
        < transformer.skewness_allowed
    )


class TestQuantilePerformanceScaler:
    @pytest.fixture
    def zero_inflated_data(self):
        """Create zero-inflated data with ~37.7% zeros."""
        np.random.seed(42)
        n = 1000
        # ~37.7% zeros
        zeros = np.zeros(377)
        # Non-zeros from exponential distribution
        nonzeros = np.random.exponential(scale=2, size=n - 377)
        raw = np.concatenate([zeros, nonzeros])
        np.random.shuffle(raw)
        return raw

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_zeros_map_to_midpoint(self, df_type, zero_inflated_data):
        """Test that zeros map to π/2 (midpoint of zero probability mass)."""
        df = df_type({"performance": zero_inflated_data})

        scaler = QuantilePerformanceScaler(features=["performance"], prefix="")
        transformed = scaler.fit_transform(df)

        if isinstance(transformed, pd.DataFrame):
            scaled = transformed["performance"].values
        else:
            scaled = transformed["performance"].to_numpy()

        pi = scaler._zero_proportion["performance"]
        is_zero = np.abs(zero_inflated_data) < 1e-10

        # Zeros should map to π/2
        assert np.allclose(scaled[is_zero], pi / 2, atol=1e-10)

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_mean_approximately_half(self, df_type, zero_inflated_data):
        """Test that mean ≈ 0.5."""
        df = df_type({"performance": zero_inflated_data})

        scaler = QuantilePerformanceScaler(features=["performance"], prefix="")
        transformed = scaler.fit_transform(df)

        if isinstance(transformed, pd.DataFrame):
            scaled = transformed["performance"].values
        else:
            scaled = transformed["performance"].to_numpy()

        # Mean should be approximately 0.5
        assert abs(np.mean(scaled) - 0.5) < 0.02

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_monotonicity_preserved(self, df_type, zero_inflated_data):
        """Test that monotonicity is preserved (sorted input → sorted output)."""
        df = df_type({"performance": zero_inflated_data})

        scaler = QuantilePerformanceScaler(features=["performance"], prefix="")
        transformed = scaler.fit_transform(df)

        if isinstance(transformed, pd.DataFrame):
            scaled = transformed["performance"].values
        else:
            scaled = transformed["performance"].to_numpy()

        # Check monotonicity: if we sort the raw data, the scaled values should also be sorted
        order = np.argsort(zero_inflated_data)
        sorted_scaled = scaled[order]
        # Allow for tiny numerical errors
        assert np.all(np.diff(sorted_scaled) >= -1e-10)

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_bounded_zero_one(self, df_type, zero_inflated_data):
        """Test that output is bounded [0, 1]."""
        df = df_type({"performance": zero_inflated_data})

        scaler = QuantilePerformanceScaler(features=["performance"], prefix="")
        transformed = scaler.fit_transform(df)

        if isinstance(transformed, pd.DataFrame):
            scaled = transformed["performance"].values
        else:
            scaled = transformed["performance"].to_numpy()

        assert np.all((scaled >= 0) & (scaled <= 1))

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_nonzeros_span_pi_to_one(self, df_type, zero_inflated_data):
        """Test that non-zeros map to range (π, 1)."""
        df = df_type({"performance": zero_inflated_data})

        scaler = QuantilePerformanceScaler(features=["performance"], prefix="")
        transformed = scaler.fit_transform(df)

        if isinstance(transformed, pd.DataFrame):
            scaled = transformed["performance"].values
        else:
            scaled = transformed["performance"].to_numpy()

        pi = scaler._zero_proportion["performance"]
        is_nonzero = np.abs(zero_inflated_data) >= 1e-10

        # Non-zeros should be >= π
        assert np.all(scaled[is_nonzero] >= pi - 1e-10)
        # Non-zeros should be <= 1
        assert np.all(scaled[is_nonzero] <= 1 + 1e-10)

    def test_with_prefix(self):
        """Test that prefix is applied correctly."""
        np.random.seed(42)
        raw = np.concatenate([np.zeros(50), np.random.exponential(2, 50)])
        df = pd.DataFrame({"feat": raw})

        scaler = QuantilePerformanceScaler(features=["feat"], prefix="scaled_")
        transformed = scaler.fit_transform(df)

        assert "scaled_feat" in transformed.columns
        assert scaler.features_out == ["scaled_feat"]

    def test_multiple_features(self):
        """Test that multiple features are handled correctly."""
        np.random.seed(42)
        raw_a = np.concatenate([np.zeros(50), np.random.exponential(2, 50)])
        raw_b = np.concatenate([np.zeros(30), np.random.exponential(3, 70)])
        df = pd.DataFrame({"a": raw_a, "b": raw_b})

        scaler = QuantilePerformanceScaler(features=["a", "b"], prefix="")
        transformed = scaler.fit_transform(df)

        assert "a" in transformed.columns
        assert "b" in transformed.columns

        # Both should have mean ≈ 0.5
        assert abs(transformed["a"].mean() - 0.5) < 0.05
        assert abs(transformed["b"].mean() - 0.5) < 0.05

    def test_all_zeros(self):
        """Test edge case: all values are zero (π=1)."""
        df = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0, 0.0]})

        scaler = QuantilePerformanceScaler(features=["x"], prefix="")
        transformed = scaler.fit_transform(df)

        # π=1, so all values should map to π/2 = 0.5
        assert np.allclose(transformed["x"].values, 0.5)
        assert scaler._zero_proportion["x"] == 1.0

    def test_no_zeros(self):
        """Test edge case: no zeros (π=0)."""
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.exponential(2, 100) + 0.1})  # All positive

        scaler = QuantilePerformanceScaler(features=["x"], prefix="")
        transformed = scaler.fit_transform(df)

        # π=0, so values should span (0, 1) via quantiles
        assert scaler._zero_proportion["x"] == 0.0
        assert transformed["x"].min() >= 0
        assert transformed["x"].max() <= 1
        # Mean should still be ~0.5
        assert abs(transformed["x"].mean() - 0.5) < 0.05

    def test_nan_handling(self):
        """Test that NaN values are preserved in output."""
        df = pd.DataFrame({"x": [0.0, 1.0, np.nan, 2.0, 0.0, np.nan, 3.0]})

        scaler = QuantilePerformanceScaler(features=["x"], prefix="")
        transformed = scaler.fit_transform(df)

        # NaN positions should remain NaN
        assert np.isnan(transformed["x"].iloc[2])
        assert np.isnan(transformed["x"].iloc[5])

        # Non-NaN values should be valid
        non_nan_mask = ~np.isnan(transformed["x"].values)
        assert np.all((transformed["x"].values[non_nan_mask] >= 0) &
                      (transformed["x"].values[non_nan_mask] <= 1))

    def test_single_unique_nonzero(self):
        """Test edge case: single unique non-zero value."""
        df = pd.DataFrame({"x": [0.0, 0.0, 5.0, 5.0, 0.0, 5.0]})

        scaler = QuantilePerformanceScaler(features=["x"], prefix="")
        transformed = scaler.fit_transform(df)

        # Should still work - zeros map to π/2, non-zeros to (π, 1)
        pi = scaler._zero_proportion["x"]
        is_zero = df["x"] == 0

        # Zeros should map to π/2
        assert np.allclose(transformed["x"].values[is_zero.values], pi / 2)

        # Non-zeros should all map to same value (since they're all equal)
        nonzero_values = transformed["x"].values[~is_zero.values]
        assert np.allclose(nonzero_values, nonzero_values[0])
