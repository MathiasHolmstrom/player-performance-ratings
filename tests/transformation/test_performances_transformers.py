import pytest

import polars as pl
from polars.testing import assert_frame_equal
import pandas as pd

from spforge.transformers.performances_transformers import (
    SklearnEstimatorImputer,
    DiminishingValueTransformer,
    SymmetricDistributionTransformer,
    GroupByTransformer,
)
from sklearn.linear_model import LinearRegression

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
        df = pd.DataFrame({"position": positions, "a": a, "b": b}).sort_values(
            by=["position", "a"]
        )
    elif df_type == "pl":
        df = pl.DataFrame({"position": positions, "a": a, "b": b}).sort(
            ["position", "a"]
        )
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
        df = pd.DataFrame({"position": positions, "a": a, "b": b}).sort_values(
            by=["position", "a"]
        )
    elif df_type == "pl":
        df = pl.DataFrame({"position": positions, "a": a, "b": b}).sort(
            ["position", "a"]
        )
    else:
        raise ValueError("df_type must be 'pd' or 'pl'")

    transformed = transformer.fit_transform(df)
    if isinstance(df, pd.DataFrame):
        df["va"] = df.groupby(transformer.granularity)[["a"]].transform("mean")
        df["vb"] = df.groupby(transformer.granularity)[["b"]].transform("mean")
        pd.testing.assert_frame_equal(
            transformed.reset_index(drop=True), df.reset_index(drop=True)
        )
    elif isinstance(df, pl.DataFrame):
        expected_transformed = df.with_columns(
            [
                pl.col("a").mean().over(["position"]).alias("va"),
                pl.col("b").mean().over(["position"]).alias("vb"),
            ]
        )
        assert_frame_equal(transformed, expected_transformed)
