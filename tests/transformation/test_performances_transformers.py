import pytest

import polars as pl
from polars.testing import assert_frame_equal
import pandas as pd

from spforge.transformers.fit_transformers import (
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


import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression
from polars.testing import assert_frame_equal
import pytest

from spforge import ColumnNames
from spforge.predictor import SklearnPredictor
from spforge.predictor_transformer import SkLearnTransformerWrapper
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spforge.transformers import RatioTeamPredictorTransformer
from spforge.transformers.fit_transformers import (
    GroupByTransformer,
    DiminishingValueTransformer,
    SymmetricDistributionTransformer,
)


def test_min_max_transformer():
    pass


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_sklearn_transformer_wrapper_one_hot_encoder(df):
    sklearn_transformer = OneHotEncoder(handle_unknown="ignore")

    data = df(
        {
            "game_id": [1, 1, 1],
            "position": ["a", "b", "a"],
        }
    )

    transformer = SkLearnTransformerWrapper(
        transformer=sklearn_transformer, features=["position"]
    )

    transformed_df = transformer.fit_transform(data)

    assert transformed_df.shape[1] == 2

    df_future = df(
        {
            "game_id": [1, 2],
            "position": ["a", "c"],
        }
    )

    future_transformed_df = transformer.transform(df_future)

    expected_future_transformed_df = pd.DataFrame(
        {"position_a": [1, 0], "position_b": [0, 0]}
    )
    if isinstance(df_future, pl.DataFrame):
        expected_future_transformed_df = pl.DataFrame(expected_future_transformed_df)
        assert_frame_equal(
            expected_future_transformed_df, future_transformed_df, check_dtype=False
        )
    else:

        pd.testing.assert_frame_equal(
            expected_future_transformed_df, future_transformed_df, check_dtype=False
        )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_sklearn_transformer_wrapper_standard_scaler(df):
    sklearn_transformer = StandardScaler()

    data = df(
        {"game_id": [1, 1, 1], "position": ["a", "b", "a"], "value": [1.2, 0.4, 2.3]}
    )

    transformer = SkLearnTransformerWrapper(
        transformer=sklearn_transformer, features=["value"]
    )

    transformed_df = transformer.fit_transform(data)

    assert transformed_df.shape[1] == 1
    assert "value" in transformed_df.columns

    df_future = df({"game_id": [1, 2], "position": ["a", "c"], "value": [1.2, 0.4]})

    future_transformed_df = transformer.transform(df_future)
    assert future_transformed_df["value"].min() < 0


def test_groupby_transformer_fit_transform():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2, 2],
            "performance": [0.2, 0.3, 0.4, 0.5, 0.6, 0.2],
            "player_id": [1, 2, 3, 1, 2, 3],
        }
    )

    transformer = GroupByTransformer(
        features=["performance"], granularity=["player_id"]
    )

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
    except:
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

    transformer = SymmetricDistributionTransformer(
        features=["performance"], max_iterations=40
    )
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

    transformer = SymmetricDistributionTransformer(
        features=["performance"], max_iterations=40
    )
    fit_transformed_df = transformer.fit_transform(ori_fit_transform_df)
    expected_value_1 = fit_transformed_df.iloc[
        ori_fit_transform_df[ori_fit_transform_df["performance"] == 0.8].index.tolist()[
            0
        ]
    ]["performance"]
    expected_value_2 = fit_transformed_df.iloc[
        ori_fit_transform_df[ori_fit_transform_df["performance"] == 0.1].index.tolist()[
            0
        ]
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
        abs(df[lambda x: x.position == "SG"]["performance"].skew())
        > transformer.skewness_allowed
    )
    assert (
        abs(transformed_df.loc[lambda x: x.position == "SG"]["performance"].skew())
        < transformer.skewness_allowed
    )


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_ratio_team_predictor(df):
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    data = df(
        {
            "performance": [0.1, 0.8, 0.8, 0.8, 0.8],
            "target": [1, 0, 1, 0, 1],
            "team_id": [1, 2, 1, 2, 1],
            "game_id": [1, 1, 2, 2, 1],
        }
    )
    transformer = RatioTeamPredictorTransformer(
        features=["performance"],
        predictor=SklearnPredictor(target="target", estimator=LinearRegression()),
    )

    fit_transformed_data = transformer.fit_transform(data, column_names=column_names)
    assert transformer.predictor.pred_column in fit_transformed_data.columns
    assert len(transformer.features_out) == 2
    for col in transformer.features_out:
        assert col in fit_transformed_data.columns
