import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression
from polars.testing import assert_frame_equal
import pytest

from spforge import ColumnNames
from spforge.predictor import SklearnPredictor, SklearnPredictor
from spforge.predictor_transformer import SkLearnTransformerWrapper
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spforge.transformers import RatioTeamPredictorTransformer
from spforge.transformers.performances_transformers import (
    DiminishingValueTransformer,
    GroupByTransformer,
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
