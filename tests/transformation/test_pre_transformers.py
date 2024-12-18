import pandas as pd
from player_performance_ratings.predictor_transformer import SkLearnTransformerWrapper
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from player_performance_ratings.transformers.performances_transformers import (
    DiminishingValueTransformer,
    GroupByTransformer,
    SymmetricDistributionTransformer,
)


def test_min_max_transformer():
    pass


def test_sklearn_transformer_wrapper_one_hot_encoder():
    sklearn_transformer = OneHotEncoder(handle_unknown="ignore")

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "position": ["a", "b", "a"],
        }
    )

    transformer = SkLearnTransformerWrapper(
        transformer=sklearn_transformer, features=["position"]
    )

    transformed_df = transformer.fit_transform(df)

    assert transformed_df.shape[1] == 2

    df_future = pd.DataFrame(
        {
            "game_id": [1, 2],
            "position": ["a", "c"],
        }
    )

    future_transformed_df = transformer.transform(df_future)

    expected_future_transformed_df = pd.DataFrame(
        {"position_a": [1, 0], "position_b": [0, 0]}
    )

    pd.testing.assert_frame_equal(
        expected_future_transformed_df, future_transformed_df, check_dtype=False
    )


def test_sklearn_transformer_wrapper_standard_scaler():
    sklearn_transformer = StandardScaler()

    df = pd.DataFrame(
        {"game_id": [1, 1, 1], "position": ["a", "b", "a"], "value": [1.2, 0.4, 2.3]}
    )

    transformer = SkLearnTransformerWrapper(
        transformer=sklearn_transformer, features=["value"]
    )

    transformed_df = transformer.fit_transform(df)

    assert transformed_df.shape[1] == 1

    df_future = pd.DataFrame(
        {"game_id": [1, 2], "position": ["a", "c"], "value": [1.2, 0.4]}
    )

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


def test_reverse_diminshing_value_transformer():
    df = pd.DataFrame(
        {
            "performance": [0.1, 0.8, 0.8, 0.8, 0.8],
            "player_id": [1, 2, 3, 1, 2],
        }
    )

    transformer = DiminishingValueTransformer(features=["performance"], reverse=True)

    ori_df = df.copy()
    transformed_df = transformer.fit_transform(df)

    assert transformed_df["performance"].iloc[0] > ori_df["performance"].iloc[0]
    assert transformed_df["performance"].iloc[3] == ori_df["performance"].iloc[3]


def test_symmetric_distribution_transformery_fit_transform():
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
        features=["performance"], max_iterations=40
    )
    transformed_df = transformer.fit_transform(df)
    assert abs(df["performance"].skew()) > transformer.skewness_allowed

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
