import pandas as pd

from player_performance_ratings.transformation.pre_transformers import GroupByTransformer, DiminishingValueTransformer


def test_groupby_transformer():
    df = pd.DataFrame({
        'game_id': [1, 1, 1, 2, 2, 2],
        "performance": [0.2, 0.3, 0.4, 0.5, 0.6, 0.2],
        "player_id": [1, 2, 3, 1, 2, 3],
    })

    transformer = GroupByTransformer(
        features=['performance'],
        granularity=["player_id"]
    )

    expected_df = df.copy()
    expected_df[transformer.prefix + "performance"] = [0.35, 0.45, 0.3, 0.35, 0.45, 0.3]

    transformed_df = transformer.transform(df)
    pd.testing.assert_frame_equal(expected_df, transformed_df)


def test_diminshing_value_transformer():
    df = pd.DataFrame({
        "performance": [0.2, 0.2, 0.2, 0.2, 0.9],
        "player_id": [1, 2, 3, 1, 2],
    })

    transformer = DiminishingValueTransformer(features=['performance'])

    ori_df = df.copy()
    transformed_df = transformer.transform(df)

    assert transformed_df['performance'].iloc[4] < ori_df['performance'].iloc[4]
    assert transformed_df['performance'].iloc[0] == ori_df['performance'].iloc[0]


def test_reverse_diminshing_value_transformer():
    df = pd.DataFrame({
        "performance": [0.1, 0.8, 0.8, 0.8, 0.8],
        "player_id": [1, 2, 3, 1, 2],
    })

    transformer = DiminishingValueTransformer(features=['performance'], reverse=True)

    ori_df = df.copy()
    transformed_df = transformer.transform(df)

    assert transformed_df['performance'].iloc[0] > ori_df['performance'].iloc[0]
    assert transformed_df['performance'].iloc[3] == ori_df['performance'].iloc[3]
