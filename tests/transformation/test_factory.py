import json

import pandas as pd
from deepdiff import DeepDiff
from sklearn.preprocessing import StandardScaler

from player_performance_ratings.transformation import ColumnWeight, DiminishingValueTransformer, \
    SkLearnTransformerWrapper, MinMaxTransformer, ColumnsWeighter

from player_performance_ratings.transformation.factory import auto_create_pre_transformers


def test_auto_create_pre_transformers():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "player_id": [1, 2, 1, 2],
            "kills": [1, 1, 2, 20],
            "deaths": [4, 3, 2, 3],
        }
    )

    column_weights = [ColumnWeight(name="kills", weight=0.5),
                      ColumnWeight(name="deaths", weight=0.5, lower_is_better=True)]

    pre_transformers = auto_create_pre_transformers(df=df, column_weights=column_weights)

    expected_pre_transformers = [
        DiminishingValueTransformer(features=["kills"]),
        SkLearnTransformerWrapper(transformer=StandardScaler(), features=["kills", "deaths"]),
        MinMaxTransformer(features=["kills", "deaths"]),
        ColumnsWeighter(weighted_column_name="weighted_performance", column_weights=column_weights)
    ]


    for idx, pre_transformer in enumerate(pre_transformers):
        diff = DeepDiff  (pre_transformer, expected_pre_transformers[idx])
        assert diff == {}

