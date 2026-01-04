import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from spforge import ColumnNames
from spforge.transformers import RatioTeamPredictorTransformer


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_ratio_team_predictor(df):

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
        estimator=LinearRegression(),
    )

    fit_transformed_data = transformer.fit(data, data['target'])
