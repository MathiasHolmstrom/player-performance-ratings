import pandas as pd
import pytest
import polars as pl

from player_performance_ratings import ColumnNames
from player_performance_ratings.pipeline_transformer import PipelineTransformer
from player_performance_ratings.ratings import (
    UpdateRatingGenerator,
    RatingKnownFeatures,
)
from player_performance_ratings.ratings.performance_generator import (
    ColumnWeight,
    Performance,
    PerformancesGenerator,
)
from player_performance_ratings.transformers import LagTransformer


@pytest.mark.parametrize("to_polars", [True, False])
def test_pipeline_transformer(to_polars):
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "player_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "team_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
            "deaths": [0.5, 1, 0.7, 2, 0.5, 0.7, 0.2, 2.1, 0.8, 1],
            "kills": [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3],
            "__target": [1, 0, 0.6, 0.3, 0.8, 0.2, 0.4, 0.1, 1, 0],
        }
    )

    column_weights = [ColumnWeight(name="kills", weight=1)]
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = UpdateRatingGenerator(
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
    )

    pipeline = PipelineTransformer(
        performances_generator=PerformancesGenerator(
            Performance(weights=column_weights)
        ),
        rating_generators=rating_generator,
        lag_generators=[
            LagTransformer(
                features=["kills", "deaths"], lag_length=1, granularity=["player_id"]
            )
        ],
        column_names=column_names,
    )


    historical_df = df[df[column_names.start_date] < pd.to_datetime("2023-01-04")]
    future_df = df[df[column_names.start_date] >= pd.to_datetime("2023-01-04")]

    if to_polars:
        historical_df = pl.DataFrame(df)
        future_df = pl.DataFrame(df)

    historical_df_predictions = pipeline.fit_transform(df=historical_df)
    future_predict = pipeline.transform(df=future_df)
    if to_polars:
        historical_df_predictions = historical_df_predictions.to_pandas()
        future_predict = future_predict.to_pandas()

    for lag_generator in pipeline.lag_generators:
        for feature_out in lag_generator.features_out:
            assert feature_out in historical_df_predictions.columns
            assert feature_out in future_predict.columns

    for rating_feature in rating_generator.features_out:
        assert rating_feature in historical_df_predictions.columns
        assert rating_feature in future_predict.columns

    for post_lag_transformer in pipeline.post_lag_transformers:
        assert post_lag_transformer in historical_df_predictions.columns
        assert post_lag_transformer in future_predict.columns
