import pandas as pd

from player_performance_ratings import ColumnNames, Pipeline
from player_performance_ratings.pipeline_transformer import PipelineTransformer
from player_performance_ratings.predictor import Predictor
from player_performance_ratings.ratings import (
    UpdateRatingGenerator,
    RatingFutureFeatures,
)
from player_performance_ratings.ratings.performance_generator import (
    ColumnWeight,
    Performance,
    PerformancesGenerator,
)
from player_performance_ratings.transformers import LagTransformer


def test_pipelien_transformer():
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
        future_features_out=[RatingFutureFeatures.RATING_DIFFERENCE_PROJECTED]
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
    historical_df_predictions = pipeline.fit_transform(df=historical_df)

    future_predict = pipeline.transform(df=future_df)

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
