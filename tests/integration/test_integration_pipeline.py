import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
import polars as pl

from examples import get_sub_sample_nba_data, get_sub_sample_lol_data
from spforge import Pipeline, ColumnNames
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.predictor import SklearnPredictor
from spforge.ratings import RatingKnownFeatures, PlayerRatingGenerator
from spforge.transformers import (
    RollingWindowTransformer,
    LagTransformer,
    RollingMeanDaysTransformer,
    BinaryOutcomeRollingMeanTransformer,
)


def test_nba_integration_pipeline_cross_validate_train_future_predict():
    """
    Asserts future_predicted features are the same as cross-validated.
    Asserts cross-validated predictions are reasonably close to future predictions
    """
    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_name",
        start_date="start_date",
    )
    data: pl.DataFrame = get_sub_sample_nba_data(as_polars=True)

    pipeline = Pipeline(
        rating_generators=[
            PlayerRatingGenerator(
                features_out=[RatingKnownFeatures.PLAYER_RATING],
                performance_column="won",
            )
        ],
        lag_transformers=[
            RollingMeanDaysTransformer(
                days=20,
                features=["points"],
                granularity=["player_id"],
            ),
            RollingWindowTransformer(
                features=["points"], window=15, granularity=["player_id"]
            ),
            LagTransformer(
                features=["points"], lag_length=3, granularity=["player_id"]
            ),
        ],
        predictor=SklearnPredictor(
            estimator=LGBMRegressor(max_depth=2, random_state=42, verbose=-100),
            target="points",
            pred_column="points_estimate",
        ),
        column_names=column_names,
    )

    game_ids = data[column_names.match_id].unique(maintain_order=True)
    hist_game_ids = game_ids[:-20]
    hist_data = data.filter(pl.col(column_names.match_id).is_in(hist_game_ids))
    future_data = data.filter(pl.col(column_names.match_id).is_in(game_ids[-20:]))
    pipeline.train(hist_data)
    future_data = pipeline.predict(future_data, return_features=True)

    min_validation_date = data.filter(
        pl.col(column_names.match_id) == hist_game_ids[len(hist_game_ids) - 20]
    )[column_names.start_date].min()

    cross_validator = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        predictor=pipeline,
        min_validation_date=min_validation_date,
    )
    validation_data = cross_validator.generate_validation_df(data, return_features=True)
    validation_future_data = validation_data.filter(
        pl.col(column_names.match_id).is_in(game_ids[-20:])
    )
    future_data = future_data.with_columns(pl.col(pipeline.features).fill_nan(-999992))
    validation_future_data = validation_future_data.with_columns(
        pl.col(pipeline.features).fill_nan(-999992)
    )
    validation_cols_renamed = {c: f"{c}_cv" for c in pipeline.features}
    validation_future_player_grouped = (
        validation_future_data.group_by(column_names.player_id)
        .agg(pl.col(pipeline.features).first())
        .rename(validation_cols_renamed)
    )
    future_joined_data = validation_future_player_grouped.join(
        future_data, on=column_names.player_id
    )

    for feat in pipeline.features:
        assert (
            future_joined_data[f"{feat}_cv"].to_list()
            == future_joined_data[feat].to_list()
        )

    validation_predictions = validation_future_data.select(
        [pipeline.pred_column, column_names.match_id, column_names.player_id]
    ).rename({pipeline.pred_column: "cv_prediction"})

    future_data_with_cv_predictions = future_data.join(
        validation_predictions, on=[column_names.match_id, column_names.player_id]
    )
    future_data_with_cv_predictions = future_data_with_cv_predictions.with_columns(
        (pl.col("cv_prediction") - pl.col(pipeline.pred_column)).abs().alias("abs_diff")
    )
    abs_diff_mean = future_data_with_cv_predictions["abs_diff"].mean()
    assert abs_diff_mean < 0.7
    mean_cv_prediction = future_data_with_cv_predictions["cv_prediction"].mean()
    mean_future_prediction = future_data_with_cv_predictions[
        pipeline.pred_column
    ].mean()
    assert abs(mean_cv_prediction - mean_future_prediction) < 0.1


def test_lol_integration_pipeline_cross_validate_train_future_predic():
    """
    Asserts future_predicted features are the same as cross-validated.
    Asserts cross-validated predictions are reasonably close to future predictions
    """
    column_names = ColumnNames(
        match_id="gameid",
        team_id="teamname",
        player_id="playername",
        start_date="date",
    )
    data: pd.DataFrame = get_sub_sample_lol_data(as_pandas=True)
    data = (
        data.loc[lambda x: x.position != "team"]
        .assign(team_count=data.groupby("gameid")["teamname"].transform("nunique"))
        .loc[lambda x: x.team_count == 2]
        .assign(
            player_count=data.groupby(["gameid", "teamname"])["playername"].transform(
                "nunique"
            )
        )
        .loc[lambda x: x.player_count == 5]
    )
    data = data.assign(
        team_count=data.groupby("gameid")["teamname"].transform("nunique")
    ).loc[lambda x: x.team_count == 2]
    data = pl.DataFrame(data)

    pipeline = Pipeline(
        rating_generators=[
            PlayerRatingGenerator(
                features_out=[RatingKnownFeatures.PLAYER_RATING],
                performance_column="result",
            )
        ],
        lag_transformers=[
            RollingMeanDaysTransformer(
                days=20,
                features=["kills"],
                granularity=["playername"],
            ),
            RollingWindowTransformer(
                features=["kills"], window=15, granularity=[column_names.player_id]
            ),
            LagTransformer(
                features=["kills"], lag_length=3, granularity=[column_names.player_id]
            ),
        ],
        predictor=SklearnPredictor(
            estimator=LGBMRegressor(max_depth=2, random_state=42, verbose=-100),
            target="kills",
        ),
        column_names=column_names,
    )

    game_ids = data[column_names.match_id].unique(maintain_order=True)
    hist_game_ids = game_ids[:-40]
    hist_data = data.filter(pl.col(column_names.match_id).is_in(hist_game_ids))
    future_data = data.filter(pl.col(column_names.match_id).is_in(game_ids[-40:]))
    pipeline.train(hist_data)
    future_data = pipeline.predict(future_data, return_features=True)
    min_validation_date = data.filter(
        pl.col(column_names.match_id) == hist_game_ids[len(hist_game_ids) - 60]
    )[column_names.start_date].min()

    cross_validator = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        predictor=pipeline,
        min_validation_date=min_validation_date,
    )

    validation_data = cross_validator.generate_validation_df(data, return_features=True)
    validation_future_data = validation_data.filter(
        pl.col(column_names.match_id).is_in(game_ids[-40:])
    )
    future_data = future_data.with_columns(pl.col(pipeline.features).fill_nan(-999992))
    validation_future_data = validation_future_data.with_columns(
        pl.col(pipeline.features).fill_nan(-999992)
    )
    validation_cols_renamed = {c: f"{c}_cv" for c in pipeline.features}
    validation_future_player_grouped = (
        validation_future_data.group_by(column_names.player_id)
        .agg(pl.col(pipeline.features).first())
        .rename(validation_cols_renamed)
    )

    future_joined_data = validation_future_player_grouped.join(
        future_data, on=column_names.player_id
    )
    for feat in pipeline.features:
        assert (
            future_joined_data[f"{feat}_cv"].to_list()
            == future_joined_data[feat].to_list()
        )

    validation_predictions = validation_future_data.select(
        [pipeline.pred_column, column_names.match_id, column_names.player_id]
    ).rename({pipeline.pred_column: "cv_prediction"})

    future_data_with_cv_predictions = future_data.join(
        validation_predictions, on=[column_names.match_id, column_names.player_id]
    )
    future_data_with_cv_predictions = future_data_with_cv_predictions.with_columns(
        (pl.col("cv_prediction") - pl.col(pipeline.pred_column)).abs().alias("abs_diff")
    )
    abs_diff_mean = future_data_with_cv_predictions["abs_diff"].mean()
    assert abs_diff_mean < 0.28
    mean_cv_prediction = future_data_with_cv_predictions["cv_prediction"].mean()
    mean_future_prediction = future_data_with_cv_predictions[
        pipeline.pred_column
    ].mean()
    assert abs(mean_cv_prediction - mean_future_prediction) < 0.1
