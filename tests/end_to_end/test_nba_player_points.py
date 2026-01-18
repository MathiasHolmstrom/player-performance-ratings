import polars as pl
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error

from examples import get_sub_sample_nba_data
from spforge import FeatureGeneratorPipeline
from spforge.autopipeline import AutoPipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.data_structures import ColumnNames
from spforge.distributions import (
    NegativeBinomialEstimator,
)
from spforge.estimator import (
    SkLearnEnhancerEstimator,
)
from spforge.feature_generator import (
    LagTransformer,
    RollingWindowTransformer,
)
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures, TeamRatingGenerator
from spforge.scorer import Filter, Operator, OrdinalLossScorer, SklearnScorer
from spforge.transformers import EstimatorTransformer, RatioEstimatorTransformer


@pytest.mark.parametrize("dataframe_type", ["pl", "pd"])
def test_nba_player_points(dataframe_type):
    df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)
    column_names = ColumnNames(
        team_id="team_id",
        match_id="game_id",
        start_date="start_date",
        player_id="player_id",
        participation_weight="minutes_ratio",
    )
    df = df.sort(
        [
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]
    )

    df = df.with_columns(
        [
            (pl.col("minutes") / pl.lit(48.25)).alias("minutes_ratio"),
            pl.col("points").sum().over("game_id").alias("total_points"),
            pl.col("points").sum().over(["game_id", "team_id"]).alias("team_points"),
            (pl.col("points") / pl.col("minutes")).alias("points_per_minute"),
        ]
    )

    df = df.with_columns(
        pl.when(pl.col("minutes") == 0)
        .then(pl.lit(0))
        .otherwise(pl.col("points_per_minute"))
        .alias("points_per_minute")
    )
    df = df.with_columns(pl.col("points").clip(0, 40).alias("points"))
    total_points_rating_generator = TeamRatingGenerator(
        performance_column="total_points",
        auto_scale_performance=True,
        performance_predictor="mean",
    )

    player_points_rating_generator = PlayerRatingGenerator(
        performance_column="points_per_minute",
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED],
        non_predictor_features_out=[
            RatingKnownFeatures.PLAYER_OFF_RATING,
            RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED,
        ],
    )

    player_plus_minus_rating_generator = PlayerRatingGenerator(
        performance_column="plus_minus",
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED],
    )

    features_generator = FeatureGeneratorPipeline(
        column_names=column_names,
        feature_generators=[
            player_plus_minus_rating_generator,
            player_points_rating_generator,
            total_points_rating_generator,
            RollingWindowTransformer(features=["points"], window=15, granularity=["player_id"]),
            LagTransformer(features=["points"], lag_length=3, granularity=["player_id"]),
        ],
    )
    if dataframe_type == "pd":
        df = df.to_pandas()
    df = features_generator.fit_transform(df)

    game_winner_pipeline = AutoPipeline(
        granularity=["game_id", "team_id"],
        estimator=LogisticRegression(),
        estimator_features=player_plus_minus_rating_generator.features_out + ["location"],
    )
    cross_validator_game_winnner = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=game_winner_pipeline,
        prediction_column_name="game_winner_probability",
        target_column="won",
        features=game_winner_pipeline.required_features,
    )
    pre_row_count = len(df)
    df = cross_validator_game_winnner.generate_validation_df(df=df, add_training_predictions=True)
    assert pre_row_count == len(df)
    assert cross_validator_game_winnner.prediction_column_name in df.columns
    game_winner_score = SklearnScorer(
        scorer_function=log_loss,
        filters=[Filter(column_name="is_validation", value=1, operator=Operator.EQUALS)],
        pred_column=cross_validator_game_winnner.prediction_column_name,
        target=cross_validator_game_winnner.target_column,
    )
    game_winner_log_loss = game_winner_score.score(df)
    assert game_winner_log_loss < 0.67

    negative_binomial = NegativeBinomialEstimator(
        max_value=40,
        point_estimate_pred_column="points_estimate",
        r_specific_granularity=["player_id"],
        predicted_r_weight=1,
        column_names=column_names,
    )

    estimator_transformer_raw = EstimatorTransformer(
        features=features_generator.features_out
        + ["location", column_names.start_date, "game_winner_probability"],
        prediction_column_name="points_estimate_raw",
        estimator=SkLearnEnhancerEstimator(
            estimator=LGBMRegressor(verbose=-100, random_state=42, max_depth=2),
            date_column=column_names.start_date,
            day_weight_epsilon=0.1,
        ),
    )
    team_ratio_transformer = RatioEstimatorTransformer(
        estimator=LGBMRegressor(verbose=-100),
        features=features_generator.features_out,
        predict_row=False,
        prediction_column_name=estimator_transformer_raw.prediction_column_name,
        granularity=[column_names.match_id, column_names.team_id],
    )
    estimator_transformer_final = EstimatorTransformer(
        features=features_generator.features_out + ["location", column_names.start_date],
        prediction_column_name="points_estimate",
        estimator=SkLearnEnhancerEstimator(
            estimator=LGBMRegressor(verbose=-100, random_state=42),
            date_column=column_names.start_date,
            day_weight_epsilon=0.1,
        ),
    )
    pipeline = AutoPipeline(
        estimator=negative_binomial,
        estimator_features=features_generator.features_out
        + ["location", "game_winner_probability"],
        predictor_transformers=[
            estimator_transformer_raw,
            team_ratio_transformer,
            estimator_transformer_final,
        ],
    )

    cross_validator = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=pipeline,
        prediction_column_name="points_probabilities",
        target_column="points",
        features=pipeline.required_features,
    )
    validation_df = cross_validator.generate_validation_df(df=df, add_training_predictions=True)
    if isinstance(validation_df, pl.DataFrame):
        validation_df = validation_df.to_pandas()
    high_player_rating_rows = validation_df[
        validation_df[player_points_rating_generator.PLAYER_DIFF_PROJ_COL] > 0
    ]
    low_points_prediction = validation_df[
        validation_df[player_points_rating_generator.PLAYER_DIFF_PROJ_COL] < 0
    ]
    assert (
        high_player_rating_rows["points_estimate"].mean()
        > low_points_prediction["points_estimate"].mean()
    )
    assert high_player_rating_rows["points"].mean() > low_points_prediction["points"].mean()
    assert high_player_rating_rows["__ratio"].mean() > low_points_prediction["__ratio"].mean()

    mean_absolute_scorer = SklearnScorer(
        pred_column=pipeline.predictor_transformers[0].prediction_column_name,
        target=cross_validator.target_column,
        scorer_function=mean_absolute_error,
        validation_column="is_validation",
        filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
    )

    mae_score = mean_absolute_scorer.score(validation_df)
    assert mae_score < 4.53

    validation_df["mean_points_per_minute"] = validation_df["points_per_minute"].mean()
    validation_df["dummy_prediction"] = (
        validation_df["mean_points_per_minute"] * validation_df["minutes"]
    )
    mean_dummy_absolute_scorer = SklearnScorer(
        pred_column="dummy_prediction",
        target=cross_validator.target_column,
        scorer_function=mean_absolute_error,
        validation_column="is_validation",
        filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
    )

    mae_dummy_score = mean_dummy_absolute_scorer.score(validation_df)
    assert mae_dummy_score > mae_score

    ordinal_scorer = OrdinalLossScorer(
        pred_column=cross_validator.prediction_column_name,
        target=cross_validator.target_column,
        validation_column="is_validation",
        filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
        classes=range(0, negative_binomial.max_value + 1),
    )
    ordinal_loss_score = ordinal_scorer.score(validation_df)

    print(f"Ordinal Loss {ordinal_loss_score}")
