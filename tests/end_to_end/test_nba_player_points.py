import polars as pl
import pytest
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from examples import get_sub_sample_nba_data
from spforge import FeatureGeneratorPipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.data_structures import ColumnNames
from spforge.estimator import (
    NegativeBinomialEstimator, SkLearnEnhancerEstimator,
)
from spforge.feature_generator import (
    LagTransformer,
    RollingWindowTransformer,
)
from spforge.pipeline import Pipeline
from spforge.ratings import TeamRatingGenerator, PlayerRatingGenerator, RatingKnownFeatures
from spforge.scorer import Filter, Operator, OrdinalLossScorer, SklearnScorer
from spforge.transformers import EstimatorTransformer, RatioEstimatorTransformer


@pytest.mark.parametrize('dataframe_type', ['pl', 'pd'])
def test_nba_player_points(dataframe_type):
    df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)
    column_names = ColumnNames(
        team_id="team_id",
        match_id="game_id",
        start_date="start_date",
        player_id="player_id",
        participation_weight='minutes_ratio'
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
        (pl.col('minutes')/pl.lit(48.25)).alias('minutes_ratio')
    )
    df = df.with_columns(
        pl.col('points').sum().over('game_id').alias('total_points')
    )
    df = df.with_columns(pl.col("points").clip(0, 40).alias("points"))
    total_points_rating_generator = TeamRatingGenerator(
        performance_column='total_points',
        auto_scale_performance=True,
        performance_predictor='mean',
    )

    player_points_rating_generator = PlayerRatingGenerator(
        performance_column='points',
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED],
        non_predictor_features_out=[RatingKnownFeatures.PLAYER_OFF_RATING, RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED],
    )

    features_generator = FeatureGeneratorPipeline(
        column_names=column_names,
        feature_generators=[
            player_points_rating_generator,
            total_points_rating_generator,
            RollingWindowTransformer(features=["points"], window=15, granularity=["player_id"]),
            LagTransformer(features=["points"], lag_length=3, granularity=["player_id"]),
        ],
    )
    if dataframe_type == 'pd':
        df = df.to_pandas()
    df = features_generator.fit_transform(df)

    predictor = NegativeBinomialEstimator(
        max_value=40,
        point_estimate_pred_column="points_estimate",
        r_specific_granularity=["player_id"],
        predicted_r_weight=1,
        column_names=column_names,
    )

    estimator_transformer_raw =  EstimatorTransformer(
            features=features_generator.features_out + ['location', column_names.start_date],
            prediction_column_name='points_estimate_raw',
            estimator=SkLearnEnhancerEstimator(
                estimator=LGBMRegressor(verbose=-100, random_state=42, max_depth=2),
                date_column=column_names.start_date,
                day_weight_epsilon=0.1
            )
        )
    team_ratio_transformer = RatioEstimatorTransformer(
        estimator=LGBMRegressor(),
        features=features_generator.features_out,
        predict_row=False,
        prediction_column_name=estimator_transformer_raw.prediction_column_name,
        granularity=[column_names.match_id, column_names.team_id]
    )
    estimator_transformer_final =  EstimatorTransformer(
            features=features_generator.features_out + ['location', column_names.start_date],
            prediction_column_name='points_estimate',
            estimator=SkLearnEnhancerEstimator(
                estimator=LGBMRegressor(verbose=-100, random_state=42),
                date_column=column_names.start_date,
                day_weight_epsilon=0.1
            )
        )
    pipeline = Pipeline(
        convert_cat_features_to_cat_dtype=True,
        estimator=predictor,
        feature_names=features_generator.features_out + ['location'],
        context_feature_names=[column_names.player_id, column_names.start_date, column_names.team_id,
                               column_names.match_id],
        predictor_transformers=[estimator_transformer_raw, team_ratio_transformer, estimator_transformer_final]
    )

    cross_validator = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=pipeline,
        prediction_column_name='points_probabilities',
        target_column='points',
        features=pipeline.context_feature_names + pipeline.feature_names
    )
    validation_df = cross_validator.generate_validation_df(df=df)

    mean_absolute_scorer = SklearnScorer(
        pred_column=pipeline.predictor_transformers[0].prediction_column_name,
        target=cross_validator.target_column,
        scorer_function=mean_absolute_error,
        validation_column="is_validation",
        filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
    )

    mae_score = mean_absolute_scorer.score(validation_df)
    print(f"MAE {mae_score}")

    ordinal_scorer = OrdinalLossScorer(
        pred_column=cross_validator.prediction_column_name,
        target=cross_validator.target_column,
        validation_column="is_validation",
        filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
        classes=range(0, predictor.max_value + 1),
    )
    ordinal_loss_score = ordinal_scorer.score(validation_df)

    print(f"Ordinal Loss {ordinal_loss_score}")
