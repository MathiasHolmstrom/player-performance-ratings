import polars as pl
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from examples import get_sub_sample_nba_data
from spforge import FeatureGeneratorPipeline
from spforge.autopipeline import AutoPipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.data_structures import ColumnNames
from spforge.estimator import NegativeBinomialEstimator
from spforge.feature_generator import (
    LagTransformer,
    RollingWindowTransformer,
)
from spforge.scorer import Filter, Operator, OrdinalLossScorer, SklearnScorer
from spforge.transformers import EstimatorTransformer

df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)
column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_id",
)
df = df.sort(
    [
        column_names.start_date,
        column_names.match_id,
        column_names.team_id,
        column_names.player_id,
    ]
)

df = df.with_columns(pl.col("points").clip(0, 40).alias("points"))

features_generator = FeatureGeneratorPipeline(
    column_names=column_names,
    feature_generators=[
        RollingWindowTransformer(features=["points"], window=15, granularity=["player_id"]),
        LagTransformer(features=["points"], lag_length=3, granularity=["player_id"]),
    ],
)
df = features_generator.fit_transform(df)

print("\n" + "=" * 60)
print("Comparison: LGBMRegressor vs NegativeBinomialEstimator")
print("=" * 60)

# Approach 1: Simple LGBMRegressor (point estimate only)
print("\nApproach 1: LGBMRegressor (point estimate)")
print("-" * 60)
pipeline_simple = AutoPipeline(
    estimator=LGBMRegressor(verbose=-100, random_state=42),
    feature_names=features_generator.features_out,
)

cross_validator_simple = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    estimator=pipeline_simple,
    prediction_column_name="points_pred_simple",
    target_column="points",
    features=pipeline_simple.feature_names,
)
validation_df_simple = cross_validator_simple.generate_validation_df(df=df)

scorer_simple = SklearnScorer(
    pred_column="points_pred_simple",
    target="points",
    scorer_function=mean_absolute_error,
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
)

mae_simple = scorer_simple.score(validation_df_simple)
print(f"Validation MAE: {mae_simple:.4f}")

# Approach 2: NegativeBinomialEstimator (full probability distribution)
print("\nApproach 2: NegativeBinomialEstimator (probability distribution)")
print("-" * 60)
predictor = NegativeBinomialEstimator(
    max_value=40,
    point_estimate_pred_column="points_estimate",
    r_specific_granularity=["player_id"],
    predicted_r_weight=1,
    column_names=column_names,
)

pipeline_dist = AutoPipeline(
    estimator=predictor,
    feature_names=features_generator.features_out,
    context_feature_names=[
        column_names.player_id,
        column_names.start_date,
        column_names.team_id,
        column_names.match_id,
    ],
    predictor_transformers=[
        EstimatorTransformer(
            prediction_column_name="points_estimate",
            estimator=LGBMRegressor(verbose=-100, random_state=42),
            features=features_generator.features_out,
        )
    ],
)

cross_validator_dist = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    estimator=pipeline_dist,
    prediction_column_name="points_probabilities",
    target_column="points",
    features=pipeline_dist.context_feature_names + pipeline_dist.feature_names,
)
validation_df_dist = cross_validator_dist.generate_validation_df(df=df)

mean_absolute_scorer = SklearnScorer(
    pred_column=pipeline_dist.predictor_transformers[0].prediction_column_name,
    target=cross_validator_dist.target_column,
    scorer_function=mean_absolute_error,
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
)

mae_dist = mean_absolute_scorer.score(validation_df_dist)
print(f"Point Estimate MAE: {mae_dist:.4f}")

ordinal_scorer = OrdinalLossScorer(
    pred_column=cross_validator_dist.prediction_column_name,
    target=cross_validator_dist.target_column,
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
    classes=range(0, predictor.max_value + 1),
)
ordinal_loss_score = ordinal_scorer.score(validation_df_dist)

print(f"Ordinal Loss: {ordinal_loss_score:.4f}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"LGBMRegressor MAE:              {mae_simple:.4f}")
print(f"NegativeBinomial Point Est MAE: {mae_dist:.4f}")
print(f"NegativeBinomial Ordinal Loss:  {ordinal_loss_score:.4f}")
print("\nNote: NegativeBinomialEstimator provides full probability")
print("distributions, enabling uncertainty quantification and")
print("expected value calculations, not just point estimates.")
