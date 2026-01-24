import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_absolute_error

from examples import get_sub_sample_nba_data
from spforge import FeatureGeneratorPipeline
from spforge.autopipeline import AutoPipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.data_structures import ColumnNames
from spforge.distributions import NegativeBinomialEstimator
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

print("\n" + "=" * 70)
print("Comparison: LGBMClassifier vs LGBMRegressor + NegativeBinomial")
print("=" * 70)

# Approach 1: LGBMClassifier (directly outputs probability distribution)
print("\nApproach 1: LGBMClassifier (direct probability prediction)")
print("-" * 70)
pipeline_classifier = AutoPipeline(
    estimator=LGBMClassifier(verbose=-100, random_state=42),
    estimator_features=features_generator.features_out,
)

cross_validator_classifier = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    estimator=pipeline_classifier,
    prediction_column_name="points_probabilities_classifier",
    target_column="points",
    features=pipeline_classifier.required_features,
)
validation_df_classifier = cross_validator_classifier.generate_validation_df(df=df)

ordinal_scorer_classifier = OrdinalLossScorer(
    pred_column="points_probabilities_classifier",
    target="points",
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
    classes=range(0, 41),
)
ordinal_loss_classifier = ordinal_scorer_classifier.score(validation_df_classifier)
print(f"Ordinal Loss: {ordinal_loss_classifier:.4f}")

# Approach 2: LGBMRegressor + NegativeBinomialEstimator
print("\nApproach 2: LGBMRegressor + NegativeBinomialEstimator")
print("-" * 70)
predictor_negbin = NegativeBinomialEstimator(
    max_value=40,
    point_estimate_pred_column="points_estimate",
    predicted_r_weight=1,
    column_names=column_names,
)

pipeline_negbin = AutoPipeline(
    estimator=predictor_negbin,
    estimator_features=features_generator.features_out,
    predictor_transformers=[
        EstimatorTransformer(
            prediction_column_name="points_estimate",
            estimator=LGBMRegressor(verbose=-100, random_state=42),
            features=features_generator.features_out,
        )
    ],
)

cross_validator_negbin = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    estimator=pipeline_negbin,
    prediction_column_name="points_probabilities_negbin",
    target_column="points",
    features=pipeline_negbin.required_features,
)
validation_df_negbin = cross_validator_negbin.generate_validation_df(df=df)

ordinal_scorer_negbin = OrdinalLossScorer(
    pred_column="points_probabilities_negbin",
    target="points",
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
    classes=range(0, predictor_negbin.max_value + 1),
)
ordinal_loss_negbin = ordinal_scorer_negbin.score(validation_df_negbin)
print(f"Ordinal Loss: {ordinal_loss_negbin:.4f}")

# Also show MAE from point estimate
mean_absolute_scorer = SklearnScorer(
    pred_column=pipeline_negbin.predictor_transformers[0].prediction_column_name,
    target="points",
    scorer_function=mean_absolute_error,
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
)
mae_score = mean_absolute_scorer.score(validation_df_negbin)
print(f"Point Estimate MAE: {mae_score:.4f}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"LGBMClassifier Ordinal Loss:              {ordinal_loss_classifier:.4f}")
print(f"LGBMRegressor + NegativeBinomial Ordinal Loss: {ordinal_loss_negbin:.4f}")
print(f"LGBMRegressor + NegativeBinomial Point Est MAE: {mae_score:.4f}")
print("\nBoth approaches output probability distributions over 0-40 points.")
print("NegativeBinomial also provides point estimates from the underlying regressor.")
