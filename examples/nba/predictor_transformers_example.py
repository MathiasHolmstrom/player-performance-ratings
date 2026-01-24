"""
Predictor Transformers Example - Hierarchical Modeling

This example demonstrates how to use predictions from one model as features for another model.
This is a common pattern in sports analytics where team-level predictions (game winner)
inform player-level predictions (individual player points).

Key concepts covered:
- predictor_transformers: Chain multiple estimators together
- EstimatorTransformer: Wrap an estimator to output predictions as features
- context_feature_names: Columns needed by transformers but not the final estimator
- Hierarchical modeling: Team strength → Player performance
"""

import polars as pl
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

from examples import get_sub_sample_nba_data
from spforge import AutoPipeline, FeatureGeneratorPipeline
from spforge.data_structures import ColumnNames
from spforge.feature_generator import LagTransformer
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures
from spforge.transformers import EstimatorTransformer

# Load sample NBA data
df = get_sub_sample_nba_data(as_pandas=False, as_polars=True)

# Define column mappings
column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_id",
)

# Sort data chronologically (critical for temporal correctness)
df = df.sort(
    [
        column_names.start_date,
        column_names.match_id,
        column_names.team_id,
        column_names.player_id,
    ]
)

# Filter to valid games
df = (
    df.with_columns(
        pl.col(column_names.team_id)
        .n_unique()
        .over(column_names.match_id)
        .alias("team_count")
    )
    .filter(pl.col("team_count") == 2)
    .drop("team_count")
)

# Train/test split (using temporal ordering)
most_recent_10_games = (
    df.select(pl.col(column_names.match_id))
    .unique(maintain_order=True)
    .tail(10)
    .get_column(column_names.match_id)
    .to_list()
)
train_df = df.filter(~pl.col(column_names.match_id).is_in(most_recent_10_games))
test_df = df.filter(pl.col(column_names.match_id).is_in(most_recent_10_games))

train_games = train_df.select(pl.col(column_names.match_id).n_unique()).to_series().item()
test_games = test_df.select(pl.col(column_names.match_id).n_unique()).to_series().item()
print(f"Training: {len(train_df)} rows, {train_games} games")
print(f"Testing: {len(test_df)} rows, {test_games} games")
print()

# ====================================================================
# STEP 1: Generate baseline features (ratings + lags)
# ====================================================================

# Player rating based on historical performance
player_rating_generator = PlayerRatingGenerator(
    performance_column="points",
    auto_scale_performance=True,
    column_names=column_names,
    features_out=[RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED],
)

# Lag features: previous game stats
lag_transformer = LagTransformer(
    features=["points", "minutes"],
    lag_length=2,
    granularity=["player_id"],
)

# Create feature pipeline
features_pipeline = FeatureGeneratorPipeline(
    column_names=column_names,
    feature_generators=[player_rating_generator, lag_transformer],
)

# Generate features
train_df = features_pipeline.fit_transform(train_df).to_pandas()
test_df = features_pipeline.future_transform(test_df).to_pandas()

print(f"Generated {len(features_pipeline.features_out)} baseline features")
print()

# ====================================================================
# STEP 2: Create hierarchical model with predictor_transformers
# ====================================================================
#
# The idea: Team strength affects individual player performance.
# Strategy:
# 1. Predict game winner (team-level) using team ratings
# 2. Use that prediction as a feature for player points (player-level)
#
# This is implemented using predictor_transformers in AutoPipeline
# ====================================================================

# STAGE 1 TRANSFORMER: Predict raw player points estimate
# This creates a point estimate that will be used by the final model
# as an additional feature alongside the original features
points_estimate_transformer = EstimatorTransformer(
    prediction_column_name="points_estimate_raw",  # Name of the output column
    estimator=LGBMRegressor(verbose=-100, n_estimators=30),
)

# STAGE 2 (FINAL ESTIMATOR): Predict player points using Stage 1 output
# This will use:
# - All baseline features (ratings, lags)
# - points_estimate_raw from Stage 1
# The final model can learn from the raw estimate and refine it
player_points_pipeline = AutoPipeline(
    estimator=LGBMRegressor(verbose=-100, n_estimators=50),
    # Features for the final estimator (only pre-game information)
    # Note: points_estimate_raw will be added by the transformer
    estimator_features=features_pipeline.features_out,
    # The predictor_transformers parameter chains the estimators
    predictor_transformers=[points_estimate_transformer],  # Stage 1 executes first
)

# ====================================================================
# UNDERSTANDING THE FLOW
# ====================================================================
# During fit():
#   1. Stage 1 (points_estimate_transformer) fits on train_df using y="points"
#   2. Stage 1 generates "points_estimate_raw" column
#   3. Stage 2 (final LGBMRegressor) fits using features + points_estimate_raw
#
# During predict():
#   1. Stage 1 generates "points_estimate_raw" for test_df
#   2. Stage 2 uses that + other features to predict final player points
#
# Why this helps: The final model can learn from the raw estimate and
# refine it, potentially correcting systematic biases or incorporating
# additional context that the first model missed.
# ====================================================================

print("Training two-stage model...")
print("  Stage 1: Raw points estimator")
print("  Stage 2: Final points estimator (refines Stage 1 output)")
print()

# Fit the pipeline
# The y target here is for the FINAL estimator (player points)
# Predictor_transformers are trained on the same target during fit()
player_points_pipeline.fit(X=train_df, y=train_df["points"])

print("Training complete!")
print()

# ====================================================================
# MAKE PREDICTIONS
# ====================================================================

predictions = player_points_pipeline.predict(test_df)
test_df["predicted_points"] = predictions

# Evaluate
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(test_df["points"], test_df["predicted_points"])
print(f"Mean Absolute Error (two-stage model): {mae:.2f} points")
print()

# Show sample predictions
print("Sample predictions (first 5 players in test set):")
sample = test_df.head(5)[[
    "player_id",
    "points",
    "predicted_points",
    player_rating_generator.features_out[0]  # Show rating as context
]]
print(sample.to_string(index=False))
print()

# ====================================================================
# COMPARE WITH SINGLE-STAGE BASELINE
# ====================================================================
# To verify the benefit of predictor_transformers, compare with a single model

single_stage_pipeline = AutoPipeline(
    estimator=LGBMRegressor(verbose=-100, n_estimators=50),
    estimator_features=features_pipeline.features_out,
)

print("Training single-stage baseline for comparison...")
single_stage_pipeline.fit(X=train_df, y=train_df["points"])
single_stage_predictions = single_stage_pipeline.predict(test_df)

mae_single = mean_absolute_error(test_df["points"], single_stage_predictions)

print(f"  MAE (single-stage baseline): {mae_single:.2f} points")
print(f"  MAE (two-stage model): {mae:.2f} points")
improvement = ((mae_single - mae) / mae_single) * 100
if improvement > 0:
    print(f"  Improvement from two-stage approach: {improvement:.1f}%")
else:
    print(f"  Note: Two-stage model similar to single-stage (typical with small datasets)")
print()

# ====================================================================
# KEY TAKEAWAYS
# ====================================================================
# 1. predictor_transformers chains estimators: output of one → input to next
# 2. EstimatorTransformer wraps any sklearn-compatible estimator
# 3. All transformers share the same target (y) during fit()
# 4. The pattern enables ensemble-like modeling where models refine each other
# 5. Common use cases:
#    - Generate point estimates for distribution models
#    - Multi-stage refinement of predictions
#    - Combining different model types (linear → tree-based)
# 6. All transformers execute during both fit() and predict()
