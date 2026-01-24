"""
Feature Engineering Example

This example demonstrates the core feature generation workflow in spforge,
showing how to create rich features from historical sports data while
maintaining temporal ordering to prevent data leakage.

Key concepts covered:
- PlayerRatingGenerator: Elo-style player ratings
- LagTransformer: Previous match statistics
- RollingWindowTransformer: Rolling averages over recent matches
- FeatureGeneratorPipeline: Chaining feature generators
- State management: fit_transform vs future_transform
"""

import polars as pl

from examples import get_sub_sample_nba_data
from spforge import FeatureGeneratorPipeline
from spforge.data_structures import ColumnNames
from spforge.feature_generator import LagTransformer, RollingWindowTransformer
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures

# Load sample NBA data
df = get_sub_sample_nba_data(as_pandas=False, as_polars=True)

# Define column mappings for your dataset
# This tells spforge which columns contain team IDs, player IDs, dates, etc.
column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_id",
)

# CRITICAL: Always sort data chronologically before generating features
# This ensures temporal ordering and prevents future leakage (using future data to predict the past)
df = df.sort(
    [
        column_names.start_date,  # First by date
        column_names.match_id,  # Then by match
        column_names.team_id,  # Then by team
        column_names.player_id,  # Finally by player
    ]
)

# Keep only games with exactly 2 teams (filter out invalid data)
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

match_count = df.select(pl.col(column_names.match_id).n_unique()).to_series().item()
start_date = df.select(pl.col(column_names.start_date).min()).to_series().item()
end_date = df.select(pl.col(column_names.start_date).max()).to_series().item()
print(f"Dataset: {len(df)} rows, {match_count} games")
print(f"Date range: {start_date} to {end_date}")
print()

# ====================================================================
# FEATURE GENERATORS
# ====================================================================

# 1. PLAYER RATING GENERATOR
# Maintains Elo-style ratings for each player that update after each game
# - Separate offense and defense ratings
# - Ratings increase after strong performances, decrease after weak ones
# - The "rating difference" feature compares a player's rating vs their opponent's
player_rating_generator = PlayerRatingGenerator(
    performance_column="points",  # Use points scored to update ratings
    auto_scale_performance=True,  # Normalize points to 0-1 range automatically
    rating_change_multiplier_offense=50,  # How quickly offense ratings change (higher = more volatile)
    rating_change_multiplier_defense=50,  # How quickly defense ratings change
    column_names=column_names,
    features_out=[
        RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED  # Output pre-match rating difference
    ],
)

# 2. LAG TRANSFORMER
# Provides the player's statistics from their previous N games
# IMPORTANT: Values are shifted by 1 to prevent data leakage
# For example, lag=1 gives you the previous game's stats (not the current game!)
lag_transformer = LagTransformer(
    features=["points", "minutes"],  # Which columns to lag
    lag_length=3,  # Create 3 lag features (t-1, t-2, t-3)
    granularity=["player_id"],  # Calculate lags per player (not globally)
)

# 3. ROLLING WINDOW TRANSFORMER
# Calculates rolling averages over a player's recent games
# Window=10 means "average of the last 10 games"
# Tradeoff: Larger windows = more stable but less reactive to recent form
rolling_transformer = RollingWindowTransformer(
    features=["points"],  # Calculate rolling average of points
    window=10,  # Use last 10 games
    min_periods=1,  # Allow calculation even if < 10 games available
    granularity=["player_id"],  # Calculate per player
)

# ====================================================================
# FEATURE GENERATOR PIPELINE
# ====================================================================

# Chain all feature generators together
# They execute sequentially: ratings → lags → rolling averages
# Each generator:
# - Maintains row count (no rows added or removed)
# - Prevents duplicate feature names
# - Updates internal state during fit_transform/transform
features_pipeline = FeatureGeneratorPipeline(
    column_names=column_names,
    feature_generators=[
        player_rating_generator,
        lag_transformer,
        rolling_transformer,
    ],
)

print("Feature generators in pipeline:")
for i, gen in enumerate(features_pipeline.feature_generators, 1):
    print(f"  {i}. {gen.__class__.__name__}")
print()

# ====================================================================
# STATE MANAGEMENT: THE CRITICAL DISTINCTION
# ====================================================================

# Split data into historical (for training) and future (for prediction)
most_recent_5_games = (
    df.select(pl.col(column_names.match_id))
    .unique(maintain_order=True)
    .tail(5)
    .get_column(column_names.match_id)
    .to_list()
)
historical_df = df.filter(~pl.col(column_names.match_id).is_in(most_recent_5_games))
future_df = df.filter(pl.col(column_names.match_id).is_in(most_recent_5_games))

historical_games = (
    historical_df.select(pl.col(column_names.match_id).n_unique()).to_series().item()
)
future_games = future_df.select(pl.col(column_names.match_id).n_unique()).to_series().item()
print(f"Historical data: {len(historical_df)} rows, {historical_games} games")
print(f"Future data: {len(future_df)} rows, {future_games} games")
print()

# FIT_TRANSFORM: Learn from historical data
# - Ratings start at defaults and update based on match outcomes
# - Lags/rolling windows build up from initial games
# - Internal state (ratings, windows) is MUTATED
print("Applying fit_transform to historical data...")
historical_df = features_pipeline.fit_transform(historical_df).to_pandas()
print(f"  Generated {len(features_pipeline.features_out)} features:")
for feature in features_pipeline.features_out:
    print(f"    - {feature}")
print()

# FUTURE_TRANSFORM: Generate features for prediction (READ-ONLY)
# - Uses current ratings but does NOT update them
# - Appends current game to lag/rolling windows but doesn't persist the update
# - This is what you use in production: generate features without affecting your model's state
print("Applying future_transform to future data (read-only)...")
future_df_transformed = features_pipeline.future_transform(future_df).to_pandas()
print(f"  Future data now has {len(future_df_transformed.columns)} columns")
print()

# ====================================================================
# VERIFY FEATURES WERE CREATED
# ====================================================================

print("Sample of generated features (first player):")
sample_player = historical_df[historical_df[column_names.player_id] == historical_df.iloc[0][column_names.player_id]].head(3)
print(sample_player[[column_names.player_id, "points", "minutes"] + features_pipeline.features_out].to_string(index=False))
print()

# Show statistics to verify features make sense
print("Feature statistics (historical data):")
feature_stats = historical_df[features_pipeline.features_out].describe().T[["count", "mean", "std", "min", "max"]]
print(feature_stats.to_string())
print()

# ====================================================================
# KEY TAKEAWAYS
# ====================================================================
# 1. Always sort your data chronologically before generating features
# 2. Use fit_transform on training data to learn patterns
# 3. Use future_transform on prediction data to avoid mutating state
# 4. Features are automatically shifted to prevent data leakage
# 5. FeatureGeneratorPipeline ensures no duplicate features and preserves row count
