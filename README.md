# spforge

**spforge** is a sports prediction framework for building feature-rich, stateful, and
sklearn-compatible modeling pipelines.

It is designed for:
- player- and team-level ratings
- rolling and lagged feature generation
- match-aware cross-validation
- probabilistic and point-estimate models
- pandas **and** polars DataFrames (via narwhals)

Typical use cases include:
- predicting game winners
- predicting player or team points
- generating probabilities using either machine learning models or distributions
- feature engineering and cross-validation

---

## Installation

```bash
pip install spforge
```

## Core assumptions

spforge assumes your data is structured as:

- **One row per entity per match**
  - e.g. `(game_id, player_id)` or `(game_id, team_id)`
- Higher-level predictions (team/game) are handled via aggregation or grouping.

## Key concepts

Before diving into examples, here are fundamental concepts that guide how spforge works:

- **Temporal ordering prevents future leakage**: Data must be sorted chronologically (by date, then match, then team/player). This ensures models never "see the future" when making predictions.

- **Elo-style ratings**: Player and team ratings evolve over time based on match performance. Think of it like a chess rating - win against strong opponents and your rating increases more. Ratings are calculated BEFORE each match to avoid leakage.

- **State management lifecycle**:
  - `fit_transform(df)`: Learn patterns from historical data (ratings update, windows build up)
  - `transform(df)`: Apply to more historical data (continues updating state)
  - `future_transform(df)`: Generate features for prediction WITHOUT updating internal state (read-only)

- **Granularity-based aggregation**: Player-level data (e.g., individual stats) can be automatically aggregated to team-level for game winner predictions.

- **pandas and polars support**: All components work identically with both DataFrame types via the narwhals library.

## Example

This example demonstrates predicting NBA game winners using player-level ratings.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from examples import get_sub_sample_nba_data
from spforge.autopipeline import AutoPipeline
from spforge.data_structures import ColumnNames
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures

df = get_sub_sample_nba_data(as_pandas=True, as_polars=False)

# Step 1: Define column mappings for your dataset
column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_name",
)

# Step 2: CRITICAL - Sort data chronologically to prevent future leakage
# This ensures ratings and features only use past information
df = df.sort_values(
    by=[
        column_names.start_date,  # First by date
        column_names.match_id,    # Then by match
        column_names.team_id,     # Then by team
        column_names.player_id,   # Finally by player
    ]
)

# Step 3: Filter to valid games (exactly 2 teams)
df = (
    df.assign(
        team_count=df.groupby(column_names.match_id)[column_names.team_id].transform("nunique")
    )
    .loc[lambda x: x.team_count == 2]
    .drop(columns=["team_count"])
)

# Step 4: Split into historical (training) and future (prediction) data
# In production, "future" would be upcoming games without outcomes
most_recent_10_games = df[column_names.match_id].unique()[-10:]
historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)]
future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(columns=["won"])

# Step 5: Generate player ratings based on win/loss history
# Each player gets a rating that updates after each game
# Unlike traditional team Elo, ratings follow individual players
rating_generator = PlayerRatingGenerator(
    performance_column="won",  # Update ratings based on wins/losses
    rating_change_multiplier=30,  # How quickly ratings adjust (higher = more volatile)
    column_names=column_names,
    non_predictor_features_out=[RatingKnownFeatures.PLAYER_RATING],
)
# fit_transform learns ratings from historical games
historical_df = rating_generator.fit_transform(historical_df)

# Step 6: Create prediction pipeline
# AutoPipeline automatically handles preprocessing (encoding, scaling)
# granularity aggregates player-level data to team-level before fitting
pipeline = AutoPipeline(
    estimator=LogisticRegression(),
    granularity=["game_id", "team_id"],  # Aggregate players → teams
    estimator_features=rating_generator.features_out + ["location"],  # Rating + home/away
)

# Train on historical data
pipeline.fit(X=historical_df, y=historical_df["won"])

# Step 7: Make predictions on future games
# future_transform generates features WITHOUT updating rating state
# This is crucial: we don't want to update ratings until games actually happen
future_df = rating_generator.future_transform(future_df)
future_predictions = pipeline.predict_proba(future_df)[:, 1]  # Probability of winning
future_df["game_winner_probability"] = future_predictions

# Aggregate player-level predictions to team-level for final output
team_grouped_predictions = future_df.groupby(column_names.match_id).first()[
    [
        column_names.start_date,
        column_names.team_id,
        "team_id_opponent",
        "game_winner_probability",
    ]
]

print(team_grouped_predictions)
```
Output:
```
            start_date     team_id  team_id_opponent  game_winner_probability
game_id                                                                      
0022200767  2023-01-31  1610612749        1610612766                 0.731718
0022200768  2023-01-31  1610612740        1610612743                 0.242622
0022200770  2023-02-01  1610612753        1610612755                 0.278237
0022200771  2023-02-01  1610612757        1610612763                 0.340883
0022200772  2023-02-01  1610612738        1610612751                 0.629010
0022200773  2023-02-01  1610612745        1610612760                 0.401803
0022200774  2023-02-01  1610612744        1610612750                 0.430164
0022200775  2023-02-01  1610612758        1610612759                 0.587513
0022200776  2023-02-01  1610612761        1610612762                 0.376864
0022200777  2023-02-01  1610612737        1610612756                 0.371888
```
## AutoPipeline

`AutoPipeline` is a sklearn-compatible wrapper that handles the full modeling pipeline,
from preprocessing to final estimation.

- Builds all required preprocessing steps automatically based on the estimator:
  - One-hot encoding and imputation for linear models (e.g. `LogisticRegression`)
  - Native categorical handling for LightGBM
  - Ordinal encoding where appropriate
- Supports **predictor transformers**, allowing upstream models to generate features
  that are consumed by the final estimator.
- Supports optional **granularity-based aggregation**, enabling row-level data
  (e.g. player-game) to be grouped before fitting (e.g. game-team level).
- Provides additional functionality such as:
  - training-time row filtering
  - target clipping and validation handling
  - consistent feature tracking for sklearn integration

## Feature Engineering

spforge provides stateful feature generators that create rich features from historical match data while maintaining temporal ordering to prevent data leakage.

### Feature types available

- **Ratings**: Elo-style player/team ratings that evolve based on performance (separate offense/defense ratings)
  - Can combine multiple stats into a composite performance metric using `performance_weights` (e.g., 60% kills + 40% assists)
  - Auto-normalizes raw stats to 0-1 range with `auto_scale_performance=True`
- **Lags**: Previous match statistics, automatically shifted to prevent leakage
- **Rolling windows**: Averages/sums over the last N matches
- **FeatureGeneratorPipeline**: Chain multiple generators together sequentially

### Example: Building a feature pipeline

```python
from spforge import FeatureGeneratorPipeline
from spforge.feature_generator import LagTransformer, RollingWindowTransformer
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures
from spforge.performance_transformers import ColumnWeight

# Create individual feature generators
player_rating_generator = PlayerRatingGenerator(
    performance_column="points",
    auto_scale_performance=True,  # Normalizes points to 0-1 range
    column_names=column_names,
    features_out=[RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED],
)

# Alternative: Combine multiple stats into a composite performance metric
# player_rating_generator = PlayerRatingGenerator(
#     performance_column="weighted_performance",  # Name for the composite metric
#     performance_weights=[
#         ColumnWeight(name="kills", weight=0.6),
#         ColumnWeight(name="assists", weight=0.4),
#     ],
#     column_names=column_names,
#     features_out=[RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED],
# )

lag_transformer = LagTransformer(
    features=["points"],
    lag_length=3,  # Last 3 games
    granularity=["player_id"],
)

rolling_transformer = RollingWindowTransformer(
    features=["points"],
    window=10,  # Last 10 games average
    granularity=["player_id"],
)

# Chain them together
features_pipeline = FeatureGeneratorPipeline(
    column_names=column_names,
    feature_generators=[
        player_rating_generator,
        lag_transformer,
        rolling_transformer,
    ],
)

# Learn from historical data
historical_df = features_pipeline.fit_transform(historical_df)

# For production predictions (doesn't update internal state)
future_df = features_pipeline.future_transform(future_df)
```

**Key points:**
- `fit_transform`: Learn ratings/patterns from historical data (updates internal state)
- `transform`: Apply to more historical data (continues updating state)
- `future_transform`: Generate features for prediction (read-only, no state updates)
- Features are automatically shifted by 1 match to prevent data leakage

See [examples/nba/feature_engineering_example.py](examples/nba/feature_engineering_example.py) for a complete example with detailed explanations.

## Cross Validation and Scorer metrics

Regular k-fold cross-validation doesn't work for time-series sports data because it can create "future leakage" - using future games to predict past games. `MatchKFoldCrossValidator` ensures training data is always BEFORE validation data, respecting temporal ordering.

### Why this matters

Sports data has strong time dependencies: teams improve, players get injured, strategies evolve. Standard CV would overestimate model performance by allowing the model to "see the future."

### Example: Time-series cross-validation

```python
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.scorer import SklearnScorer, Filter, Operator
from sklearn.metrics import mean_absolute_error

# Set up temporal cross-validation
cross_validator = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    estimator=pipeline,  # Your AutoPipeline
    prediction_column_name="points_pred",
    target_column="points",
    n_splits=3,  # Number of temporal folds
    # Must include both estimator features and context features
    features=pipeline.required_features,
)

# Generate validation predictions
# add_training_predictions=True also returns predictions on training data
validation_df = cross_validator.generate_validation_df(df=df, add_training_predictions=True)

# Score only validation rows, filtering to players who actually played
scorer = SklearnScorer(
    pred_column="points_pred",
    target="points",
    scorer_function=mean_absolute_error,
    validation_column="is_validation",  # Only score where is_validation == 1
    filters=[
        Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)
    ],
)

mae = scorer.score(validation_df)
print(f"Validation MAE: {mae:.2f}")
```

**Key points:**
- `add_training_predictions=True` returns both training and validation predictions
  - `is_validation=1` marks validation rows, `is_validation=0` marks training rows
  - Use `validation_column` in scorer to score only validation rows
- Training data always comes BEFORE validation data chronologically
- Must pass all required features (use `pipeline.required_features`)
- Scorers can filter rows (e.g., only score players who played minutes > 0)

See [examples/nba/cross_validation_example.py](examples/nba/cross_validation_example.py) for a complete example.

## Distributions (Advanced)

Instead of predicting a single point estimate, you can predict full probability distributions. For example, instead of "player will score 15 points", predict P(0 points), P(1 point), ..., P(40 points).

### When to use distributions

- Modeling count data (points, goals, kills, assists)
- When you need uncertainty estimates or confidence intervals
- For expected value calculations in betting or DFS
- When the outcome has inherent randomness

### What NegativeBinomialEstimator does during fit

During training, `NegativeBinomialEstimator`:

1. Takes the point estimates (from `point_estimate_pred_column`) and actual target values
2. Optimizes a dispersion parameter `r` using maximum likelihood estimation on the negative binomial distribution
3. If `r_specific_granularity` is set (e.g., per player), calculates entity-specific `r` values by:
   - Computing rolling means and variances of point estimates over recent matches
   - Binning entities by quantiles of mean and variance
   - Fitting separate `r` values for each bin to capture different uncertainty patterns

During prediction, it uses the learned `r` parameter(s) and the point estimates to generate a full probability distribution over all possible values (0 to max_value).

### Example: Comparing classifiers vs distribution estimators

A key advantage is comparing different approaches for generating probability distributions. Both LGBMClassifier and LGBMRegressor+NegativeBinomial output probabilities in the same format, making them directly comparable.

```python
from spforge.distributions import NegativeBinomialEstimator
from spforge.transformers import EstimatorTransformer
from lightgbm import LGBMClassifier, LGBMRegressor

# Approach 1: LGBMClassifier (direct probability prediction)
pipeline_classifier = AutoPipeline(
    estimator=LGBMClassifier(verbose=-100, random_state=42),
    estimator_features=features_pipeline.features_out,
)

# Approach 2: LGBMRegressor + NegativeBinomialEstimator
distribution_estimator = NegativeBinomialEstimator(
    max_value=40,  # Predict 0-40 points
    point_estimate_pred_column="points_estimate",  # Uses regressor output
    r_specific_granularity=["player_id"],  # Player-specific dispersion
    predicted_r_weight=1,
    column_names=column_names,
)

pipeline_negbin = AutoPipeline(
    estimator=distribution_estimator,
    estimator_features=features_pipeline.features_out,
    predictor_transformers=[
        EstimatorTransformer(
            prediction_column_name="points_estimate",
            estimator=LGBMRegressor(verbose=-100, random_state=42),
            features=features_pipeline.features_out,
        )
    ],
)

# Compare using cross-validation (see examples for full setup)
# Results on NBA player points prediction:
# LGBMClassifier Ordinal Loss:              1.0372
# LGBMRegressor + NegativeBinomial Ordinal Loss: 0.3786
# LGBMRegressor + NegativeBinomial Point Est MAE: 4.5305
```

**Key points:**
- Both approaches output probability distributions over the same range
- `NegativeBinomialEstimator` performs significantly better (lower ordinal loss)
- Distribution approach provides both probability distributions and point estimates
- Can model player-specific variance with `r_specific_granularity`

See [examples/nba/cross_validation_example.py](examples/nba/cross_validation_example.py) for a complete runnable example with both approaches.

## Predictions as features for downstream models (Advanced)

A common pattern in sports analytics is using output from one model as input to another. For example, team strength (game winner probability) often influences individual player performance.

### Why this matters

Hierarchical modeling captures dependencies: team context → player performance, game flow → outcome probabilities. By chaining models, each stage can specialize and the final model combines their insights.

### Example: Two-stage modeling with predictor_transformers

```python
from spforge.transformers import EstimatorTransformer
from lightgbm import LGBMRegressor

# Stage 1: Create a raw point estimate
points_estimate_transformer = EstimatorTransformer(
    prediction_column_name="points_estimate_raw",
    estimator=LGBMRegressor(verbose=-100, n_estimators=30),
)

# Stage 2: Refine estimate using Stage 1 output
player_points_pipeline = AutoPipeline(
    estimator=LGBMRegressor(verbose=-100, n_estimators=50),
    estimator_features=features_pipeline.features_out,  # Original features
    # predictor_transformers execute first, adding their predictions
    predictor_transformers=[points_estimate_transformer],
)

# During fit:
#   1. Stage 1 fits and generates "points_estimate_raw" column
#   2. Stage 2 fits using original features + points_estimate_raw
player_points_pipeline.fit(X=train_df, y=train_df["points"])

# During predict:
#   1. Stage 1 generates "points_estimate_raw"
#   2. Stage 2 uses it to make final prediction
predictions = player_points_pipeline.predict(test_df)
```

**Key points:**
- `predictor_transformers` chains estimators: output of one becomes input to next
- All transformers share the same target (y) during fit
- Transformers execute during both `fit()` and `predict()`
- Common use cases:
  - Generate point estimates for distribution models
  - Multi-stage refinement of predictions
  - Combining different model types (linear → tree-based)

See [examples/nba/predictor_transformers_example.py](examples/nba/predictor_transformers_example.py) for a complete example. Also demonstrated in [examples/nba/cross_validation_example.py](examples/nba/cross_validation_example.py).

## More Examples

For complete, runnable examples with detailed explanations:

- **[examples/nba/feature_engineering_example.py](examples/nba/feature_engineering_example.py)** - Feature generation lifecycle (ratings, lags, rolling windows)
- **[examples/nba/cross_validation_example.py](examples/nba/cross_validation_example.py)** - Time-series CV, distributions, and scoring
- **[examples/nba/predictor_transformers_example.py](examples/nba/predictor_transformers_example.py)** - Multi-stage hierarchical modeling
- **[examples/nba/game_winner_example.py](examples/nba/game_winner_example.py)** - Basic workflow for game winner prediction
