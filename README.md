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

## Core assumptions

spforge assumes your data is structured as:

- **One row per entity per match**
  - e.g. `(game_id, player_id)` or `(game_id, team_id)`
- Higher-level predictions (team/game) are handled via aggregation or grouping.
```

## Example
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from spforge.autopipeline import AutoPipeline
from spforge.data_structures import ColumnNames
from spforge.ratings import RatingKnownFeatures
from spforge.ratings._player_rating import PlayerRatingGenerator

df = pd.read_parquet("data/game_player_subsample.parquet")

# Defines the column names as they appear in the dataframe
column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_name",
)
# Sorts the dataframe. The dataframe must always be sorted as below
df = df.sort_values(
    by=[
        column_names.start_date,
        column_names.match_id,
        column_names.team_id,
        column_names.player_id,
    ]
)

# Drops games with less or more than 2 teams
df = (
    df.assign(
        team_count=df.groupby(column_names.match_id)[column_names.team_id].transform("nunique")
    )
    .loc[lambda x: x.team_count == 2]
    .drop(columns=["team_count"])
)

# Pretends the last 10 games are future games. The most will be trained on everything before that.
most_recent_10_games = df[column_names.match_id].unique()[-10:]
historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)]
future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(columns=["won"])

# Defining a simple PlayerRatingGenerator-generator. It will use the "won" column to update the ratings.
# In contrast to a typical Elo, ratings will follow players.


rating_generator = PlayerRatingGenerator(
    performance_column="won",
    rating_change_multiplier=30,
    column_names=column_names,
    non_predictor_features_out=[RatingKnownFeatures.PLAYER_RATING],
)
historical_df = rating_generator.fit_transform(historical_df)

# LogisticRegression inside AutoPipeline with `granularity` set.
# This results in:
# - Rows being aggregated to (game_id, team_id) before fitting.
# - Categorical features (e.g. "location") being one-hot encoded automatically
#   so LogisticRegression can consume them.
pipeline = AutoPipeline(
    estimator=LogisticRegression(),
    granularity=["game_id", "team_id"],
    feature_names=rating_generator.features_out + ["location"],
)

pipeline.fit(X=historical_df, y=historical_df["won"])

# Future predictions on future results
future_df = rating_generator.future_transform(future_df)
future_predictions = pipeline.predict_proba(future_df)[:, 1]
future_df["game_winner_probability"] = future_predictions
# Grouping predictions from game-player level to game-level.
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

## Cross Validation and Scorer metrics

## Distributions (Advanced)

## Predictions as features for downstream models (Advanced)

