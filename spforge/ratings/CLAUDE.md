# Ratings Module

## Overview

This module implements Elo-style rating systems for teams and players in sports analytics. Ratings track offensive and defensive performance separately, updating after each match based on actual vs predicted performance.

## Key Classes

### `TeamRatingGenerator` (`_team_rating.py`)
Generates team-level ratings. Each team has separate offense and defense ratings.

### `PlayerRatingGenerator` (`_player_rating.py`)
Generates player-level ratings with offense/defense split. More complex than team ratings due to participation weights and team context.

### `RatingGenerator` (`_base.py`)
Abstract base class containing shared logic for confidence decay, rating updates, and performance scaling.

## Rating System Architecture

### Dual Offense/Defense Ratings
Both generators maintain **two separate rating dictionaries**:
- `_team_off_ratings` / `_player_off_ratings`: Offensive ratings (dict[str, RatingState])
- `_team_def_ratings` / `_player_def_ratings`: Defensive ratings (dict[str, RatingState])

### RatingState (from `data_structures.py`)
```python
@dataclass
class RatingState:
    id: str
    rating_value: float          # The actual rating (starts ~1000)
    confidence_sum: float = 0.0  # Accumulated confidence from matches
    games_played: float = 0.0
    last_match_day_number: int | None = None
```

### Rating Update Formula
```
rating_change = (actual_performance - predicted_performance) * applied_multiplier
```

Where:
- `actual_performance`: Normalized 0-1 value from `performance_column`
- `predicted_performance`: Based on rating difference (see `performance_predictor`)
- `applied_multiplier`: Decreases as `confidence_sum` grows (new players change faster)

## Accessing Ratings

### Properties (added for external access)

**TeamRatingGenerator.team_ratings** -> `dict[str, TeamRatingsResult]`
```python
generator.team_ratings["team_id"]  # Returns TeamRatingsResult with offense/defense ratings
```

**PlayerRatingGenerator.player_ratings** -> `dict[str, PlayerRatingsResult]`
```python
generator.player_ratings["player_id"]  # Returns PlayerRatingsResult
```

### Result Dataclasses
```python
@dataclass
class TeamRatingsResult:
    id: str
    offense_rating: float
    defense_rating: float
    offense_games_played: float
    defense_games_played: float
    offense_confidence_sum: float
    defense_confidence_sum: float

@dataclass
class PlayerRatingsResult:
    # Same as TeamRatingsResult plus:
    most_recent_team_id: str | None
```

## Feature Outputs

### RatingKnownFeatures (enums.py)
Features available at prediction time (before match result is known):
- `TEAM_RATING_PROJECTED`, `OPPONENT_RATING_PROJECTED`
- `TEAM_OFF_RATING_PROJECTED`, `TEAM_DEF_RATING_PROJECTED`
- `PLAYER_RATING`, `PLAYER_OFF_RATING`, `PLAYER_DEF_RATING`
- `TEAM_RATING_DIFFERENCE_PROJECTED`, `PLAYER_RATING_DIFFERENCE_PROJECTED`

### RatingUnknownFeatures (enums.py)
Features only available after match (for analysis):
- `PERFORMANCE`, `PLAYER_RATING_CHANGE`
- `TEAM_RATING_DIFFERENCE`, `PLAYER_RATING_DIFFERENCE`

## Performance Predictors

The `performance_predictor` parameter controls how expected performance is calculated:

- `"difference"` (default): Uses rating difference between player/team and opponent
- `"mean"`: Uses average of both ratings
- `"ignore_opponent"`: Only considers own rating

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `performance_column` | required | Column containing 0-1 normalized performance |
| `auto_scale_performance` | False | Auto-normalize performance to 0-1 range |
| `rating_change_multiplier_offense` | 50 | Base multiplier for offense rating changes |
| `rating_change_multiplier_defense` | 50 | Base multiplier for defense rating changes |
| `confidence_weight` | 0.9 | How much confidence affects multiplier |
| `confidence_max_sum` | 150 | Maximum confidence accumulation |

## File Structure

```
ratings/
├── __init__.py              # Public exports
├── _base.py                 # RatingGenerator base class
├── _team_rating.py          # TeamRatingGenerator
├── _player_rating.py        # PlayerRatingGenerator
├── enums.py                 # RatingKnownFeatures, RatingUnknownFeatures
├── start_rating_generator.py      # Initial rating for new players
├── team_start_rating_generator.py # Initial rating for new teams
├── player_performance_predictor.py # Performance prediction logic
├── team_performance_predictor.py   # Team performance prediction
├── league_identifier.py     # League-based rating adjustments
└── utils.py                 # Helper functions for adding rating columns
```

## Transform Methods

The rating generators have three transform methods with different purposes:

### `fit_transform(df, column_names)`
**Use for: Historical data with known outcomes**

- Processes data chronologically
- **Updates internal ratings** after each match based on actual performance
- Adds rating features to the dataframe
- Must be called first to initialize the rating system

```python
# Process historical matches - ratings are updated after each match
df = rating_gen.fit_transform(historical_df, column_names=column_names)
```

### `transform(df)`
**Use for: Additional historical data after fit_transform**

- Same as `fit_transform` but uses existing `column_names`
- **Updates internal ratings** based on match outcomes
- Use when processing more historical data in batches

```python
# Process more historical data
df2 = rating_gen.transform(more_historical_df)
```

### `future_transform(df)`
**Use for: Future fixtures (predictions)**

- Uses current ratings to compute pre-match features
- **Does NOT update ratings** (no match outcomes yet)
- Use for generating predictions on upcoming matches

```python
# Generate predictions for upcoming matches - ratings stay unchanged
upcoming_with_features = rating_gen.future_transform(upcoming_fixtures_df)
```

### Key Difference Summary

| Method | Updates Ratings? | Use Case |
|--------|-----------------|----------|
| `fit_transform` | Yes | Initial historical data processing |
| `transform` | Yes | Additional historical data |
| `future_transform` | No | Upcoming match predictions |

## Common Usage Pattern

```python
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures

# Create generator
rating_gen = PlayerRatingGenerator(
    performance_column="points_per_minute",
    auto_scale_performance=True,
    features_out=[RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED],
)

# Transform data (updates internal ratings)
df = rating_gen.fit_transform(df, column_names=column_names)

# Access final ratings
for player_id, result in rating_gen.player_ratings.items():
    print(f"{player_id}: OFF={result.offense_rating:.1f}, DEF={result.defense_rating:.1f}")
```

## Important Implementation Details

1. **Ratings are updated chronologically** - Data must be sorted by date before `fit_transform`

2. **Performance must be 0-1 normalized** - Use `auto_scale_performance=True` or pre-normalize

3. **Confidence decays over time** - Players who haven't played recently have lower confidence, allowing faster rating changes when they return

4. **Team context matters for players** - Player ratings consider team strength via `participation_weight`

5. **Internal state is mutable** - Calling `fit_transform` updates the internal `_*_off_ratings` and `_*_def_ratings` dicts
