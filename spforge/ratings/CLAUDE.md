# Ratings module (Claude notes)

## Purpose
Elo-style ratings for teams and players with separate offense/defense states.
Used to create pre-match rating features and (for historical data) update ratings after each match.

## Non-negotiable invariants
- Updates must be chronological (by match day / start_date).
- `future_transform(...)` MUST NOT update internal rating state.
- Ratings are maintained separately for offense and defense (do not merge or “net” them).
- `performance_column` is expected to be 0–1 normalized (or `auto_scale_performance=True` must be used).
- Internal rating dictionaries are mutable state and should be treated as the source of truth.

## Where things live
- `_base.py`: shared update logic (confidence decay, multiplier application, update loop scaffolding).
- `_team_rating.py`: TeamRatingGenerator orchestration and team-specific state handling.
- `_player_rating.py`: PlayerRatingGenerator orchestration (participation weights, team context).
- `*_performance_predictor.py`: expected-performance calculation strategies.
- `enums.py`: feature names (known vs unknown).
- `utils.py`: helpers for adding rating columns to frames.
- `start_rating_generator.py` / `team_start_rating_generator.py`: initialization for new entities.

## State layout
Each generator maintains two dicts of RatingState:
- offense ratings: `_team_off_ratings` / `_player_off_ratings` (dict[str, RatingState])
- defense ratings: `_team_def_ratings` / `_player_def_ratings` (dict[str, RatingState])

RatingState fields (see data_structures.py) must remain stable for external consumers:
- `id`, `rating_value`, `confidence_sum`, `games_played`, `last_match_day_number`
(+ optional: `most_recent_group_id`, etc. depending on current dataclass)

## Update rule (conceptual)
- predicted_performance comes from `performance_predictor` ("difference" / "mean" / "ignore_opponent").
- rating_change = (actual_performance - predicted_performance) * applied_multiplier
- applied_multiplier decreases as confidence_sum grows (new entities update faster).

## Transform methods (behavior contract)
- `fit_transform(df, ...)`: chronological processing, updates internal ratings after each match; adds features.
- `transform(df)`: same behavior as fit_transform but uses stored column_names; still updates ratings.
- `future_transform(df)`: adds pre-match features only; MUST NOT update ratings.

## Public accessors
Generators expose ratings via properties:
- TeamRatingGenerator.team_ratings -> dict[str, TeamRatingsResult]
- PlayerRatingGenerator.player_ratings -> dict[str, PlayerRatingsResult]
These should reflect offense/defense rating values + confidence/games.

## Common pitfalls
- Accidentally mutating ratings inside feature-only paths (especially future_transform).
- Using unsorted input (silently produces nonsense; ensure sorting occurs once).
- Mixing “actual” and “predicted” performance scales (ensure both are in 0–1 domain).
- Forgetting that defense performance is usually inverted (depends on your performance definition).

## When you change something, also update…
- Update formula / confidence logic => `_base.py` + related predictors + unit tests in `tests/ratings/...`
- Feature enums => `enums.py` + utils that add columns + any pipelines consuming feature names
- RatingState fields => data_structures + any serialization / downstream usage + tests
