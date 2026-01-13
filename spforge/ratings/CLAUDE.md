# Ratings module

Elo-style ratings for teams/players with separate offense/defense states.

## Critical Invariants
- `future_transform()` MUST NOT update internal rating state (features only)
- Updates must be chronological (by match day/start_date)
- `performance_column` expected 0-1 normalized (or use `auto_scale_performance=True`)

## Common Pitfalls
- Accidentally mutating ratings in feature-only paths (especially future_transform)
- Using unsorted input produces nonsense (ensure sorting happens once)
- Mixing "actual" and "predicted" performance scales
- Defense performance is usually inverted (depends on your performance definition)
