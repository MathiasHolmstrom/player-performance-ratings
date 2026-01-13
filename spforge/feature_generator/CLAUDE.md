# Feature Generator module

Lag-based feature generators for time-series sports data with state management.

## Critical Invariants
- `fit_transform()` stores historical data internally (`_df`) for use by `future_transform()`
- Output row count must equal input row count
- Features are shifted by 1 to avoid data leakage (you see past, not current match)

## Common Pitfalls
- `column_names` must be provided at instantiation or in `fit_transform()`, otherwise `_df` won't be stored and `future_transform()` will fail
- Using `add_opponent=True` without proper `column_names` (needs team_id, match_id)
- `unique_constraint` mismatch causing duplicate rows or dropped data
- Forward-fill may produce unexpected results for new entities in `future_transform()`
