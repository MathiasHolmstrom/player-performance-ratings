# Feature Generator module (Claude notes)

## Purpose
Lag-based feature generators for time-series sports data. Create rolling means, lags, and opponent-based features while maintaining state for future predictions.

## Non-negotiable invariants
- `fit_transform()` stores historical data internally (`_df`) for later use in `future_transform()`.
- `future_transform()` must be called AFTER `fit_transform()` (uses stored state).
- Output row count must equal input row count (enforced by `@transformation_validator`).
- All original columns must be preserved in output.
- Features are shifted by 1 to avoid data leakage (you see past, not current match).

## Where things live
- `_base.py`: `LagGenerator` base class with storage, grouping, opponent feature logic.
- `_lag.py`: `LagTransformer` - simple lag features (lag1, lag2, ...).
- `_rolling_window.py`: `RollingWindowTransformer` - rolling mean/sum/var over N matches.
- `_rolling_against_opponent.py`: `RollingAgainstOpponentTransformer` - how opponents perform against a granularity.
- `_rolling_mean_days.py`: `RollingMeanDaysTransformer` - rolling mean over time window (days).
- `_rolling_mean_binary.py`: `BinaryOutcomeRollingMeanTransformer` - for binary outcomes.
- `_utils.py`: decorators for validation, pandas/polars conversion, column_names handling.

## Key parameters (common across transformers)
- `features`: columns to transform
- `granularity`: grouping columns (e.g., `["player_id"]` or `["player_id", "position"]`)
- `window` / `lag_length`: how many past observations to use
- `add_opponent`: if True, also compute features for opponent team
- `update_column`: which column triggers a new observation (usually match_id)
- `column_names`: ColumnNames dataclass with standard column mappings

## State layout
Each transformer stores processed data in `_df` (native DataFrame):
- Used by `future_transform()` to compute features for upcoming matches
- Accessed via `historical_df` property
- Reset with `reset()` method

## Transform methods (behavior contract)
- `fit_transform(df, column_names)`: process historical data, store state, add feature columns.
- `future_transform(df)`: use stored state to add features; forward-fills missing values for new entities.

## Decorator stack (applied in order, bottom-up)
```python
@nw.narwhalify                          # narwhals compatibility
@historical_lag_transformations_wrapper  # setup, pandas↔polars, cleanup
@required_lag_column_names              # validate/set column_names
@transformation_validator               # assert row count preserved
def fit_transform(self, df, ...):
```

## Common pitfalls
- `column_names` must either be injected when instantiating the object or when passed to `fit_transform()`. 
  - Otherwise it won't store the _df, which is needed before calling `future_transform()`. 
  - If `FeatureGeneratorPipeline`, it will take care of passing `column_names` to `fit_transform()`. 
- Using `add_opponent=True` without proper `column_names` (needs team_id, match_id).
- `unique_constraint` mismatch causing duplicate rows or dropped data.

## When you change something, also update…
- Base class storage logic (`_store_df`, `_concat_with_stored`) => affects all transformers
- Decorator logic in `_utils.py` => affects all fit_transform/future_transform calls
- Feature naming (`prefix`, `features_out`) => update tests + any pipelines using feature names
- Granularity/grouping logic => `_maybe_group`, `_group_to_granularity_level` + related tests
