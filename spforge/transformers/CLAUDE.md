# Transformers module (Claude notes)

## Purpose
Sklearn-compatible transformers that wrap estimators, perform column operations, and compute derived features. All use narwhals for pandas/polars compatibility.

## Non-negotiable invariants
- All transformers implement sklearn `TransformerMixin` interface (`fit`, `transform`).
- Must handle both pandas and polars DataFrames (via `@nw.narwhalify` decorator).
- `get_feature_names_out()` returns output column names for sklearn ColumnTransformer integration.
- Transformers MUST be cloneable via sklearn `clone()` (no required constructor args set during fit).
- Transformers convert to pandas internally for estimator fitting (`.to_pandas()`).

## Where things live
- `_predictor.py`: `EstimatorTransformer` - fits estimator, outputs predictions.
- `_operator.py`: `OperatorTransformer` - arithmetic between columns.
- `_net_over_predicted.py`: `NetOverPredictedTransformer` - residual computation.
- `_team_ratio_predictor.py`: `RatioEstimatorTransformer` - row vs granularity ratio.

## Transformer types

**`EstimatorTransformer`**: Wraps any sklearn estimator.
- `features`: columns to use for prediction (if None, uses all columns).
- `prediction_column_name`: output column name.
- Outputs only the prediction column (drops input columns).

**`OperatorTransformer`**: Column arithmetic.
- `Operation.SUBTRACT`, `MULTIPLY`, `DIVIDE`.
- Auto-generates column name if not provided (e.g., `feature1_minus_feature2`).
- Returns silently if input columns missing.

**`NetOverPredictedTransformer`**: Computes `target - predicted`.
- Requires regressor estimator.
- Outputs `net_over_predicted_col` (and optionally the raw prediction).

**`RatioEstimatorTransformer`**: Row prediction / group prediction ratio.
- `granularity`: columns defining groups (e.g., `["game_id", "team_id"]`).
- Can use existing prediction columns (`predict_row=False`, `predict_granularity=False`).
- Trains on group-aggregated data, predicts at both row and group level.

## Common pitfalls
- `EstimatorTransformer` outputs ONLY prediction column—upstream columns are dropped.
- `OperatorTransformer` silently returns unchanged df if columns missing (no error).
- `RatioEstimatorTransformer` requires `prediction_column_name` when `predict_row=False`.


## When you change something, also update…
- `get_feature_names_out()` => affects sklearn pipeline column tracking
- narwhals usage => must maintain pandas/polars compatibility
- Output column names => downstream transformers/scorers may depend on them
