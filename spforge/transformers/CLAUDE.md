# Transformers module

Sklearn-compatible transformers that wrap estimators and perform column operations.

## Critical Invariants
- All implement sklearn `TransformerMixin` interface (`fit`, `transform`)
- Must handle both pandas and polars DataFrames (via `@nw.narwhalify`)
- `get_feature_names_out()` returns output column names for sklearn integration
- Transformers must be cloneable via sklearn `clone()`

## Common Pitfalls
- `EstimatorTransformer` outputs ONLY prediction column—upstream columns are dropped
- `OperatorTransformer` silently returns unchanged df if columns missing (no error)
- `RatioEstimatorTransformer` requires `prediction_column_name` when `predict_row=False`
