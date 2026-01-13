# Scorer module

Evaluation of model predictions with filtering, aggregation, and granularity support.

## Critical Invariants
- All scorers extend `BaseScorer` and implement `score(df) -> float | dict`
- Filters are applied BEFORE scoring (including automatic `validation_column == 1` filter)
- `aggregation_level` groups rows before scoring (e.g., player → team level)
- `granularity` returns separate scores per group as `dict[tuple, float]`

## Common Pitfalls
- Forgetting `validation_column` causes scoring on training data too (if cross validation generated with `add_training_predictions=True`)
- `OrdinalLossScorer` requires consecutive integer classes starting from min
- Distribution scorers expect `pred_column` to contain list/array per row
- Empty DataFrame after filters returns 0.0 or empty dict (not error)
- `aggregation_level` uses sum for numeric columns by default (not mean)
