# Scorer module (Claude notes)

## Purpose
Evaluate model predictions with filtering, aggregation, and granularity support. Wraps sklearn metrics and adds sports-specific scorers for probability distributions.

## Non-negotiable invariants
- All scorers extend `BaseScorer` and implement `score(df) -> float | dict`.
- Filters are applied BEFORE scoring (including automatic `validation_column == 1` filter).
- `aggregation_level` groups rows before scoring (e.g., player → team level).
- `granularity` returns separate scores per group as `dict[tuple, float]`.
- Scorers must handle both pandas and polars DataFrames.

## Where things live
- `_score.py`: All scorer implementations + Filter/Operator utilities.

## Core classes

**`Filter`** + **`Operator`**: Dataclass for row filtering.
```python
Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)
```

**`BaseScorer`**: Abstract base with common params:
- `target`: column with actual values
- `pred_column`: column with predictions
- `validation_column`: if set, auto-filters to rows where value == 1
- `filters`: list of Filter objects
- `aggregation_level`: group by these columns before scoring
- `granularity`: compute separate scores per unique combination

## Scorer types

**`SklearnScorer`**: Wraps any sklearn metric function.
```python
SklearnScorer(scorer_function=mean_absolute_error, pred_column="pred", target="actual")
```
- Handles both scalar predictions and probability arrays.
- Pass extra params via `params` dict.

**`MeanBiasScorer`**: Mean(prediction - target).
- Positive = overpredicting, negative = underpredicting.

**`OrdinalLossScorer`**: Weighted log-loss for ordinal distributions.
- Requires `classes` list (consecutive integers).
- `pred_column` must contain probability arrays matching len(classes).
- Computes cumulative "under threshold" log-loss weighted by class frequency.

**`PWMSE`**: Probability-weighted mean squared error.
- For distribution predictions: weights squared errors by predicted probabilities.
- Requires `labels` list matching distribution indices.

**`ProbabilisticMeanBias`**: Calibration scorer for distributions.
- Checks if predicted CDF probabilities match actual outcome frequencies.

**`ThresholdEventScorer`**: Scores over/under threshold events.
- Converts distribution + threshold into binary event probability.
- Delegates to `binary_scorer` (default: log_loss).
- `comparator`: which operator defines the event (>=, >, <, <=, ==, !=).

## Key patterns

### Validation column auto-filter
```python
# Automatically adds: Filter(validation_column, 1, Operator.EQUALS)
scorer = SklearnScorer(..., validation_column="is_validation")
```

### Aggregation before scoring
```python
# Score at team level instead of player level
scorer = SklearnScorer(..., aggregation_level=["game_id", "team_id"])
```

### Granularity for per-group scores
```python
scorer = SklearnScorer(..., granularity=["position"])
scores = scorer.score(df)  # Returns {"PG": 0.5, "SG": 0.6, ...}
```

## Common pitfalls
- Forgetting `validation_column` causes scoring on training data too (if cross validation was generated with `add_training_predictions = True` )
- `OrdinalLossScorer` requires consecutive integer classes starting from min.
- Distribution scorers expect `pred_column` to contain list/array per row.
- Empty DataFrame after filters returns 0.0 or empty dict (not error).
- `aggregation_level` uses sum for numeric columns by default.

## When you change something, also update…
- `BaseScorer` interface => all scorer subclasses must comply
- Filter logic (`apply_filters`) => affects all scorers
- Aggregation logic (`_apply_aggregation_level`) => affects all scorers with aggregation
- Distribution column handling => `OrdinalLossScorer`, `PWMSE`, `ThresholdEventScorer`
