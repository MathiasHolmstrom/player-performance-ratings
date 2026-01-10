# Cross Validator module (Claude notes)

## Purpose
Time-series aware cross-validation for sports predictions. Splits data chronologically by match, trains on past, validates on future—respecting temporal ordering.

## Non-negotiable invariants
- Training data is always BEFORE validation data (no future leakage).
- Estimator is deep-copied for each fold (folds are independent).
- Output includes `is_validation` column (1 = validation, 0 = training).
- Data is sorted by `date_column_name` + `match_id_column_name` before splitting.

## Where things live
- `_base.py`: `CrossValidator` abstract base class with `cross_validation_score()` method.
- `cross_validator.py`: `MatchKFoldCrossValidator` - the main implementation.

## Key parameters
- `match_id_column_name`: column identifying unique matches
- `date_column_name`: column for temporal ordering
- `target_column`: what we're predicting
- `estimator`: the model/pipeline to train (must have `fit()` and `predict()`/`predict_proba()`)
- `prediction_column_name`: output column name for predictions
- `n_splits`: number of validation folds (default 3)
- `features`: explicit feature list; if None, infers from df (excludes target, date, match_id, prediction)
- `min_validation_date`: earliest date for validation data (default: median date)

## Splitting logic
1. Sort by date + match_id
2. Assign `__match_num` (cumulative match counter)
3. Find first match >= `min_validation_date`
4. Divide remaining matches into `n_splits` equal chunks
5. For each fold: train on all prior matches, validate on chunk

## Feature selection (`_get_features`)
If `features` is None, auto-infers by excluding:
- `target_column`, `date_column_name`, `match_id_column_name`, `prediction_column_name`, `__match_num`

**Pitfall**: If estimator needs context columns (e.g., `AutoPipeline` needs `context_feature_names`), you MUST pass `features` explicitly:
```python
features=pipeline.context_feature_names + pipeline.feature_names
```

## Main methods
- `generate_validation_df(df, add_training_predictions=False)`: run CV, return df with predictions + `is_validation` column.
- `cross_validation_score(validation_df, scorer)`: compute score using a `BaseScorer`.

## Common pitfalls
- Not passing `features` when using `AutoPipeline` (causes "Missing required feature columns" error).
- Empty training/validation splits (check `min_validation_date` isn't too early/late).
- Estimator not deep-copyable (must support `copy.deepcopy()`).
- Forgetting `is_validation` filter when computing final metrics and `add_training_predictions` is set to True


## When you change something, also update…
- Split logic => update tests that check fold boundaries
- Feature inference (`_get_features`) => may break pipelines relying on auto-inference
- `is_validation` column name => update `_base.py` property + any downstream scorers
