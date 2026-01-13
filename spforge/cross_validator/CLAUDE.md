# Cross Validator module

Time-series aware cross-validation for sports predictions with temporal ordering.

## Critical Invariants
- Training data is always BEFORE validation data (no future leakage)
- Estimator is deep-copied for each fold (folds are independent)
- Output includes `is_validation` column (1 = validation, 0 = training)

## Common Pitfalls
- Not passing `features` when using `AutoPipeline` causes "Missing required feature columns" error. Must explicitly pass: `features=pipeline.context_feature_names + pipeline.feature_names`
- Empty training/validation splits (check `min_validation_date` isn't too early/late)
- Estimator not deep-copyable (must support `copy.deepcopy()`)
- Forgetting `is_validation` filter when computing final metrics (when `add_training_predictions=True`)
