# Estimator module

Sklearn-compatible estimators with sports-specific enhancements (granularity, date-weighting, distributions).

## Critical Invariants
- All require DataFrame input (NOT numpy arrays)
- All follow sklearn interface: `fit(X, y)`, `predict(X)`, `predict_proba(X)`
- Distribution estimators return shape `(n_samples, n_classes)`

## Common Pitfalls
- Passing numpy array instead of DataFrame raises TypeError
- Distribution estimators: forgetting `point_estimate_pred_column` must exist in X
- `OrdinalClassifier` requires 3+ classes (not binary)
- `GranularityEstimator` fails if granularity value in predict wasn't seen in fit
- Distribution estimators need `classes_` to match `range(min_value, max_value+1)`
