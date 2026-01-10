# Estimator module (Claude notes)

## Purpose
Sklearn-compatible estimators with sports-specific enhancements: granularity-aware fitting, date-based weighting, distribution predictions, and ordinal classification.

## Non-negotiable invariants
- All estimators require DataFrame input (pandas/polars), NOT numpy arrays.
- All follow sklearn interface: `fit(X, y)`, `predict(X)`, `predict_proba(X)`.
- Estimators must expose `classes_` after fitting (for compatibility with scorers/pipelines).
- Distribution estimators return probability arrays of shape `(n_samples, n_classes)`.

## Where things live
- `sklearn_estimator.py`: Wrappers that enhance sklearn estimators.
- `_distribution.py`: Probability distribution estimators for count/continuous data.

## Estimator types

### Sklearn wrappers (`sklearn_estimator.py`)

**`GroupByEstimator`**: Aggregates rows by `granularity` columns before fitting.
- Use when multiple rows map to same prediction (e.g., player rows → team prediction).
- Internally uses `GroupByReducer` to aggregate X and y.
- Can be both a regressor and a classifier

**`SkLearnEnhancerEstimator`**: Adds date-based sample weighting.
- `date_column`: column for temporal weighting
- `day_weight_epsilon`: controls decay (recent data weighted higher)
- Drops date column before fitting underlying estimator.
- Can be both a regressor and a classifier

**`OrdinalClassifier`**: For ordinal targets with 3+ ordered classes.
- Trains binary classifiers for each threshold (class > k).
- Combines binary probabilities into ordinal distribution.

**`GranularityEstimator`**: Fits separate model per unique value in `granularity_column_name`.
- Use for position-specific or player-specific models.
- Stores dict of fitted estimators: `_granularity_estimators[value]`.
- Can be both a regressor and a classifier

**`ConditionalEstimator`**: Gate-based two-stage prediction.
- `gate_estimator`: predicts which branch (0 or 1)
- `outcome_0_estimator` / `outcome_1_estimator`: branch-specific models
- Combines probabilities weighted by gate predictions.

### Distribution estimators (`_distribution.py`)

** Wrappers around distributions using the sklearn interface.

**`NegativeBinomialEstimator`**: For overdispersed count data (e.g., points scored).
- Requires `point_estimate_pred_column` (mean prediction from upstream model).
- Learns dispersion parameter `r` from residuals.
- `r_specific_granularity`: learns granularity-specific `r` values (e.g., per player).
- Uses rolling mean/var internally to bucket r estimates.

**`NormalDistributionPredictor`**: Simple Gaussian discretized to integer outcomes.
- Fixed `sigma`, uses `point_estimate_pred_column` as mean.

**`StudentTDistributionEstimator`**: Heavy-tailed alternative to normal.
- Heteroskedastic: learns sigma per conditioning bin.
- `support_cap_column`: can cap max outcome per row.

## Key patterns

### DataFrame requirement
```python
# WRONG - will raise TypeError
estimator.fit(X.to_numpy(), y)

# CORRECT
estimator.fit(X_dataframe, y)
```

### Distribution estimator flow
1. Upstream model predicts point estimate → stored in `point_estimate_pred_column`
2. Distribution estimator uses point estimate as mean
3. Learns variance/dispersion from training residuals
4. `predict_proba()` returns full distribution over `[min_value, max_value]`

## Common pitfalls
- Passing numpy array instead of DataFrame.
- Forgetting `point_estimate_pred_column` must exist in X for distribution estimators.
- `OrdinalClassifier` requires 3+ classes (not binary).
- `GranularityEstimator` fails if granularity value in predict wasn't seen in fit.
- Distribution estimators need `classes_` to match `range(min_value, max_value+1)`.

## When you change something, also update…
- sklearn interface (`fit`/`predict`/`predict_proba`) => ensure `classes_` is set
- Distribution parameters => update tests + any downstream scorers expecting specific shapes
- Granularity logic => affects `GroupByEstimator`, `GranularityEstimator`, `NegativeBinomialEstimator`
