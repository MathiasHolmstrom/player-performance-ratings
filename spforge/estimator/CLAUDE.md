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

## Context features protocol

**CRITICAL: Estimators that need context features should implement a `context_features` property.**

Context features are columns required for fitting but not passed to the underlying model (e.g., date columns for weighting, granularity columns for grouping).

### Implementing context_features

Estimators should implement a `context_features` property that returns a list of column names:

```python
@property
def context_features(self) -> list[str]:
    """Returns columns needed for fitting but not for the wrapped estimator."""
    return [self.date_column] if self.date_column else []
```

**Rules:**
- Return empty list `[]` if no context needed
- Return columns dynamically based on configuration (e.g., only if `date_column` is set)
- Deduplicate if multiple sources might provide the same column
- Duck-typed: no base class required, just add the property

**Why this matters:**
- Consumer code (AutoPipeline, EstimatorTransformer, etc.) auto-detects context via this property
- NEVER hardcode attribute checks (like `hasattr(obj, 'date_column')`) ANYWHERE in consumer code
- NEVER assume any estimator has specific attributes like `date_column`, `r_specific_granularity`, etc.
- Extensible: new estimators just add the property, zero changes needed in consumer code

### Examples

- `SkLearnEnhancerEstimator`: Returns `[date_column]` if configured for temporal weighting
- `NegativeBinomialEstimator`: Returns `r_specific_granularity` + `column_names` fields if configured
- Plain sklearn estimators: Don't have the property (consumers check with `hasattr`)

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
