# AGENTS.md — spforge

## Project overview
spforge is a sports prediction framework focused on feature generation (ratings/lags/rolling),
stateful transformers, and sklearn-compatible pipelines. It supports both pandas and polars via narwhals.

Primary themes:
- temporal ordering to prevent leakage
- stateful feature generators with `fit_transform`, `transform`, `future_transform`
- sklearn-compatible estimators/transformers and pipeline usage

## Non-negotiable invariants
- Support pandas + polars via narwhals (`@nw.narwhalify`)
- sklearn API compatibility: `fit(X, y)`, `transform(X)`, `predict(X)`, `predict_proba(X)`
- Public classes/Enums must be importable from module `__init__.py`
- Update version in `pyproject.toml` when releasing/pushing changes
- Always create a new branch for new tasks
- Always run all tests before committing

## Git hygiene
- Never commit IDE files (`.idea/`, `.vscode/`), temp files (`tmpclaude-*`, `tmp*`), or caches (`__pycache__/`, `*.pyc`)
- Only commit source code, config files (`.toml`, `.md`, `.yml`), test data, and documentation

## Code style
- Fail fast: raise errors early; no silent fallbacks
- Avoid comments unless logic is genuinely unintuitive
- Public methods at top, chronological order

## Extensibility protocol
- Never hardcode specific attribute names in consumer code.
- Use duck-typed protocol properties (e.g., `hasattr(obj, "context_features")`) instead of checking specific attribute names.

## Module-specific invariants and pitfalls

### cross_validator
- Training data must always be before validation data
- Estimator is deep-copied per fold
- Output includes `is_validation` column (1=validation, 0=training)
Pitfalls:
- With `AutoPipeline`, pass `features=pipeline.context_feature_names + pipeline.feature_names`
- Empty splits (check `min_validation_date`)
- Estimator not deepcopyable
- Remember `is_validation` when scoring if `add_training_predictions=True`

### estimator
- Require DataFrame input (not numpy)
- sklearn interface (`fit`, `predict`, `predict_proba`)
- Distribution estimators return shape `(n_samples, n_classes)`
Pitfalls:
- `OrdinalClassifier` needs 3+ classes
- `GranularityEstimator` fails if unseen granularity at predict
- Distribution `classes_` must match `range(min_value, max_value+1)`
- `point_estimate_pred_column` must exist for distribution estimators

### feature_generator
- `fit_transform()` must store historical data in `_df` for `future_transform()`
- Output row count must equal input row count
- Features shifted by 1 to avoid leakage
Pitfalls:
- `column_names` must be provided at init or `fit_transform()` for `future_transform()` to work
- `add_opponent=True` needs team_id + match_id
- `unique_constraint` mismatch causes duplicates or drops
- Forward-fill can surprise with new entities in `future_transform()`

### ratings
- `future_transform()` must not update state
- Updates strictly chronological (by match date)
- `performance_column` expected 0-1 (or set `auto_scale_performance=True`)
Pitfalls:
- Accidental mutation in feature-only paths
- Unsorted input leads to nonsense
- Mixing actual vs predicted performance scales
- Defense performance often inverted depending on definition

### scorer
- All scorers extend `BaseScorer` and implement `score(df) -> float | dict`
- Filters apply before scoring (including `validation_column == 1`)
- `aggregation_level` groups before scoring
- `granularity` returns `dict[tuple, float]`
Pitfalls:
- Forgetting `validation_column` means scoring training data too
- `OrdinalLossScorer` needs consecutive integer classes
- Distribution scorers expect list/array per row
- Empty df after filters returns 0.0/empty dict (not error)
- `aggregation_level` uses sum by default (not mean)

### transformers
- sklearn `TransformerMixin` interface
- Must handle pandas + polars via `@nw.narwhalify`
- `get_feature_names_out()` for sklearn integration
- Must be cloneable via sklearn `clone()`
Pitfalls:
- `EstimatorTransformer` outputs only prediction column (drops upstream)
- `OperatorTransformer` silently no-ops if columns missing
- `RatioEstimatorTransformer` requires `prediction_column_name` when `predict_row=False`

## Tests
- pytest, function-level only
- naming: `test_<unit>__<behavior>`
- test pandas and polars inputs with `@pytest.mark.parametrize`
- test public methods only (not private/protected)
