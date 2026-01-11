# spforge — Claude notes (root)

## Purpose
Sports prediction framework: feature generation (ratings/lags/etc.) + sklearn-compatible pipelines.
Supports pandas/polars via narwhals.

## Where to find module-specific rules
- cross_validator.CLAUDE.md
- estimator.CLAUDE.md
- feature_generator.CLAUDE.md
- ratings.CLAUDE.md
- scorer.CLAUDE.md
- transformers.CLAUDE.md


## Non-negotiable invariants
- Public-facing generators/transformers must support pandas+polars via narwhals (use `@nw.narwhalify` on fit/transform unless documented otherwise).
- sklearn API compatibility: `fit(X, y)`, `transform(X)`, `predict(X)`, `predict_proba(X)` where applicable.
- Feature generators (ratings/lags/etc.) MUST NOT change row count in `fit_transform/transform/future_transform`.
- Pipelines may filter rows during `fit` only (if designed that way), but `transform/future_transform` must preserve row count unless explicitly documented.
- Stateful generators store internal mutable state in private dicts; feature-only paths must not mutate state.
- No mutations - always return a new frame via `with_columns` / assignment copy.
- Any class or Enum intended to be instantiated or referenced by users MUST be importable from the package’s `__init__.py`.
- Users should not need to import from private modules (`_*.py`).


## Core classes (high level)

### FeatureGeneratorPipeline
- Chains feature generators sequentially.
- Contract: preserves row count and does not create duplicate feature names.
- Lifecycle:
  - `fit_transform`: initialize state using historical outcomes
  - `transform`: apply to more historical data (updates state)
  - `future_transform`: compute pre-match features only (no outcome-dependent logic)

### AutoPipeline
Sklearn-compatible estimator pipeline with preprocessing + optional predictor_transformers.
Key gotchas:
- Provide `context_feature_names` for any columns needed by predictor_transformers but not the final estimator.
- `categorical_handling="auto"` must be deterministic and covered by tests.
- `native` categorical handling is for LightGBM (category dtype).

## When you change something, also update…
- FeatureGeneratorPipeline behavior => update generator integration tests (pandas + polars).
- AutoPipeline categorical handling => update `_resolve_categorical_handling()` + tests for each mode.
- Anything affecting feature names => update downstream consumers + tests.
- Anything stateful => add tests that assert “future_transform does not mutate state”.

## Coding standards
- Public methods/classes: docstring explaining intended usage + key parameters.
- Avoid extra comments; only comment where logic is genuinely unintuitive.
- Prefer minimal diffs; do not add new abstractions unless they remove more complexity than they add.
- Raise errors as early as possible (init if static; otherwise in `fit`/`transform`); never silently continue.
- Fail fast on unexpected or invalid input.
- Do NOT infer intent, auto-correct input, or silently fall back.
- Any non-strict behavior must be explicit (e.g. `strict=False`) and documented; otherwise raise a clear error.



## Unit test conventions
- pytest, function-level tests only (no `Test*` classes).
- Naming: `test_<unit>__<behavior>` (use `__` to separate context from behavior).
  Examples:
  - `test_PlayerRatingGenerator_future_transform__does_not_mutate_state`
  - `test_confidence_decay__reduces_multiplier_over_time`
- Tests must run for both pandas and polars inputs where supported.
- Prefer .parameterize for testing different inputs (e.g. pandas vs polars)

## Canonical end-to-end example
See `tests/e2e/test_nba_player_points.py::test_nba_player_points` (or equivalent path).

This is the reference integration test for how components are composed:
FeatureGeneratorPipeline -> MatchKFoldCrossValidator -> AutoPipeline (+ predictor_transformers) -> estimator.

