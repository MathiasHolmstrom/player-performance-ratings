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
- Any class or Enum intended to be instantiated or referenced by users MUST be importable from the package's `__init__.py`.
- Users should not need to import from private modules (`_*.py`).
- Update version when in toml pushing a new change.
- When working on a new task. Always create a new branch.

## Git hygiene
**CRITICAL: Never commit IDE or temporary files.**

Before committing, always check `git status` and ensure you're not adding:
- IDE configuration files (`.idea/`, `.vscode/`, `*.iml`, etc.)
- Temporary files (tmpclaude-*, tmp*, *.tmp, etc.)
- OS-specific files (.DS_Store, Thumbs.db, etc.)
- Python cache (`__pycache__/`, `*.pyc`, etc.)
- Virtual environments (`.venv/`, `venv/`, etc.)

These files are listed in `.gitignore`. If you see them in `git status`:
1. DO NOT add them to commits
2. Verify they're in `.gitignore`
3. Use `git rm --cached <file>` if accidentally staged

Only commit:
- Source code (`.py` files in `spforge/`, `tests/`)
- Configuration files (`.toml`, `.md`, `.yml` if relevant)
- Test data files (in `tests/` or `examples/data/`)
- Documentation


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

**Breaking change (v0.7.0):** `context_feature_names` and `context_predictor_transformer_feature_names` parameters have been **removed**. Context features are now auto-computed from predictor_transformers.

**New property:**
- `required_features`: All features needed by the pipeline (feature_names + context + granularity + filters)

**Cross-validator usage:**
```python
cross_validator = MatchKFoldCrossValidator(
    estimator=pipeline,
    features=pipeline.required_features,  # Or omit entirely - auto-detected!
    ...
)
```

Key gotchas:
- Context features are auto-computed from predictor_transformers' `context_features` property + estimator's `context_features` property.
- AutoPipeline uses protocol-based detection: checks `hasattr(obj, 'context_features')` first, falls back to legacy attribute checks.
- `categorical_handling="auto"` must be deterministic and covered by tests.
- `native` categorical handling is for LightGBM (category dtype).

## Code style guide
- Public methods/classes: docstring explaining intended usage + key parameters.
- Avoid extra comments within the code; only comment where logic is genuinely unintuitive.
- Prefer minimal diffs; do not add new abstractions unless they remove more complexity than they add.
- Fail fast: raise errors as early as possible (init if static; otherwise in `fit`/`transform`). Never infer intent, auto-correct input, or silently fall back. Any non-strict behavior must be explicit (e.g. `strict=False`) and documented.
- Write methods in chronological order based on when they are called (public methods at the top)

## Extensibility principles

**CRITICAL: NEVER hardcode specific attribute names in consumer code.**

This applies EVERYWHERE - AutoPipeline, transformers, cross-validators, scorers, any code consuming objects from other modules.

**Rules:**
- ✅ DO: Use duck-typed protocol properties that objects can implement
- ✅ DO: Check for the protocol property (`hasattr(obj, 'protocol_property')`)
- ❌ DON'T: Hardcode specific attribute names (`date_column`, `r_specific_granularity`, `column_names`, etc.)
- ❌ DON'T: Assume any object has a specific attribute
- ❌ DON'T: Maintain lists of known types/attributes in consumer code
- ❌ DON'T: Add if/elif chains checking for different attribute names

**Why this matters:**
- New types shouldn't require updating consumer code
- Reduces coupling between modules
- Makes the codebase extensible by users
- Duck typing is more Pythonic and maintainable

**Example (context features):**
```python
# GOOD: Protocol-based, extensible
# Objects that need context implement the context_features property
if hasattr(estimator, 'context_features'):
    context.extend(estimator.context_features)

# BAD: Hardcoded attribute names, not extensible
# Every new estimator type requires updating this code
if hasattr(estimator, 'date_column') and estimator.date_column:
    context.append(estimator.date_column)
if hasattr(estimator, 'r_specific_granularity') and estimator.r_specific_granularity:
    context.extend(estimator.r_specific_granularity)
if hasattr(estimator, 'column_names') and estimator.column_names:
    # Extract individual fields...
# ... needs updating every time a new estimator is added
```

**How to design extensible code:**
1. Define a protocol property that objects can implement (e.g., `context_features`)
2. Make objects compute their needs in that property (dynamic, based on configuration)
3. Consumer code just checks for the protocol property
4. Legacy fallback is OK during migration, but should be clearly marked for removal

**When you need to add new capabilities:**
- If you find yourself checking for a specific attribute name → STOP
- Ask: "Could other objects need this capability in the future?"
- If yes: Design a protocol property instead
- Document it in the relevant CLAUDE.md file
- Update existing objects to implement the protocol
- Consumer code checks for the protocol, not specific attributes

See `estimator.CLAUDE.md` and `transformers.CLAUDE.md` for the context_features protocol example.



## Unit test conventions
- pytest, function-level tests only (no `Test*` classes).
- Naming: `test_<unit>__<behavior>` (use `__` to separate context from behavior).
  Examples:
  - `test_PlayerRatingGenerator_future_transform__does_not_mutate_state`
- Tests must run for both pandas and polars inputs where supported.
- Prefer .parameterize for testing different inputs (e.g. pandas vs polars)
- Test behaviours through public methods. Only in rare cases must you test the protected methods.
- Prefer simple tests with specific input and expected output values.

## Canonical end-to-end example
See `tests/e2e/test_nba_player_points.py::test_nba_player_points` (or equivalent path).

This is the reference integration test for how components are composed:
FeatureGeneratorPipeline -> MatchKFoldCrossValidator -> AutoPipeline (+ predictor_transformers) -> estimator.

