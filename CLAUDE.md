# spforge — Claude notes

Sports prediction framework: feature generation (ratings/lags/etc.) + sklearn-compatible pipelines. Supports pandas/polars via narwhals.

## Module-specific rules
See: cross_validator.CLAUDE.md, estimator.CLAUDE.md, feature_generator.CLAUDE.md, ratings.CLAUDE.md, scorer.CLAUDE.md, transformers.CLAUDE.md

## Non-negotiable invariants
- Support pandas+polars via narwhals (`@nw.narwhalify`)
- sklearn API compatibility: `fit(X, y)`, `transform(X)`, `predict(X)`, `predict_proba(X)`
- Feature generators MUST NOT change row count in transform/future_transform
- No mutations - always return new frame via `with_columns` / assignment copy
- Public classes/Enums must be importable from `__init__.py`
- Update version in toml when pushing changes
- Always create a new branch for new tasks

## Git hygiene
Never commit IDE files (.idea/, .vscode/), temporary files (tmpclaude-*, tmp*), or Python cache (__pycache__/, *.pyc).
Only commit source code, config files (.toml, .md, .yml), test data, and documentation.

## Code style
- Fail fast: raise errors early. Never infer intent or silently fall back.
- Avoid comments; only comment where logic is genuinely unintuitive.
- Public methods at top, chronological order.

## Extensibility
NEVER hardcode specific attribute names in consumer code. Use duck-typed protocol properties (`hasattr(obj, 'context_features')`) instead of checking for specific attributes (`date_column`, `r_specific_granularity`, etc.). See module-specific CLAUDE.md files for protocol examples.

## Tests
- pytest, function-level only. Naming: `test_<unit>__<behavior>`
- Test pandas and polars inputs via `.parameterize`
- Reference: `tests/e2e/test_nba_player_points.py`

