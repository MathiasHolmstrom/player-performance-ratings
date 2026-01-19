# hyperparameter_tuning — Claude notes

Optuna-based hyperparameter tuning for rating generators with cross-validation.

## Purpose
Optimize rating generator parameters (e.g., `rating_change_multiplier_offense`, `confidence_weight`) using cross-validation performance as the objective. Integrates with existing `CrossValidator` and `Scorer` infrastructure.

## Workflow
```
User provides:
├─ RatingGenerator (PlayerRatingGenerator or TeamRatingGenerator)
├─ CrossValidator (with estimator like AutoPipeline)
├─ Scorer (evaluation metric)
└─ Optional: Custom search space (partial override)

Per trial:
1. Deep copy rating_generator (fresh state)
2. Sample parameters from Optuna trial
3. Set parameters on copied generator
4. rating_gen.fit_transform(df) → df with rating features
5. cross_validator.generate_validation_df(df_with_features) → validation predictions
6. scorer.score(validation_df) → metric value
7. Return score to Optuna

Output:
└─ OptunaResult (best_params, best_value, study object)
```

## Critical Invariants

### 1. State Isolation via Deep Copy
**NEVER** reuse a rating generator across trials. Each trial MUST use `copy.deepcopy()`:
```python
copied_gen = copy.deepcopy(self.rating_generator)
```
Rationale: Rating generators maintain internal state (ratings history, confidence values). Without deep copy, trials pollute each other's state, making optimization invalid.

### 2. Chronological Data Order
Input DataFrame MUST be pre-sorted chronologically. The tuner does NOT sort data. Rating generators require chronological order for correct state evolution.

### 3. Explicit Direction
User MUST specify `direction="minimize"` or `"maximize"`. No auto-detection. This forces conscious decision about optimization goal.

### 4. Search Space Merging
Custom search space merges with defaults (custom takes precedence):
```python
default = {"param_a": spec_a, "param_b": spec_b}
custom = {"param_a": custom_spec_a}
merged = {"param_a": custom_spec_a, "param_b": spec_b}  # custom wins
```

### 5. Error Handling Strategy
Trials that fail (invalid params, numerical errors) return `inf` (minimize) or `-inf` (maximize). Study continues. Never crash the entire optimization.

## Common Pitfalls

### Forgetting to Deep Copy
```python
# WRONG - all trials share state
for trial in trials:
    self.rating_generator.fit_transform(df)

# CORRECT - each trial has independent state
for trial in trials:
    copied_gen = copy.deepcopy(self.rating_generator)
    copied_gen.fit_transform(df)
```

### Wrong Direction
```python
# WRONG - want to minimize loss but specified maximize
tuner = RatingHyperparameterTuner(..., direction="maximize")

# CORRECT
tuner = RatingHyperparameterTuner(..., direction="minimize")
```

### Dict Score Handling
If scorer returns dict (granular scores), tuner aggregates to mean. Ensure this makes sense for your use case:
```python
# Scorer returns: {"team_A": 0.5, "team_B": 0.7}
# Tuner uses: mean([0.5, 0.7]) = 0.6
```

### Not Pre-Sorting Data
```python
# WRONG - unsorted data
df = pd.read_csv("data.csv")
tuner.optimize(df)

# CORRECT - chronologically sorted
df = pd.read_csv("data.csv").sort_values("date")
tuner.optimize(df)
```

### Partial Override Confusion
```python
# This ONLY overrides rating_change_multiplier_offense
# All other defaults (confidence_weight, etc.) still apply
custom_space = {
    "rating_change_multiplier_offense": ParamSpec(...)
}
tuner = RatingHyperparameterTuner(..., param_search_space=custom_space)
```

## Usage Patterns

### Basic Usage (Defaults)
```python
from spforge import (
    PlayerRatingGenerator,
    AutoPipeline,
    MatchKFoldCrossValidator,
    OrdinalLossScorer,
    RatingHyperparameterTuner,
)

rating_gen = PlayerRatingGenerator(
    performance_column="points",
    column_names=column_names,
)

pipeline = AutoPipeline(
    estimator=RandomForestClassifier(),
    estimator_features=rating_gen.features_out + ["minutes", "is_home"],
)

cv = MatchKFoldCrossValidator(
    match_id_column_name="gameid",
    date_column_name="date",
    target_column="points",
    estimator=pipeline,
    prediction_column_name="points_pred",
    n_splits=3,
    features=pipeline.required_features,
)

scorer = OrdinalLossScorer(
    pred_column="points_pred",
    target="points",
    classes=range(0, 41),
    validation_column="is_validation",
)

tuner = RatingHyperparameterTuner(
    rating_generator=rating_gen,
    cross_validator=cv,
    scorer=scorer,
    direction="minimize",
    n_trials=50,
)

df = df.sort_values("date")  # CRITICAL
result = tuner.optimize(df)

# Apply best parameters
optimized_gen = PlayerRatingGenerator(
    performance_column="points",
    column_names=column_names,
    **result.best_params,
)
```

### Custom Search Space (Partial Override)
```python
from spforge import ParamSpec

custom_space = {
    "rating_change_multiplier_offense": ParamSpec(
        param_type="float",
        low=30.0,  # Narrower range than default
        high=80.0,
        log=True,
    ),
    # Add parameter not in defaults
    "confidence_days_ago_multiplier": ParamSpec(
        param_type="float",
        low=0.01,
        high=0.2,
        log=True,
    ),
}

tuner = RatingHyperparameterTuner(
    rating_generator=rating_gen,
    cross_validator=cv,
    scorer=scorer,
    direction="minimize",
    param_search_space=custom_space,  # Merged with defaults
    n_trials=100,
)
```

### Parallel + Persistent Storage
```python
tuner = RatingHyperparameterTuner(
    rating_generator=rating_gen,
    cross_validator=cv,
    scorer=scorer,
    direction="minimize",
    n_trials=100,
    n_jobs=4,  # 4 parallel trials
    storage="sqlite:///optuna_studies.db",
    study_name="nba_player_ratings_v1",
    timeout=3600,  # 1 hour max
)

result = tuner.optimize(df)

# Resume later
tuner2 = RatingHyperparameterTuner(
    rating_generator=rating_gen,
    cross_validator=cv,
    scorer=scorer,
    direction="minimize",
    n_trials=50,  # 50 more trials
    storage="sqlite:///optuna_studies.db",
    study_name="nba_player_ratings_v1",  # Same name loads existing
)
result2 = tuner2.optimize(df)
```

### Visualization
```python
import optuna.visualization as vis

result = tuner.optimize(df)

# Plot optimization history
fig = vis.plot_optimization_history(result.study)
fig.show()

# Plot parameter importances
fig = vis.plot_param_importances(result.study)
fig.show()

# Plot parallel coordinate
fig = vis.plot_parallel_coordinate(result.study)
fig.show()
```

## Default Search Spaces

### PlayerRatingGenerator (7 params)
- `rating_change_multiplier_offense`: float [20, 100] (log)
- `rating_change_multiplier_defense`: float [20, 100] (log)
- `confidence_weight`: float [0.5, 1.0]
- `confidence_value_denom`: float [50, 300]
- `confidence_max_sum`: float [50, 300]
- `use_off_def_split`: bool
- `performance_predictor`: categorical ["difference", "mean", "ignore_opponent"]

### TeamRatingGenerator (7 params)
Same as PlayerRatingGenerator

## ParamSpec Types

### Float
```python
ParamSpec(
    param_type="float",
    low=0.1,
    high=10.0,
    log=True,  # Log-scale sampling
)
```

### Int
```python
ParamSpec(
    param_type="int",
    low=1,
    high=100,
    step=5,  # Optional step size
)
```

### Categorical
```python
ParamSpec(
    param_type="categorical",
    choices=["option_a", "option_b", "option_c"],
)
```

### Bool
```python
ParamSpec(
    param_type="bool",
)
```

## Integration Points

### With CrossValidator
Tuner calls `cross_validator.generate_validation_df(df_with_features)`. The cross_validator must have:
- Configured estimator (e.g., AutoPipeline)
- Features list matching what will be in df_with_features
- Correct date/match_id columns

### With Scorer
Tuner calls `scorer.score(validation_df)`. The scorer must:
- Return float OR dict[str, float]
- Use validation_column to filter if needed
- Handle granular scoring (dict gets aggregated to mean)

### With RatingGenerator
Tuner calls:
1. `copy.deepcopy(rating_generator)` for isolation
2. `setattr(copied_gen, param_name, param_value)` for each param
3. `copied_gen.fit_transform(df)` to generate features

Rating generator must support:
- All parameters in search space as public attributes
- `fit_transform(df)` returning df with new features
- `features_out` property listing generated feature names

## Extensibility

To add new default parameters, edit `_default_search_spaces.py`:
```python
def get_default_player_rating_search_space() -> dict[str, ParamSpec]:
    return {
        # ... existing params ...
        "new_param": ParamSpec(
            param_type="float",
            low=0.0,
            high=1.0,
        ),
    }
```

To support new rating generator types, update `get_default_search_space()`:
```python
def get_default_search_space(rating_generator):
    if isinstance(rating_generator, NewRatingGeneratorType):
        return get_default_new_rating_search_space()
    # ... existing checks ...
```

## Testing Notes

Tests MUST verify:
1. Deep copy isolation (modify original, verify trial copies unaffected)
2. Direction correctness (minimize finds lower, maximize finds higher)
3. Custom search space merging (partial override works)
4. Dict score aggregation (mean of values)
5. Error handling (invalid params don't crash study)
6. Parallel execution (n_jobs > 1)
7. Storage persistence (study resumable)
8. Both pandas and polars DataFrames work
