import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import optuna
from narwhals.typing import IntoFrameT

from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.ratings import PlayerRatingGenerator, TeamRatingGenerator
from spforge.scorer import BaseScorer

logger = logging.getLogger(__name__)


@dataclass
class ParamSpec:
    """
    Declarative parameter specification for search space.

    Attributes:
        param_type: Type of parameter ("float", "int", "categorical", "bool")
        low: Lower bound for float/int parameters
        high: Upper bound for float/int parameters
        log: Whether to use log-scale sampling for float/int
        step: Step size for int parameters
        choices: List of choices for categorical parameters
    """

    param_type: Literal["float", "int", "categorical", "bool"]
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    step: float | int | None = None
    choices: list[Any] | None = None

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        """Generate suggestion using appropriate trial method."""
        if self.param_type == "float":
            if self.low is None or self.high is None:
                raise ValueError(f"float parameter '{name}' requires low and high bounds")
            return trial.suggest_float(name, self.low, self.high, log=self.log)
        elif self.param_type == "int":
            if self.low is None or self.high is None:
                raise ValueError(f"int parameter '{name}' requires low and high bounds")
            if self.step is None:
                return trial.suggest_int(name, int(self.low), int(self.high))
            return trial.suggest_int(name, int(self.low), int(self.high), step=self.step)
        elif self.param_type == "categorical":
            if self.choices is None:
                raise ValueError(f"categorical parameter '{name}' requires choices")
            return trial.suggest_categorical(name, self.choices)
        elif self.param_type == "bool":
            return trial.suggest_categorical(name, [True, False])
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


@dataclass
class OptunaResult:
    """
    Result container from optimization.

    Attributes:
        best_params: Dictionary of best parameter values found
        best_value: Best objective value achieved
        best_trial: The best trial object
        study: The Optuna study object for further analysis/visualization
    """

    best_params: dict[str, Any]
    best_value: float
    best_trial: optuna.trial.FrozenTrial
    study: optuna.Study


class RatingHyperparameterTuner:
    """
    Main tuning orchestrator for rating generator hyperparameters.

    Uses Optuna to optimize rating generator parameters by evaluating
    performance through cross-validation with a specified scorer.
    """

    def __init__(
        self,
        rating_generator: PlayerRatingGenerator | TeamRatingGenerator,
        cross_validator: MatchKFoldCrossValidator,
        scorer: BaseScorer,
        direction: Literal["minimize", "maximize"],
        param_search_space: dict[str, ParamSpec] | None = None,
        param_ranges: dict[str, tuple[float | int, float | int]] | None = None,
        exclude_params: list[str] | None = None,
        fixed_params: dict[str, Any] | None = None,
        n_trials: int = 50,
        n_jobs: int = 1,
        storage: str | None = None,
        study_name: str | None = None,
        timeout: float | None = None,
        show_progress_bar: bool = True,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            rating_generator: The rating generator to tune (will be deep copied per trial)
            cross_validator: Pre-configured cross-validator with estimator
            scorer: Scorer for evaluation (must have score(df) -> float | dict)
            direction: "minimize" or "maximize"
            param_search_space: Custom search space (merges with defaults if provided)
            param_ranges: Easy range override for float/int params. Maps param name to
                (low, high) tuple. Preserves param_type and log scale from defaults.
                Example: {"confidence_weight": (0.2, 1.0)}
            exclude_params: List of param names to exclude from tuning entirely.
                Example: ["performance_predictor", "use_off_def_split"]
            fixed_params: Parameters to fix at specific values (not tuned).
                These values are applied to the rating generator each trial.
                Example: {"performance_predictor": "mean"}
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs (1 = sequential)
            storage: Optuna storage URL (e.g., "sqlite:///optuna.db") for persistence
            study_name: Name for the study (required if using storage)
            timeout: Stop after N seconds
            show_progress_bar: Display Optuna progress bar
            sampler: Custom Optuna sampler (default: TPESampler)
            pruner: Custom Optuna pruner (default: None)
        """
        self.rating_generator = rating_generator
        self.cross_validator = cross_validator
        self.scorer = scorer
        self.direction = direction
        self.custom_search_space = param_search_space
        self.param_ranges = param_ranges
        self.exclude_params = exclude_params or []
        self.fixed_params = fixed_params or {}
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.storage = storage
        self.study_name = study_name
        self.timeout = timeout
        self.show_progress_bar = show_progress_bar
        self.sampler = sampler
        self.pruner = pruner

        if direction not in ["minimize", "maximize"]:
            raise ValueError(f"direction must be 'minimize' or 'maximize', got: {direction}")

        if storage is not None and study_name is None:
            raise ValueError("study_name is required when using storage")

    def optimize(self, df: IntoFrameT) -> OptunaResult:
        """
        Run hyperparameter optimization on the provided DataFrame.

        Args:
            df: Input DataFrame with historical data

        Returns:
            OptunaResult containing best parameters, best value, and study object
        """
        from spforge.hyperparameter_tuning._default_search_spaces import (
            get_default_search_space,
        )

        default_search_space = get_default_search_space(self.rating_generator)
        search_space = self._merge_search_spaces(self.custom_search_space, default_search_space)

        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True if self.storage else False,
        )

        study.optimize(
            lambda trial: self._objective(trial, df, search_space),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=self.show_progress_bar,
        )

        return OptunaResult(
            best_params=study.best_params,
            best_value=study.best_value,
            best_trial=study.best_trial,
            study=study,
        )

    def _objective(
        self, trial: optuna.Trial, df: IntoFrameT, search_space: dict[str, ParamSpec]
    ) -> float:
        """
        Objective function for a single trial.

        Args:
            trial: Optuna trial object
            df: Input DataFrame
            search_space: Merged search space

        Returns:
            Score value (float)
        """
        try:
            copied_gen = copy.deepcopy(self.rating_generator)

            for param_name, param_value in self.fixed_params.items():
                setattr(copied_gen, param_name, param_value)

            trial_params = self._suggest_params(trial, search_space)

            for param_name, param_value in trial_params.items():
                setattr(copied_gen, param_name, param_value)

            df_with_features = copied_gen.fit_transform(df)

            validation_df = self.cross_validator.generate_validation_df(df_with_features)

            score = self.scorer.score(validation_df)

            score_value = self._aggregate_score(score)

            if math.isnan(score_value) or math.isinf(score_value):
                logger.warning(f"Trial {trial.number} returned invalid score: {score_value}")
                return float("inf") if self.direction == "minimize" else float("-inf")

            return score_value

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed with error: {e}")
            return float("inf") if self.direction == "minimize" else float("-inf")

    def _suggest_params(
        self, trial: optuna.Trial, search_space: dict[str, ParamSpec]
    ) -> dict[str, Any]:
        """
        Suggest parameters for this trial.

        Args:
            trial: Optuna trial object
            search_space: Parameter search space

        Returns:
            Dictionary of suggested parameter values
        """
        params = {}
        for param_name, param_spec in search_space.items():
            params[param_name] = param_spec.suggest(trial, param_name)
        return params

    def _merge_search_spaces(
        self,
        custom: dict[str, ParamSpec] | None,
        defaults: dict[str, ParamSpec],
    ) -> dict[str, ParamSpec]:
        """
        Merge custom search space with defaults.

        Priority order (highest to lowest):
        1. exclude_params - removes param entirely
        2. fixed_params - removes from search (applied separately)
        3. custom (param_search_space) - full ParamSpec override
        4. param_ranges - updates only low/high bounds
        5. defaults - base search space

        Args:
            custom: Custom search space (may be None)
            defaults: Default search space

        Returns:
            Merged search space (excludes fixed_params, those are applied separately)
        """
        merged = defaults.copy()

        if self.param_ranges:
            for param_name, (low, high) in self.param_ranges.items():
                if param_name not in merged:
                    raise ValueError(
                        f"param_ranges contains unknown parameter: '{param_name}'. "
                        f"Available parameters: {list(merged.keys())}"
                    )
                existing = merged[param_name]
                if existing.param_type not in ("float", "int"):
                    raise ValueError(
                        f"param_ranges can only override float/int parameters. "
                        f"'{param_name}' is {existing.param_type}."
                    )
                merged[param_name] = ParamSpec(
                    param_type=existing.param_type,
                    low=low,
                    high=high,
                    log=existing.log,
                    step=existing.step,
                )

        if custom:
            merged.update(custom)

        for param_name in self.exclude_params:
            merged.pop(param_name, None)

        for param_name in self.fixed_params:
            merged.pop(param_name, None)

        return merged

    @staticmethod
    def _aggregate_score(score: float | dict) -> float:
        """
        Convert dict scores to float (mean of values).

        Args:
            score: Score value (float or dict)

        Returns:
            Aggregated float score
        """
        if isinstance(score, dict):
            values = list(score.values())
            if any(math.isnan(v) or math.isinf(v) for v in values):
                raise ValueError("Scorer returned invalid values in dict")
            return float(np.mean(values))
        return float(score)


def _is_estimator(obj: object) -> bool:
    return hasattr(obj, "get_params") and hasattr(obj, "set_params")


def _get_leaf_estimator_paths(estimator: Any) -> dict[str, Any]:
    if not _is_estimator(estimator):
        raise ValueError("estimator must implement get_params and set_params")

    params = estimator.get_params(deep=True)
    estimator_keys = [k for k, v in params.items() if _is_estimator(v)]

    if not estimator_keys:
        return {"": estimator}

    leaves: list[str] = []
    for key in estimator_keys:
        if not any(other != key and other.startswith(f"{key}__") for other in estimator_keys):
            leaves.append(key)

    return {key: params[key] for key in sorted(leaves)}


def _build_search_space_for_targets(
    targets: dict[str, dict[str, ParamSpec]],
) -> dict[str, ParamSpec]:
    search_space: dict[str, ParamSpec] = {}
    for path, params in targets.items():
        for param_name, param_spec in params.items():
            full_name = f"{path}__{param_name}" if path else param_name
            if full_name in search_space:
                raise ValueError(f"Duplicate parameter name detected: {full_name}")
            search_space[full_name] = param_spec
    return search_space


def _enqueue_predicted_r_weight_zero(study: optuna.Study, search_space: dict[str, ParamSpec]):
    zero_params: dict[str, float] = {}
    for name, spec in search_space.items():
        if not name.endswith("predicted_r_weight"):
            continue
        if spec.param_type not in {"float", "int"}:
            continue
        if spec.low is None or spec.high is None:
            continue
        if spec.low <= 0 <= spec.high:
            zero_params[name] = 0.0

    if zero_params:
        study.enqueue_trial(zero_params)


class EstimatorHyperparameterTuner:
    """
    Hyperparameter tuner for sklearn-compatible estimators.

    Supports nested estimators and can target deepest leaf estimators.
    """

    def __init__(
        self,
        estimator: Any,
        cross_validator: MatchKFoldCrossValidator,
        scorer: BaseScorer,
        direction: Literal["minimize", "maximize"],
        param_search_space: dict[str, ParamSpec] | None = None,
        param_targets: dict[str, dict[str, ParamSpec]] | None = None,
        n_trials: int = 50,
        n_jobs: int = 1,
        storage: str | None = None,
        study_name: str | None = None,
        timeout: float | None = None,
        show_progress_bar: bool = True,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ):
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.scorer = scorer
        self.direction = direction
        self.param_search_space = param_search_space
        self.param_targets = param_targets
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.storage = storage
        self.study_name = study_name
        self.timeout = timeout
        self.show_progress_bar = show_progress_bar
        self.sampler = sampler
        self.pruner = pruner

        if direction not in ["minimize", "maximize"]:
            raise ValueError(f"direction must be 'minimize' or 'maximize', got: {direction}")

        if storage is not None and study_name is None:
            raise ValueError("study_name is required when using storage")

        if param_search_space is not None and param_targets is not None:
            raise ValueError("param_search_space and param_targets cannot both be provided")

    def optimize(self, df: IntoFrameT) -> OptunaResult:
        from spforge.hyperparameter_tuning._default_search_spaces import (
            get_default_estimator_search_space,
        )

        leaf_estimators = _get_leaf_estimator_paths(self.estimator)
        default_targets = {
            path: get_default_estimator_search_space(est)
            for path, est in leaf_estimators.items()
        }
        default_targets = {path: space for path, space in default_targets.items() if space}

        if self.param_targets is not None:
            unknown = set(self.param_targets) - set(leaf_estimators)
            if unknown:
                raise ValueError(f"param_targets contains unknown estimator paths: {unknown}")
            targets = self.param_targets
        elif self.param_search_space is not None:
            targets = {path: self.param_search_space for path in leaf_estimators}
        elif default_targets:
            targets = default_targets
        else:
            raise ValueError(
                "param_search_space is required when no default search space is available"
            )

        search_space = _build_search_space_for_targets(targets)
        if not search_space:
            raise ValueError("Resolved search space is empty")

        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True if self.storage else False,
        )

        _enqueue_predicted_r_weight_zero(study, search_space)

        study.optimize(
            lambda trial: self._objective(trial, df, search_space),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=self.show_progress_bar,
        )

        return OptunaResult(
            best_params=study.best_params,
            best_value=study.best_value,
            best_trial=study.best_trial,
            study=study,
        )

    def _objective(
        self, trial: optuna.Trial, df: IntoFrameT, search_space: dict[str, ParamSpec]
    ) -> float:
        try:
            trial_params = self._suggest_params(trial, search_space)

            copied_estimator = copy.deepcopy(self.estimator)
            copied_estimator.set_params(**trial_params)

            cv = copy.deepcopy(self.cross_validator)
            cv.estimator = copied_estimator

            validation_df = cv.generate_validation_df(df)
            score = self.scorer.score(validation_df)
            score_value = RatingHyperparameterTuner._aggregate_score(score)

            if math.isnan(score_value) or math.isinf(score_value):
                logger.warning(f"Trial {trial.number} returned invalid score: {score_value}")
                return float("inf") if self.direction == "minimize" else float("-inf")

            return score_value

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed with error: {e}")
            return float("inf") if self.direction == "minimize" else float("-inf")

    def _suggest_params(
        self, trial: optuna.Trial, search_space: dict[str, ParamSpec]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for param_name, param_spec in search_space.items():
            params[param_name] = param_spec.suggest(trial, param_name)
        return params
