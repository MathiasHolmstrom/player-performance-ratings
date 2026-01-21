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
        Merge custom search space with defaults (custom takes precedence).

        Args:
            custom: Custom search space (may be None)
            defaults: Default search space

        Returns:
            Merged search space
        """
        merged = defaults.copy()
        if custom:
            merged.update(custom)
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
