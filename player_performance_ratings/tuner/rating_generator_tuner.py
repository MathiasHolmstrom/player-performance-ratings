import copy
import inspect
import logging
from abc import abstractmethod, ABC
from typing import Optional, Match

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial
from player_performance_ratings.tuner.start_rating_optimizer import (
    StartLeagueRatingOptimizer,
)

from player_performance_ratings.cross_validator.cross_validator import CrossValidator
from player_performance_ratings.ratings.rating_calculators import MatchRatingGenerator

from player_performance_ratings.ratings.rating_calculators.start_rating_generator import (
    StartRatingGenerator,
)
from player_performance_ratings.ratings import UpdateRatingGenerator

from player_performance_ratings.ratings.rating_generator import RatingGenerator

from player_performance_ratings import PipelineFactory
from player_performance_ratings.tuner.utils import (
    ParameterSearchRange,
    add_params_from_search_range,
)

DEFAULT_TEAM_SEARCH_RANGES = [
    ParameterSearchRange(name="confidence_weight", type="uniform", low=0.6, high=0.95),
    ParameterSearchRange(
        name="confidence_days_ago_multiplier",
        type="uniform",
        low=0.02,
        high=0.12,
    ),
    ParameterSearchRange(
        name="confidence_max_days",
        type="uniform",
        low=40,
        high=200,
    ),
    ParameterSearchRange(
        name="confidence_max_sum",
        type="uniform",
        low=20,
        high=180,
    ),
    ParameterSearchRange(
        name="confidence_value_denom",
        type="uniform",
        low=20,
        high=180,
    ),
    ParameterSearchRange(
        name="rating_change_multiplier", type="uniform", low=25, high=140
    ),
    ParameterSearchRange(
        name="min_rating_change_multiplier_ratio",
        type="uniform",
        low=0.05,
        high=0.2,
    ),
    ParameterSearchRange(
        name="team_id_change_confidence_sum_decrease",
        type="uniform",
        low=0,
        high=10,
    ),
]

DEFAULT_START_RATING_SEARCH_RANGE = [
    ParameterSearchRange(
        name="league_quantile",
        type="uniform",
        low=0.12,
        high=0.5,
    ),
    ParameterSearchRange(
        name="min_count_for_percentiles",
        type="int",
        low=50,
        high=200,
    ),
    ParameterSearchRange(name="team_rating_subtract", type="int", low=0, high=200),
    ParameterSearchRange(name="team_weight", type="uniform", low=0, high=1),
]


class RatingGeneratorTuner(ABC):

    @abstractmethod
    def tune(
        self,
        df: pd.DataFrame,
        rating_idx: int,
        cross_validator: CrossValidator,
        pipeline_factory: PipelineFactory,
        matches: list[Match],
    ) -> RatingGenerator:
        pass


class UpdateRatingGeneratorTuner(RatingGeneratorTuner):

    def __init__(
        self,
        team_rating_search_ranges: Optional[list[ParameterSearchRange]] = None,
        team_rating_n_trials: int = 30,
        start_rating_search_ranges: Optional[list[ParameterSearchRange]] = None,
        start_rating_n_trials: int = 8,
        optimize_league_ratings: bool = False,
    ):
        self.team_rating_search_ranges = (
            team_rating_search_ranges or DEFAULT_TEAM_SEARCH_RANGES
        )
        self.start_rating_search_ranges = (
            start_rating_search_ranges or DEFAULT_START_RATING_SEARCH_RANGE
        )
        self.team_rating_n_trials = team_rating_n_trials
        self.start_rating_n_trials = start_rating_n_trials
        self.optimize_league_ratings = optimize_league_ratings

    def tune(
        self,
        df: pd.DataFrame,
        rating_idx: int,
        cross_validator: CrossValidator,
        pipeline_factory: PipelineFactory,
        matches: list[Match],
    ) -> UpdateRatingGenerator:

        if pipeline_factory.rating_generators:
            best_rating_generator = copy.deepcopy(
                pipeline_factory.rating_generators[rating_idx]
            )
        else:
            raise ValueError("rating_generators are not specified")
        # potential_rating_features = [v for k, v in RatingColumnNames.__dict__.items() if isinstance(v, str)]

        # best_rating_generator = UpdateRatingGenerator(
        # features_out=[f for f in pipeline_factory.predictor.features if f in potential_rating_features],
        #      features_out=pipeline_factory.predictor.features,
        #      column_names=pipeline_factory.rating_generators[rating_idx].column_names
        #  )

        if self.team_rating_n_trials > 0:
            logging.info("Tuning Team Rating")
            best_team_rating_generator = self._tune_team_rating(
                df=df,
                rating_generator=best_rating_generator,
                rating_index=rating_idx,
                cross_validator=cross_validator,
                matches=matches,
                pipeline_factory=pipeline_factory,
            )

            best_rating_generator.match_rating_generator = best_team_rating_generator

        if self.optimize_league_ratings:
            start_rating_optimizer = StartLeagueRatingOptimizer(
                pipeline_factory=pipeline_factory,
                cross_validator=cross_validator,
            )
            optimized_league_ratings = start_rating_optimizer.optimize(
                df=df,
                rating_model_idx=rating_idx,
                matches=matches,
                rating_generator=best_rating_generator,
                column_names=pipeline_factory.column_names,
            )
            best_rating_generator.match_rating_generator.start_rating_generator.league_ratings = (
                optimized_league_ratings
            )

        if self.start_rating_n_trials > 0:
            logging.info("Tuning Start Rating")

            best_start_rating = self._tune_start_rating(
                df=df,
                matches=matches,
                rating_generator=best_rating_generator,
                rating_index=rating_idx,
                cross_validator=cross_validator,
                match_predictor_factory=pipeline_factory,
            )
            best_rating_generator.match_rating_generator.start_rating_generator = (
                best_start_rating
            )

        return UpdateRatingGenerator(
            match_rating_generator=best_rating_generator.match_rating_generator,
            non_estimator_known_features_out=best_rating_generator.non_estimator_known_features_out,
            known_features_out=best_rating_generator._known_features_out,
            performance_column=best_rating_generator.performance_column,
        )

    def _tune_team_rating(
        self,
        df: pd.DataFrame,
        rating_generator: UpdateRatingGenerator,
        rating_index: int,
        matches: list[Match],
        cross_validator: CrossValidator,
        pipeline_factory: PipelineFactory,
    ) -> MatchRatingGenerator:

        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:

            team_rating_generator_params = list(
                inspect.signature(
                    rating_generator.match_rating_generator.__class__.__init__
                ).parameters.keys()
            )[1:]

            params = {
                attr: getattr(rating_generator.match_rating_generator, attr)
                for attr in team_rating_generator_params
                if attr not in ("performance_predictor", "start_rating_generator")
            }
            params = add_params_from_search_range(
                params=params,
                trial=trial,
                parameter_search_range=self.team_rating_search_ranges,
            )

            performance_predictor = copy.deepcopy(
                rating_generator.match_rating_generator.performance_predictor
            )
            performance_predictor_params = list(
                inspect.signature(
                    performance_predictor.__class__.__init__
                ).parameters.keys()
            )[1:]

            for param in params.copy():
                if param in performance_predictor_params:
                    performance_predictor.__setattr__(param, params[param])
                    params.pop(param)

            team_rating_generator = MatchRatingGenerator(
                **params,
                performance_predictor=performance_predictor,
                start_rating_generator=copy.deepcopy(
                    rating_generator.match_rating_generator.start_rating_generator
                )
            )

            rating_g = copy.deepcopy(rating_generator)
            rating_g.match_rating_generator = team_rating_generator
            rating_generators = copy.deepcopy(pipeline_factory.rating_generators)

            rating_generators[rating_index] = rating_g
            pipeline = pipeline_factory.create(
                rating_generators=rating_generators,
            )
            return pipeline.cross_validate_score(
                df=df,
                matches=matches,
                cross_validator=cross_validator,
                create_performance=False,
            )

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(
            direction=direction, study_name=study_name, sampler=sampler
        )
        callbacks = []
        study.optimize(
            lambda trial: objective(trial, df),
            n_trials=self.team_rating_n_trials,
            callbacks=callbacks,
        )

        best_params = study.best_params
        team_rating_generator_params = list(
            inspect.signature(
                rating_generator.match_rating_generator.__class__.__init__
            ).parameters.keys()
        )[1:]
        other_params = {
            attr: getattr(rating_generator.match_rating_generator, attr)
            for attr in team_rating_generator_params
            if attr not in ("performance_predictor", "start_rating_generator")
        }
        for param in other_params:
            if param not in best_params:
                best_params[param] = other_params[param]

        performance_predictor = (
            rating_generator.match_rating_generator.performance_predictor
        )
        performance_predictor_params = list(
            inspect.signature(
                performance_predictor.__class__.__init__
            ).parameters.keys()
        )[1:]

        for param in best_params.copy():
            if param in performance_predictor_params:
                performance_predictor.__setattr__(param, best_params[param])
                best_params.pop(param)
        return MatchRatingGenerator(
            **best_params,
            performance_predictor=performance_predictor,
            start_rating_generator=rating_generator.match_rating_generator.start_rating_generator
        )

    def _tune_start_rating(
        self,
        df: pd.DataFrame,
        rating_generator: UpdateRatingGenerator,
        rating_index: int,
        matches: list[Match],
        cross_validator: CrossValidator,
        match_predictor_factory: PipelineFactory,
    ):
        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:
            start_rating_generator_params = list(
                inspect.signature(
                    rating_generator.match_rating_generator.start_rating_generator.__class__.__init__
                ).parameters.keys()
            )[1:]

            params = {
                attr: getattr(
                    rating_generator.match_rating_generator.start_rating_generator, attr
                )
                for attr in start_rating_generator_params
                if attr != "league_ratings"
            }

            params = add_params_from_search_range(
                params=params,
                trial=trial,
                parameter_search_range=self.start_rating_search_ranges,
            )

            league_ratings = copy.deepcopy(
                rating_generator.match_rating_generator.start_rating_generator.league_ratings
            )
            start_rating_generator = StartRatingGenerator(
                league_ratings=league_ratings, **params
            )
            rating_g = copy.deepcopy(rating_generator)
            rating_g.match_rating_generator.start_rating_generator = (
                start_rating_generator
            )
            if match_predictor_factory.rating_generators:
                rating_generators = copy.deepcopy(
                    match_predictor_factory.rating_generators
                )
                rating_generators[rating_index] = rating_g
            else:
                rating_generators = [rating_g]
            match_predictor = match_predictor_factory.create(
                rating_generators=rating_generators,
            )

            return match_predictor.cross_validate_score(
                df=df,
                matches=matches,
                create_performance=False,
                cross_validator=cross_validator,
            )

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(
            direction=direction, study_name=study_name, sampler=sampler
        )
        callbacks = []
        study.optimize(
            lambda trial: objective(trial, df),
            n_trials=self.start_rating_n_trials,
            callbacks=callbacks,
        )
        start_rating_generator_params = list(
            inspect.signature(
                rating_generator.match_rating_generator.start_rating_generator.__class__.__init__
            ).parameters.keys()
        )[1:]

        other_params = {
            attr: getattr(
                rating_generator.match_rating_generator.start_rating_generator, attr
            )
            for attr in start_rating_generator_params
        }

        best_params = study.best_params
        for param in other_params:
            if param not in best_params:
                best_params[param] = other_params[param]

        return StartRatingGenerator(**best_params)
