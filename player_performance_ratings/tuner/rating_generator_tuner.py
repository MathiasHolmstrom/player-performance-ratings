import copy
import inspect
import logging
from abc import abstractmethod, ABC
from typing import Optional, Match

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial
from player_performance_ratings.ratings import RatingColumnNames

from player_performance_ratings.ratings.opponent_adjusted_rating import TeamRatingGenerator

from player_performance_ratings.ratings.opponent_adjusted_rating.start_rating_generator import StartRatingGenerator
from player_performance_ratings.ratings.opponent_adjusted_rating import \
    OpponentAdjustedRatingGenerator
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.scorer import BaseScorer

from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.utils import ParameterSearchRange, add_params_from_search_range

DEFAULT_TEAM_SEARCH_RANGES = [
    ParameterSearchRange(
        name='confidence_weight',
        type='uniform',
        low=0.6,
        high=0.95
    ),
    ParameterSearchRange(
        name='confidence_days_ago_multiplier',
        type='uniform',
        low=0.02,
        high=.12,
    ),
    ParameterSearchRange(
        name='confidence_max_days',
        type='uniform',
        low=40,
        high=200,
    ),
    ParameterSearchRange(
        name='confidence_max_sum',
        type='uniform',
        low=20,
        high=180,
    ),
    ParameterSearchRange(
        name='confidence_value_denom',
        type='uniform',
        low=20,
        high=180,
    ),
    ParameterSearchRange(
        name='rating_change_multiplier',
        type='uniform',
        low=25,
        high=140
    ),
    ParameterSearchRange(
        name='min_rating_change_multiplier_ratio',
        type='uniform',
        low=0.05,
        high=0.2,
    ),
]

DEFAULT_START_RATING_SEARCH_RANGE = [
    ParameterSearchRange(
        name='league_quantile',
        type='uniform',
        low=0.12,
        high=.4,
    ),
    ParameterSearchRange(
        name='min_count_for_percentiles',
        type='int',
        low=50,
        high=200,
    ),
    ParameterSearchRange(
        name='team_rating_subtract',
        type='int',
        low=20,
        high=300
    ),
    ParameterSearchRange(
        name='team_weight',
        type='uniform',
        low=0,
        high=0.7
    )
]


class RatingGeneratorTuner(ABC):

    @abstractmethod
    def tune(self, df: pd.DataFrame, rating_idx: int, scorer: BaseScorer,
             match_predictor_factory: MatchPredictorFactory,
             matches: list[Match]) -> RatingGenerator:
        pass


class OpponentAdjustedRatingGeneratorTuner(RatingGeneratorTuner):

    def __init__(self,
                 team_rating_search_ranges: Optional[list[ParameterSearchRange]] = None,
                 team_rating_n_trials: int = 30,
                 start_rating_search_ranges: Optional[list[ParameterSearchRange]] = None,
                 start_rating_n_trials: int = 8,
                 ):
        self.team_rating_search_ranges = team_rating_search_ranges or DEFAULT_TEAM_SEARCH_RANGES
        self.start_rating_search_ranges = start_rating_search_ranges or DEFAULT_START_RATING_SEARCH_RANGE
        self.team_rating_n_trials = team_rating_n_trials
        self.start_rating_n_trials = start_rating_n_trials

    def tune(self,
             df: pd.DataFrame,
             rating_idx: int,
             scorer: BaseScorer,
             match_predictor_factory: MatchPredictorFactory,
             matches: list[Match]) -> OpponentAdjustedRatingGenerator:

        if match_predictor_factory.rating_generators:
            best_rating_generator = copy.deepcopy(match_predictor_factory.rating_generators[rating_idx])
        else:
            potential_rating_features = [v for k, v in RatingColumnNames.__dict__.items() if isinstance(v, str)]

            best_rating_generator = OpponentAdjustedRatingGenerator(
                features_out=[f for f in match_predictor_factory.predictor.features if f in potential_rating_features],
                column_names=match_predictor_factory.rating_generators[rating_idx].column_names
            )

        if self.team_rating_n_trials > 0:
            logging.info("Tuning Team Rating")
            best_team_rating_generator = self._tune_team_rating(df=df,
                                                                rating_generator=best_rating_generator,
                                                                rating_index=rating_idx,
                                                                scorer=scorer,
                                                                matches=matches,
                                                                match_predictor_factory=match_predictor_factory)

            best_rating_generator.team_rating_generator = best_team_rating_generator

        if self.start_rating_n_trials > 0:
            logging.info("Tuning Start Rating")

            best_start_rating = self._tune_start_rating(df=df,
                                                        matches=matches,
                                                        rating_generator=best_rating_generator,
                                                        rating_index=rating_idx, scorer=scorer,
                                                        match_predictor_factory=match_predictor_factory
                                                        )
            best_rating_generator.team_rating_generator.start_rating_generator = best_start_rating

        return OpponentAdjustedRatingGenerator(team_rating_generator=best_rating_generator.team_rating_generator,
                                               column_names=best_rating_generator.column_names)

    def _tune_team_rating(self,
                          df: pd.DataFrame,
                          rating_generator: OpponentAdjustedRatingGenerator,
                          rating_index: int,
                          matches: list[Match],
                          scorer: BaseScorer,
                          match_predictor_factory: MatchPredictorFactory,
                          ) -> TeamRatingGenerator:

        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:

            team_rating_generator_params = list(
                inspect.signature(rating_generator.team_rating_generator.__class__.__init__).parameters.keys())[1:]

            params = {attr: getattr(rating_generator.team_rating_generator, attr) for attr in
                      team_rating_generator_params if attr not in ('performance_predictor', 'start_rating_generator')}
            params = add_params_from_search_range(params=params,
                                                  trial=trial,
                                                  parameter_search_range=self.team_rating_search_ranges)

            performance_predictor = rating_generator.team_rating_generator.performance_predictor
            performance_predictor_params = list(
                inspect.signature(performance_predictor.__class__.__init__).parameters.keys())[1:]

            for param in params.copy():
                if param in performance_predictor_params:
                    performance_predictor.__setattr__(param, params[param])
                    params.pop(param)

            team_rating_generator = TeamRatingGenerator(**params,
                                                        performance_predictor=performance_predictor)

            rating_g = copy.deepcopy(rating_generator)
            rating_g.team_rating_generator = team_rating_generator
            rating_generators = copy.deepcopy(match_predictor_factory.rating_generators)

            rating_generators[rating_index] = rating_g
            match_predictor = match_predictor_factory.create(
                rating_generators=rating_generators,
            )

            df_with_prediction = match_predictor.generate_historical(df=df, matches=matches, store_ratings=False)
            test_df = df_with_prediction[df_with_prediction[match_predictor.date_column_name] > match_predictor.train_split_date]
            return scorer.score(test_df, classes_=match_predictor.predictor.classes_)

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(lambda trial: objective(trial, df), n_trials=self.team_rating_n_trials, callbacks=callbacks)

        best_params = study.best_params
        team_rating_generator_params = list(
            inspect.signature(rating_generator.team_rating_generator.__class__.__init__).parameters.keys())[1:]
        other_params = {attr: getattr(rating_generator.team_rating_generator, attr) for attr in
                        team_rating_generator_params
                        if attr not in ('performance_predictor', 'start_rating_generator')}
        for param in other_params:
            if param not in best_params:
                best_params[param] = other_params[param]

        performance_predictor = rating_generator.team_rating_generator.performance_predictor
        performance_predictor_params = list(
            inspect.signature(performance_predictor.__class__.__init__).parameters.keys())[1:]

        for param in best_params.copy():
            if param in performance_predictor_params:
                performance_predictor.__setattr__(param, best_params[param])
                best_params.pop(param)

        return TeamRatingGenerator(**best_params,
                                   performance_predictor=performance_predictor,
                                   start_rating_generator=rating_generator.team_rating_generator.start_rating_generator
                                   )

    def _tune_start_rating(self,
                           df: pd.DataFrame,
                           rating_generator: OpponentAdjustedRatingGenerator,
                           rating_index: int,
                           matches: list[Match],
                           scorer: BaseScorer,
                           match_predictor_factory: MatchPredictorFactory):
        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:
            start_rating_generator_params = list(
                inspect.signature(
                    rating_generator.team_rating_generator.start_rating_generator.__class__.__init__).parameters.keys())[
                                            1:]

            params = {attr: getattr(rating_generator.team_rating_generator.start_rating_generator, attr) for attr in
                      start_rating_generator_params}

            params = add_params_from_search_range(params=params,
                                                  trial=trial,
                                                  parameter_search_range=self.start_rating_search_ranges)

            start_rating_generator = StartRatingGenerator(**params)
            rating_g = copy.deepcopy(rating_generator)
            rating_g.team_rating_generator.start_rating_generator = start_rating_generator
            if match_predictor_factory.rating_generators:
                rating_generators = copy.deepcopy(match_predictor_factory.rating_generators)
                rating_generators[rating_index] = rating_g
            else:
                rating_generators = [rating_g]
            match_predictor = match_predictor_factory.create(
                rating_generators=rating_generators,

            )

            df_with_prediction = match_predictor.generate_historical(df=df, matches=matches, store_ratings=False)
            test_df = df_with_prediction[
                df_with_prediction[match_predictor.date_column_name] > match_predictor.train_split_date]
            return scorer.score(test_df, classes_=match_predictor.predictor.classes_)

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(lambda trial: objective(trial, df), n_trials=self.start_rating_n_trials, callbacks=callbacks)
        start_rating_generator_params = list(
            inspect.signature(
                rating_generator.team_rating_generator.start_rating_generator.__class__.__init__).parameters.keys())[1:]

        other_params = {attr: getattr(rating_generator.team_rating_generator.start_rating_generator, attr) for attr in
                        start_rating_generator_params}

        best_params = study.best_params
        for param in other_params:
            if param not in best_params:
                best_params[param] = other_params[param]

        return StartRatingGenerator(**best_params)
