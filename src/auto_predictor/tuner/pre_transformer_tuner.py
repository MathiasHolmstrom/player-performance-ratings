from dataclasses import dataclass
from typing import Optional, Match, Tuple, Literal, Union, Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from src.auto_predictor.tuner.base_tuner import BaseTuner
from src.predictor.match_predictor import MatchPredictor
from src.ratings.data_prepararer import MatchGenerator
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.factory.match_generator_factory import RatingGeneratorFactory
from src.ratings.league_identifier import LeagueIdentifier
from src.ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator
from src.scorer.base_score import BaseScorer, LogLossScorer
from src.transformers import BaseTransformer

RC = RatingColumnNames


@dataclass
class ParameterSearchRange:
    name: str
    type: Literal["uniform", "loguniform", "int", "categorical"]
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[list[Any]] = None


def add_custom_hyperparams(params: dict[str, Union[float, None, bool, int, str]], trial: BaseTrial,
                           parameter_search_range: list[ParameterSearchRange]) -> dict[
    str, Union[float, None, bool, int, str]]:
    for config in parameter_search_range:
        if config.type == "uniform":
            params[config.name] = trial.suggest_uniform(config.name, low=config.low, high=config.high)
        elif config.type == "loguniform":
            params[config.name] = trial.suggest_loguniform(config.name, low=config.low, high=config.high)
        elif config.type == "int":
            params[config.name] = trial.suggest_int(config.name, low=config.low, high=config.high)
        elif config.type == "categorical":
            params[config.name] = trial.suggest_categorical(config.name, config.choices)

    return params


class PreTransformerTuner(BaseTuner):

    def __init__(self,
                 pre_transformer_search_ranges: list[Tuple[BaseTransformer, list[ParameterSearchRange]]],
                 column_names: ColumnNames,
                 n_trials: int = 30,
                 scorer: Optional[BaseScorer] = None,
                 match_predictor: Optional[MatchPredictor] = None,
                 start_rating_generator: Optional[StartRatingGenerator] = None,
                 player_rating_generator: Optional[PlayerRatingGenerator] = None,
                 performance_predictor: Optional[PerformancePredictor] = None,
                 team_rating_generator: Optional[TeamRatingGenerator] = None,
                 league_identifier: Optional[LeagueIdentifier] = None,
                 ):
        self.pre_transformer_search_ranges = pre_transformer_search_ranges
        rating_column_names = [RC.rating_difference]

        self.n_trials = n_trials
        self.match_predictor = match_predictor or MatchPredictor(column_names=column_names,
                                                                 rating_features=rating_column_names)
        self.scorer = scorer or LogLossScorer(target=self.match_predictor.predictor.target, weight_cross_league=3,
                                              prob_column_name=self.match_predictor.predictor.prob_column_name)

        self.start_rating_generator = start_rating_generator or StartRatingGenerator()
        self.player_rating_generator = player_rating_generator or PlayerRatingGenerator()
        self.performance_predictor = performance_predictor or PerformancePredictor()
        self.team_rating_generator = team_rating_generator or TeamRatingGenerator()
        self.league_identifier = league_identifier or LeagueIdentifier()
        self.column_names = column_names
        self._scores = []

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> dict[str, float]:
        rating_generator_factory = RatingGeneratorFactory(
            start_rating_generator=self.start_rating_generator,
            team_rating_generator=self.team_rating_generator,
            player_rating_generator=self.player_rating_generator,
            performance_predictor=self.performance_predictor,
        )

        rating_generator = rating_generator_factory.create()
        self.match_predictor.rating_generator = rating_generator

        def objective(trial: BaseTrial, mlflow_logging=False) -> float:
            params = {}
            for transformer, parameter_search_range in self.pre_transformer_search_ranges:
                params = add_custom_hyperparams(params=params, trial=trial,
                                                parameter_search_range=parameter_search_range)

            match_generator = MatchGenerator(column_names=self.column_names)

            matches = match_generator.generate(df=df)

            return cross_validator.score()

            direction = "minimize"
            study_name = "optuna_study"
            optuna_seed = 42
            sampler = TPESampler(seed=optuna_seed)
            study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
            callbacks = []
            study.optimize(objective, n_trials=self.n_trials, callbacks=callbacks)
