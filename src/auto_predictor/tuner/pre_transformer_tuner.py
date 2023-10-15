import copy
import inspect
import logging
from dataclasses import dataclass
from typing import Optional, Match, Tuple, Literal, Union, Any

import optuna
import pandas as pd
import pendulum
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from src.auto_predictor.tuner.base_tuner import BaseTuner
from src.predictor.match_predictor import MatchPredictor
from src.predictor.ml_wrappers.base_wrapper import BaseMLWrapper
from src.ratings.data_prepararer import MatchGenerator
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.factory.match_generator_factory import RatingGeneratorFactory
from src.ratings.league_identifier import LeagueIdentifier
from src.ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator
from src.scorer.base_score import BaseScorer, LogLossScorer
from src.transformers import BaseTransformer
from src.transformers.common import ColumnWeight

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


def add_hyperparams_to_common_transformers(object: object, params: dict[str, Union[float, None, bool, int, str]],
                                           trial: BaseTrial,
                                           parameter_search_range: list[ParameterSearchRange]) -> dict[
    str, Union[float, None, bool, int, str]]:
    params = add_custom_hyperparams(params=params, trial=trial, parameter_search_range=parameter_search_range)

    if object.__class__.__name__ == "ColumnsWeighter":
        column_weights = [ColumnWeight(name=p.name, weight=params[p.name]) for p in parameter_search_range]
        for p in parameter_search_range:
            del params[p.name]
        params['column_weights'] = column_weights

    return params


class PreTransformerTuner(BaseTuner):

    def __init__(self,
                 pre_transformer_search_ranges: list[Tuple[BaseTransformer, list[ParameterSearchRange]]],
                 column_names: ColumnNames,
                 rating_generator: RatingGenerator,
                 predictor: BaseMLWrapper,
                 n_trials: int = 30,
                 scorer: Optional[BaseScorer] = None,
                 train_split_date: Optional[pendulum.datetime] = None,

                 ):
        self.pre_transformer_search_ranges = pre_transformer_search_ranges
        self.predictor = predictor
        self.train_split_date = train_split_date
        self.rating_generator = rating_generator

        self.n_trials = n_trials

        self.scorer = scorer or LogLossScorer(target=self.predictor.target, weight_cross_league=3,
                                              pred_column=self.predictor.pred_column)


        self.column_names = column_names
        self._scores = []

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> dict[str, float]:

        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:

            rating_generator = copy.deepcopy(self.rating_generator)
            pre_rating_transformers = []
            for transformer, parameter_search_range in self.pre_transformer_search_ranges:
                transformer_params = list(
                    inspect.signature(transformer.__class__.__init__).parameters.keys())[1:]
                params = {attr: getattr(transformer, attr) for attr in
                          dir(transformer) if
                          attr in transformer_params}
                params = add_hyperparams_to_common_transformers(object=transformer, params=params, trial=trial,
                                                                parameter_search_range=parameter_search_range)
                class_transformer = type(transformer)
                pre_rating_transformers.append(class_transformer(**params))
                predictor = copy.deepcopy(self.predictor)

                match_predictor = MatchPredictor(column_names=self.column_names,
                                                 rating_generator=rating_generator,
                                                 pre_rating_transformers=pre_rating_transformers,
                                                 predictor=predictor,
                                                 rating_features=[RC.rating_difference],
                                                 train_split_date=self.train_split_date,
                                                 )
            df = match_predictor.generate(df=df)
            return self.scorer.score(df)


        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 42
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(lambda trial: objective(trial, df), n_trials=self.n_trials, callbacks=callbacks)
