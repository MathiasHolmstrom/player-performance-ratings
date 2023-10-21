import copy
from typing import Optional

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.tuner.base_tuner import ParameterSearchRange, add_custom_hyperparams
from src import MatchPredictor
from src import Match
from src import PlayerRatingGenerator
from src import BaseScorer, LogLossScorer

class PlayerRatingTuner():

    def __init__(self,
                 search_ranges: Optional[list[ParameterSearchRange]],
                 match_predictor: MatchPredictor,
                 scorer: Optional[BaseScorer] = None,
                 n_trials: int = 30,
                 ):
        self.search_ranges = search_ranges
        self.match_predictor = match_predictor
        self.scorer = scorer or LogLossScorer(target=self.match_predictor.predictor.target,
                                              weight_cross_league=3,
                                              pred_column=self.match_predictor.predictor.pred_column
                                              )
        self.n_trials = n_trials

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> PlayerRatingGenerator:
        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:
            params = {}
            params = add_custom_hyperparams(params=params,
                                            trial=trial,
                                            parameter_search_range=self.search_ranges)
            player_rating_generator = PlayerRatingGenerator(**params)
            match_predictor = copy.deepcopy(self.match_predictor)
            match_predictor.rating_generator.team_rating_generator.player_rating_generator = player_rating_generator
            df = match_predictor.generate(df=df)
            return self.scorer.score(df)

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(lambda trial: objective(trial, df), n_trials=self.n_trials, callbacks=callbacks)

        best_params = study.best_params

        return PlayerRatingGenerator(**best_params)
