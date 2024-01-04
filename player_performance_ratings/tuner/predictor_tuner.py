import copy
import inspect
from typing import Optional

import optuna
import pandas as pd
import pendulum
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from player_performance_ratings.data_structures import Match
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory

from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper
from player_performance_ratings.scorer import BaseScorer

from player_performance_ratings.tuner.utils import ParameterSearchRange, add_params_from_search_range


class PredictorTuner():

    def __init__(self,
                 search_ranges: list[ParameterSearchRange],
                 date_column_name: str,
                 train_split_date: Optional[pendulum.datetime] = None,
                 n_trials: int = 30
                 ):
        self.search_ranges = search_ranges
        self.date_column_name = date_column_name
        self.train_split_date = train_split_date
        self.n_trials = n_trials

    def tune(self, df: pd.DataFrame,  scorer: BaseScorer,
             match_predictor_factory: MatchPredictorFactory) -> BaseMLWrapper:

        if not self.train_split_date:
            self.train_split_date = df.iloc[int(len(df) / 1.3)][self.date_column_name]

        def objective(trial: BaseTrial, df: pd.DataFrame) -> float:

            predictor = copy.deepcopy(match_predictor_factory.predictor)
            param_names = list(
                inspect.signature(predictor.estimator.__class__.__init__).parameters.keys())[1:]
            params = {attr: getattr(predictor.estimator, attr) for attr in param_names if attr != 'kwargs'}
            if '_other_params' in predictor.estimator.__dict__:
                params.update(predictor.estimator._other_params)
            params = add_params_from_search_range(params=params,
                                                  trial=trial,
                                                  parameter_search_range=self.search_ranges)

            predictor = copy.deepcopy(match_predictor_factory.predictor)
            for param in params:
                setattr(predictor.estimator, param, params[param])
            train_df = df[df[self.date_column_name] <= self.train_split_date]
            predict_df = df[df[self.date_column_name] > self.train_split_date]

            predictor.train(train_df)
            df_with_prediction = predictor.add_prediction(predict_df)
            return scorer.score(df_with_prediction, classes_=match_predictor_factory.predictor.classes_)

        direction = "minimize"
        study_name = "optuna_study"
        optuna_seed = 12
        sampler = TPESampler(seed=optuna_seed)
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
        callbacks = []
        study.optimize(lambda trial: objective(trial, df), n_trials=self.n_trials, callbacks=callbacks)
        best_model_params = study.best_params
        other_predictor_params = list(
            inspect.signature(match_predictor_factory.predictor.__class__.__init__).parameters.keys())[1:]

        if '_other_params' in match_predictor_factory.predictor.estimator.__dict__:
            best_model_params.update(match_predictor_factory.predictor.estimator._other_params)

        other_predictor_params = {attr: getattr(match_predictor_factory.predictor, attr) for attr in
                                  other_predictor_params if attr not in ('estimator')}

        predictor_class = match_predictor_factory.predictor.__class__
        model_class = match_predictor_factory.predictor.estimator.__class__
        return predictor_class(estimator=model_class(**best_model_params), **other_predictor_params)
