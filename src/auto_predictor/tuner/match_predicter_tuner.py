import copy
import inspect
from typing import Optional, Match, Tuple

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from src.auto_predictor.tuner import StartRatingTuner
from src.auto_predictor.tuner.base_tuner import add_custom_hyperparams, ParameterSearchRange
from src.auto_predictor.tuner.player_rating_tuner import PlayerRatingTuner
from src.auto_predictor.tuner.pre_transformer_tuner import add_hyperparams_to_common_transformers, PreTransformerTuner
from src.predictor.match_predictor import MatchPredictor
from src.ratings.data_prepararer import MatchGenerator
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.scorer.base_score import LogLossScorer, BaseScorer
from src.transformers import BaseTransformer


class MatchPredictorTuner():

    def __init__(self,
                 pre_transformer_tuner: Optional[PreTransformerTuner] = None,
                 start_rating_tuner: Optional[StartRatingTuner] = None,
                 player_rating_tuner: Optional[PlayerRatingTuner] = None,
                 ):
        self.pre_transformer_tuner = pre_transformer_tuner
        self.start_rating_tuner = start_rating_tuner
        self.player_rating_tuner = player_rating_tuner

    def tune(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> MatchPredictor:
        if self.pre_transformer_tuner:
            column_names = self.pre_transformer_tuner.column_names

        elif self.player_rating_tuner:
            column_names = self.player_rating_tuner.match_predictor.column_names
        else:
            column_names = self.start_rating_tuner.column_names

        if self.pre_transformer_tuner:
            best_pre_transformers = self.pre_transformer_tuner.tune(df)
            if self.player_rating_tuner:
                self.player_rating_tuner.match_predictor.pre_rating_transformers = best_pre_transformers
            if self.start_rating_tuner:
                self.start_rating_tuner.match_predictor.pre_rating_transformers = best_pre_transformers
        else:
            best_pre_transformers = None

        if self.player_rating_tuner:
            best_player_rating_generator = self.player_rating_tuner.tune(df)
            if self.start_rating_tuner:
                self.start_rating_tuner.match_predictor.rating_generator.team_rating_generator.player_rating_generator = best_player_rating_generator
        else:
            best_player_rating_generator = None

        if self.start_rating_tuner:
            best_start_rating = self.start_rating_tuner.tune(df)
            if best_player_rating_generator:
                best_player_rating_generator.start_rating_generator = best_start_rating

        team_rating_generator = TeamRatingGenerator(player_rating_generator=best_player_rating_generator)
        rating_generator = RatingGenerator(team_rating_generator=team_rating_generator)
        return MatchPredictor(column_names=column_names, rating_generator=rating_generator,
                              pre_rating_transformers=best_pre_transformers)
