import logging
from typing import Optional

import pandas as pd

from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings.ratings.data_prepararer import MatchGenerator
from player_performance_ratings.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.tuner import StartRatingTuner, PreTransformerTuner
from player_performance_ratings.tuner.player_rating_tuner import PlayerRatingTuner

logging.basicConfig(level=logging.INFO)


class MatchPredictorTuner():

    def __init__(self,
                 pre_transformer_tuner: Optional[PreTransformerTuner] = None,
                 start_rating_tuner: Optional[StartRatingTuner] = None,
                 player_rating_tuner: Optional[PlayerRatingTuner] = None,
                 fit_best: bool = True
                 ):
        self.pre_transformer_tuner = pre_transformer_tuner
        self.start_rating_tuner = start_rating_tuner
        self.player_rating_tuner = player_rating_tuner
        self.fit_best = fit_best

    def tune(self, df: pd.DataFrame) -> MatchPredictor:
        if self.pre_transformer_tuner:
            column_names = self.pre_transformer_tuner.column_names
            match_predictor = self.pre_transformer_tuner.match_predictor

        elif self.player_rating_tuner:
            column_names = self.player_rating_tuner.match_predictor.column_names
            match_predictor = self.player_rating_tuner.match_predictor
        else:
            column_names = self.start_rating_tuner.column_names
            match_predictor = self.start_rating_tuner.match_predictor

        if self.pre_transformer_tuner:
            logging.info("Tuning PreTransformers")
            best_pre_transformers = self.pre_transformer_tuner.tune(df)
            for pre_rating_transformer in best_pre_transformers:
                df = pre_rating_transformer.transform(df)
            match_generator = MatchGenerator(column_names=column_names)
            matches = match_generator.generate(df=df)
            if self.player_rating_tuner:
                self.player_rating_tuner.match_predictor.pre_rating_transformers = best_pre_transformers
            if self.start_rating_tuner:
                self.start_rating_tuner.match_predictor.pre_rating_transformers = best_pre_transformers
        else:
            best_pre_transformers = None
            matches = None

        if self.player_rating_tuner:
            logging.info("Tuning Player Rating")
            best_player_rating_generator = self.player_rating_tuner.tune(df, matches=matches)
            if self.start_rating_tuner:
                self.start_rating_tuner.match_predictor.rating_generator.team_rating_generator.player_rating_generator = best_player_rating_generator
        else:
            best_player_rating_generator = None

        if self.start_rating_tuner:
            logging.info("Tuning Start Rating")
            best_start_rating = self.start_rating_tuner.tune(df, matches=matches)
            if best_player_rating_generator:
                best_player_rating_generator.start_rating_generator = best_start_rating

        team_rating_generator = TeamRatingGenerator(player_rating_generator=best_player_rating_generator)
        rating_generator = RatingGenerator(team_rating_generator=team_rating_generator, store_game_ratings=True,
                                           column_names=column_names)
        best_match_predictor = MatchPredictor(column_names=column_names, rating_generator=rating_generator,
                                              pre_rating_transformers=best_pre_transformers,
                                              predictor=match_predictor.predictor, target=match_predictor.target)
        if self.fit_best:
            best_match_predictor.generate(df=df, matches=matches)

        return best_match_predictor
