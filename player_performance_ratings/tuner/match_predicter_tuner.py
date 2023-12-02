import logging
from typing import Optional

import pandas as pd

from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings.ratings.data_prepararer import MatchGenerator
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.tuner import StartRatingTuner, PreTransformerTuner
from player_performance_ratings.tuner.team_rating_tuner import TeamRatingTuner

logging.basicConfig(level=logging.INFO)


class MatchPredictorTuner():

    def __init__(self,
                 pre_transformer_tuner: Optional[PreTransformerTuner] = None,
                 start_rating_tuner: Optional[StartRatingTuner] = None,
                 team_rating_tuner: Optional[TeamRatingTuner] = None,
                 fit_best: bool = True
                 ):
        self.pre_transformer_tuner = pre_transformer_tuner
        self.start_rating_tuner = start_rating_tuner
        self.team_rating_tuner = team_rating_tuner
        self.fit_best = fit_best

    def tune(self, df: pd.DataFrame) -> MatchPredictor:

        if self.pre_transformer_tuner:
            column_names = self.pre_transformer_tuner.column_names
            match_predictor = self.pre_transformer_tuner.match_predictor

        elif self.team_rating_tuner:
            column_names = self.team_rating_tuner.match_predictor.column_names
            match_predictor = self.team_rating_tuner.match_predictor
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
            if self.team_rating_tuner:
                self.team_rating_tuner.match_predictor.pre_rating_transformers = best_pre_transformers
            if self.start_rating_tuner:
                self.start_rating_tuner.match_predictor.pre_rating_transformers = best_pre_transformers
        else:
            best_pre_transformers = None
            matches = None

        if self.team_rating_tuner:

            if matches is None:
                for pre_rating_transformer in self.team_rating_tuner.match_predictor.pre_rating_transformers:
                    df = pre_rating_transformer.transform(df)
                match_generator = MatchGenerator(column_names=column_names)
                matches = match_generator.generate(df=df)

            logging.info("Tuning Team Rating")
            best_team_rating_generator = self.team_rating_tuner.tune(df, matches=matches)
            if self.start_rating_tuner:
                self.start_rating_tuner.match_predictor.rating_generator.team_rating_generator = best_team_rating_generator
        else:
            best_team_rating_generator = None

        if self.start_rating_tuner:
            logging.info("Tuning Start Rating")

            if matches is None:
                for pre_rating_transformer in self.start_rating_tuner.match_predictor.pre_rating_transformers:
                    df = pre_rating_transformer.transform(df)
                match_generator = MatchGenerator(column_names=column_names)
                matches = match_generator.generate(df=df)

            best_start_rating = self.start_rating_tuner.tune(df, matches=matches)
            if best_team_rating_generator:
                best_team_rating_generator.start_rating_generator = best_start_rating

        team_rating_generator = best_team_rating_generator
        rating_generator = RatingGenerator(team_rating_generator=team_rating_generator, store_game_ratings=True,
                                           column_names=column_names)
        best_match_predictor = MatchPredictor(column_names=column_names, rating_generator=rating_generator,
                                              pre_rating_transformers=best_pre_transformers,
                                              predictor=match_predictor.predictor)
        if self.fit_best:
            best_match_predictor.generate(df=df, matches=matches)

        return best_match_predictor
