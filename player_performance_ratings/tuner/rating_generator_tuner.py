import logging
from abc import abstractmethod, ABC
from typing import Optional, Match

import pandas as pd

from player_performance_ratings import RatingGenerator, TeamRatingTuner, StartRatingTuner, MatchPredictor, BaseScorer
from player_performance_ratings.ratings.data_prepararer import MatchGenerator
from player_performance_ratings.ratings.rating_generator import OpponentAdjustedRatingGenerator


class RatingGeneratorTuner(ABC):

    @abstractmethod
    def tune(self, df: pd.DataFrame, rating_idx: int, scorer: BaseScorer,
             matches: Optional[list[Match]] = None) -> RatingGenerator:
        pass


class OpponentAdjustedRatingGeneratorTuner():

    def __init__(self,
                 rating_generator: OpponentAdjustedRatingGenerator,
                 team_rating_tuner: Optional[TeamRatingTuner],
                 start_rating_tuner: Optional[StartRatingTuner],
                 ):
        self.rating_generator = rating_generator
        self.team_rating_tuner = team_rating_tuner
        self.start_rating_tuner = start_rating_tuner



    def tune(self, df: pd.DataFrame, rating_idx: int, scorer: BaseScorer,matches: Optional[list[Match]] = None) -> RatingGenerator:
        column_names = self.rating_generator.column_names


        if self.team_rating_tuner:

            if matches is None:
                for pre_rating_transformer in self.team_rating_tuner.match_predictor.pre_rating_transformers:
                    df = pre_rating_transformer.transform(df)
                match_generator = MatchGenerator(column_names=column_names)
                matches = match_generator.generate(df=df)

            logging.info("Tuning Team Rating")
            best_team_rating_generator = self.team_rating_tuner.tune(df, matches=matches)
            if self.start_rating_tuner:

                self.start_rating_tuner.match_predictor.rating_generators.team_rating_generator = best_team_rating_generator
        else:
            best_team_rating_generator = None

        if self.start_rating_tuner:
            logging.info("Tuning Start Rating")

            if matches is None:
                for pre_rating_transformer in self.start_rating_tuner.match_predictor.pre_rating_transformers:
                    df = pre_rating_transformer.transform(df)
                match_generator = MatchGenerator(column_names=column_names)
                matches = match_generator.generate(df=df)

            best_start_rating = self.start_rating_tuner.tune(df, matches=matches, rating_generator=self.rating_generator,rating_index=rating_idx, scorer=scorer)
            if best_team_rating_generator:
                best_team_rating_generator.start_rating_generator = best_start_rating

        team_rating_generator = best_team_rating_generator

        return OpponentAdjustedRatingGenerator(team_rating_generator=team_rating_generator, store_game_ratings=True,
                                               column_names=column_names)
