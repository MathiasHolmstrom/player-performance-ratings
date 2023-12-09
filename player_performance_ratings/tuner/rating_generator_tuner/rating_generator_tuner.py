import copy
import logging
from abc import abstractmethod, ABC
from typing import Optional, Match

import pandas as pd

from player_performance_ratings import RatingGenerator, BaseScorer, \
    BaseTransformer, ColumnNames
from player_performance_ratings.ratings.rating_generator import OpponentAdjustedRatingGenerator
from player_performance_ratings.tuner.rating_generator_tuner import TeamRatingTuner, StartRatingTuner
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory


class RatingGeneratorTuner(ABC):

    @abstractmethod
    def tune(self, df: pd.DataFrame, rating_idx: int, scorer: BaseScorer,

             match_predictor_factory: MatchPredictorFactory,
             matches: list[Match]) -> RatingGenerator:
        pass

    @property
    @abstractmethod
    def column_names(self) -> ColumnNames:
        pass


class OpponentAdjustedRatingGeneratorTuner(RatingGeneratorTuner):

    def __init__(self,
                 rating_generator: OpponentAdjustedRatingGenerator,
                 team_rating_tuner: Optional[TeamRatingTuner],
                 start_rating_tuner: Optional[StartRatingTuner],
                 ):
        self.rating_generator = rating_generator
        self.team_rating_tuner = team_rating_tuner
        self.start_rating_tuner = start_rating_tuner

    def tune(self, df: pd.DataFrame,
             rating_idx: int,
             scorer: BaseScorer,
             match_predictor_factory: MatchPredictorFactory,
             matches: list[Match]) -> RatingGenerator:


        best_rating_generator = copy.deepcopy(self.rating_generator)
        if self.team_rating_tuner:

            logging.info("Tuning Team Rating")
            best_team_rating_generator = self.team_rating_tuner.tune(df=df,
                                                                     rating_generator=best_rating_generator,
                                                                     rating_index=rating_idx,
                                                                     scorer=scorer,
                                                                     matches=matches,
                                                                     match_predictor_factory=match_predictor_factory)

            best_rating_generator.team_rating_generator = best_team_rating_generator
        else:
            best_rating_generator = self.rating_generator

        if self.start_rating_tuner:
            logging.info("Tuning Start Rating")

            best_start_rating = self.start_rating_tuner.tune(df, matches=matches,
                                                             rating_generator=best_rating_generator,
                                                             rating_index=rating_idx, scorer=scorer,
                                                             match_predictor_factory=match_predictor_factory)
            best_rating_generator.team_rating_generator.start_rating_generator = best_start_rating

        return OpponentAdjustedRatingGenerator(team_rating_generator=best_rating_generator.team_rating_generator)
