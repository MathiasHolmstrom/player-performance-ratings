import logging
from typing import Optional

import pandas as pd

from player_performance_ratings import BaseScorer
from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings.ratings.data_prepararer import MatchGenerator
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.tuner import StartRatingTuner, PreTransformerTuner
from player_performance_ratings.tuner.rating_generator_tuner import RatingGeneratorTuner
from player_performance_ratings.tuner.team_rating_tuner import TeamRatingTuner

logging.basicConfig(level=logging.INFO)


class MatchPredictorTuner():

    def __init__(self,
                 scorer: BaseScorer,
                 pre_transformer_tuner: Optional[PreTransformerTuner] = None,
                 rating_generator_tuners: Optional[list[RatingGeneratorTuner]] = None,
                 fit_best: bool = True
                 ):
        self.scorer = scorer
        self.pre_transformer_tuner = pre_transformer_tuner
        self.rating_generator_tuners = rating_generator_tuners
        self.fit_best = fit_best

    def tune(self, df: pd.DataFrame) -> MatchPredictor:

        if self.pre_transformer_tuner:
            column_names = self.pre_transformer_tuner.column_names
            match_predictor = self.pre_transformer_tuner.match_predictor

        else:
            column_names = self.rating_generator_tuner.match_predictor.column_names
            match_predictor = self.rating_generator_tuner.match_predictor

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

        tuned_rating_generators = []
        for rating_idx, rating_generator_tuner in enumerate(self.rating_generator_tuners):
            tuned_rating_generator = rating_generator_tuner.tune(df=df, matches=matches, rating_idx=rating_idx, scorer=self.scorer)
            tuned_rating_generators.append(tuned_rating_generator)


        best_match_predictor = MatchPredictor(column_names=column_names, rating_generators=tuned_rating_generators,
                                              pre_rating_transformers=best_pre_transformers,
                                              predictor=match_predictor.predictor)
        if self.fit_best:
            best_match_predictor.generate(df=df, matches=matches)

        return best_match_predictor
