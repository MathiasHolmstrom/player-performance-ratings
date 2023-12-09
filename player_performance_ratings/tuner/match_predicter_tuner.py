import logging
from typing import Optional, Union

import pandas as pd

from player_performance_ratings import BaseScorer
from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings.ratings.match_generator import convert_df_to_matches

from player_performance_ratings.tuner import PreTransformerTuner
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.rating_generator_tuner.rating_generator_tuner import RatingGeneratorTuner

logging.basicConfig(level=logging.INFO)


class MatchPredictorTuner():

    def __init__(self,
                 scorer: BaseScorer,
                 match_predictor_factory: MatchPredictorFactory,
                 pre_transformer_tuner: Optional[PreTransformerTuner] = None,
                 rating_generator_tuners: Optional[Union[list[RatingGeneratorTuner], RatingGeneratorTuner]] = None,
                 fit_best: bool = True
                 ):
        self.scorer = scorer
        self.match_predictor_factory = match_predictor_factory
        self.pre_transformer_tuner = pre_transformer_tuner
        self.rating_generator_tuners = rating_generator_tuners
        if isinstance(self.rating_generator_tuners, RatingGeneratorTuner):
            self.rating_generator_tuners = [self.rating_generator_tuners]
        self.fit_best = fit_best

        if not self.pre_transformer_tuner and not self.rating_generator_tuners:
            raise ValueError("If no pre_transformer_tuner is provided, rating_generator_tuners must be provided")


    def tune(self, df: pd.DataFrame) -> MatchPredictor:

        column_names = self.pre_transformer_tuner.column_names if self.pre_transformer_tuner else \
        self.rating_generator_tuners[0].column_names

        if self.pre_transformer_tuner:
            logging.info("Tuning PreTransformers")
            best_pre_transformers = self.pre_transformer_tuner.tune(df)
            for pre_rating_transformer in best_pre_transformers:
                df = pre_rating_transformer.transform(df)
            matches = convert_df_to_matches(df=df, column_names=column_names)

        else:
            best_pre_transformers = self.match_predictor_factory.pre_transformers
            matches = None

        tuned_rating_generators = []
        for rating_idx, rating_generator_tuner in enumerate(self.rating_generator_tuners):
            tuned_rating_generator = rating_generator_tuner.tune(df=df, matches=matches, rating_idx=rating_idx,
                                                                 scorer=self.scorer,
                                                                 match_predictor_factory=self.match_predictor_factory)
            tuned_rating_generators.append(tuned_rating_generator)

        best_match_predictor = MatchPredictor(column_names=column_names,
                                              rating_generators=tuned_rating_generators,
                                              pre_rating_transformers=best_pre_transformers,
                                              predictor=self.match_predictor_factory.predictor)
        if self.fit_best:
            best_match_predictor.generate_historical(df=df, store_ratings=True)

        return best_match_predictor
