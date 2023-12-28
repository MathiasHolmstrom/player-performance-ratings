import copy
import logging
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.scorer import BaseScorer

from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner
from player_performance_ratings.tuner.rating_generator_tuner import RatingGeneratorTuner
from player_performance_ratings.tuner.transformer_tuner import TransformerTuner

logging.basicConfig(level=logging.INFO)


class MatchPredictorTuner():
    """
    Given a search range, it will identify the best combination of pre_transformers, rating_generators and post_transformers.
    Using optuna, it will tune the hyperparameters of each transformer.
    """

    def __init__(self,
                 scorer: BaseScorer,
                 match_predictor_factory: MatchPredictorFactory,
                 pre_transformer_tuner: Optional[TransformerTuner] = None,
                 rating_generator_tuners: Optional[Union[list[RatingGeneratorTuner], RatingGeneratorTuner]] = None,
                 post_transformer_tuner: Optional[TransformerTuner] = None,
                 predictor_tuner: Optional[PredictorTuner] = None,
                 fit_best: bool = True
                 ):

        """
        :param scorer:
            The class object that scores the predictions
        :param match_predictor_factory:
            The factory that creates the MatchPredictor.
            Contains the parameters to create the MatchPredictor if no parameter-tuning was done.
            Based on hyperparameter-tuning the final way the match-predictor is generated will be based on a combination of the parameters in the factory and the tuned parameters.
        :param pre_transformer_tuner:
            The tuner that tunes the hyperparameters of the pre_transformers.
            If left none or as [], the pre_transformers in the match_predictor_factory will be used.
        :param rating_generator_tuners:
            The tuner that tunes the hyperparameters of the rating_generators.
            If left none or as [], the rating_generators in the match_predictor_factory will be used.
        :param post_transformer_tuner:
            The tuner that tunes the hyperparameters of the post_transformers.
            If left none or as [], the post_transformers in the match_predictor_factory will be used.
        :param predictor_tuner:
            The tuner that tunes the hyperparameters of the predictor model.

        :param fit_best: Whether to refit the match_predictor with the entire training use data using best hyperparameters
        """

        self.scorer = scorer
        self.match_predictor_factory = match_predictor_factory
        self.pre_transformer_tuner = pre_transformer_tuner or []
        self.rating_generator_tuners = rating_generator_tuners or []
        self.post_transformer_tuner = post_transformer_tuner or []
        self.predictor_tuner = predictor_tuner
        if isinstance(self.rating_generator_tuners, RatingGeneratorTuner):
            self.rating_generator_tuners = [self.rating_generator_tuners]
        self.fit_best = fit_best

        if not self.pre_transformer_tuner and not self.rating_generator_tuners and not self.post_transformer_tuner:
            raise ValueError("No tuning has been provided in config")

    def tune(self, df: pd.DataFrame) -> MatchPredictor:

        column_names = self.match_predictor_factory.column_names

        best_pre_transformers = copy.deepcopy(self.match_predictor_factory.pre_transformers)


        if self.pre_transformer_tuner:
            logging.info("Tuning PreTransformers")
            best_pre_transformers = self.pre_transformer_tuner.tune(df=df,
                                                                    match_predictor_factory=self.match_predictor_factory,
                                                                    scorer=self.scorer)
        for pre_rating_transformer in best_pre_transformers:
            df = pre_rating_transformer.fit_transform(df)

        best_rating_generators = copy.deepcopy(self.match_predictor_factory.rating_generators)

        matches = []
        for col_name in column_names:
            rating_matches = convert_df_to_matches(df=df, column_names=col_name)
            matches.append(rating_matches)

        for rating_idx, rating_generator_tuner in enumerate(self.rating_generator_tuners):
            rating_matches = convert_df_to_matches(df=df, column_names=column_names[rating_idx])
            matches.append(rating_matches)
            tuned_rating_generator = rating_generator_tuner.tune(df=df, matches=matches[rating_idx], rating_idx=rating_idx,
                                                                 scorer=self.scorer,
                                                                 match_predictor_factory=self.match_predictor_factory)
            if best_rating_generators:
                best_rating_generators[rating_idx] = tuned_rating_generator
            else:
                best_rating_generators = [tuned_rating_generator]

        best_post_transformers = copy.deepcopy(self.match_predictor_factory.post_transformers)
        if self.post_transformer_tuner:
            logging.info("Tuning PostTransformers")
            best_post_transformers = self.post_transformer_tuner.tune(df,
                                                                      match_predictor_factory=self.match_predictor_factory,
                                                                      scorer=self.scorer,
                                                                      pre_rating_transformers=best_pre_transformers,
                                                                      rating_generators=best_rating_generators,
                                                                      matches=matches
                                                                      )
            for post_rating_transformer in best_post_transformers:
                df = post_rating_transformer.fit_transform(df)

        if self.predictor_tuner:
            logging.info("Tuning Predictor")
            best_predictor = self.predictor_tuner.tune(df=df, matches=matches, scorer=self.scorer,
                                                       match_predictor_factory=self.match_predictor_factory)
        else:
            best_predictor = self.match_predictor_factory.predictor

        best_match_predictor = MatchPredictor(column_names=column_names,
                                              rating_generators=best_rating_generators,
                                              pre_rating_transformers=best_pre_transformers,
                                              post_rating_transformers=best_post_transformers,
                                              predictor=best_predictor)
        if self.fit_best:
            best_match_predictor.generate_historical(df=df, matches=matches, store_ratings=True)

        return best_match_predictor
