import copy
import logging
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings.ratings import PerformancesGenerator
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.scorer import BaseScorer

from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner
from player_performance_ratings.tuner.rating_generator_tuner import RatingGeneratorTuner
from player_performance_ratings.tuner.performances_generator_tuner import PerformancesGeneratorTuner

logging.basicConfig(level=logging.INFO)


class MatchPredictorTuner():
    """
    Given a search range, it will identify the best combination of pre_transformers, rating_generators and post_transformers.
    Using optuna, it will tune the hyperparameters of each transformer.
    """

    def __init__(self,
                 scorer: BaseScorer,
                 match_predictor_factory: MatchPredictorFactory,
                 performances_generator_tuner: Optional[PerformancesGeneratorTuner] = None,
                 rating_generator_tuners: Optional[Union[list[RatingGeneratorTuner], RatingGeneratorTuner]] = None,
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
        :param performances_generator_tuner:
            The tuner that tunes the hyperparameters of the pre_transformers.
            If left none or as [], the pre_transformers in the match_predictor_factory will be used.
        :param rating_generator_tuners:
            The tuner that tunes the hyperparameters of the rating_generators.
            If left none or as [], the rating_generators in the match_predictor_factory will be used.

        :param predictor_tuner:
            The tuner that tunes the hyperparameters of the predictor model.

        :param fit_best: Whether to refit the match_predictor with the entire training use data using best hyperparameters
        """

        self.scorer = scorer
        self.match_predictor_factory = match_predictor_factory
        self.performances_generator_tuner = performances_generator_tuner
        self.rating_generator_tuners = rating_generator_tuners or []
        self.predictor_tuner = predictor_tuner
        if isinstance(self.rating_generator_tuners, RatingGeneratorTuner):
            self.rating_generator_tuners = [self.rating_generator_tuners]
        self.fit_best = fit_best

        if len(self.rating_generator_tuners) != len(self.match_predictor_factory.rating_generators) and self.rating_generator_tuners:
            raise ValueError("Number of rating_generator_tuners must match number of rating_generators")

        if not self.performances_generator_tuner and not self.rating_generator_tuners and not self.predictor_tuner:
            raise ValueError("No tuning has been provided in config")

    def tune(self, df: pd.DataFrame) -> MatchPredictor:

        original_df = df.copy()

        column_names = [rating_generator.column_names for rating_generator in
                        self.match_predictor_factory.rating_generators]

        best_performances_generator: PerformancesGenerator = copy.deepcopy(
            self.match_predictor_factory.performances_generator)

        if self.performances_generator_tuner:
            logging.info("Tuning PreTransformers")
            best_performances_generator = self.performances_generator_tuner.tune(df=df,
                                                                                 match_predictor_factory=self.match_predictor_factory,
                                                                                 scorer=self.scorer)
        if best_performances_generator:
            df = best_performances_generator.generate(df)

        best_rating_generators = copy.deepcopy(self.match_predictor_factory.rating_generators)

        matches = []
        for col_name in column_names:
            rating_matches = convert_df_to_matches(df=df, column_names=col_name)
            matches.append(rating_matches)

        rating_generators = self.match_predictor_factory.rating_generators

        for rating_idx, rating_generator in enumerate(rating_generators):

            if self.rating_generator_tuners:
                rating_generator_tuner = self.rating_generator_tuners[rating_idx]

                rating_matches = convert_df_to_matches(df=df, column_names=column_names[rating_idx])
                matches.append(rating_matches)
                tuned_rating_generator = rating_generator_tuner.tune(df=df, matches=matches[rating_idx],
                                                                     rating_idx=rating_idx,
                                                                     scorer=self.scorer,
                                                                     match_predictor_factory=self.match_predictor_factory)

                best_rating_generators[rating_idx] = tuned_rating_generator

            match_ratings = best_rating_generators[rating_idx].generate_historical(df=df, matches=matches[rating_idx])

            for rating_feature in best_rating_generators[rating_idx].features_out:
                values = match_ratings[rating_feature]

                if len(self.rating_generator_tuners) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature
                df[rating_feature_str] = values

        best_post_transformers = copy.deepcopy(self.match_predictor_factory.post_rating_transformers)

        for post_rating_transformer in best_post_transformers:
            df = post_rating_transformer.fit_transform(df)

        if self.predictor_tuner:
            logging.info("Tuning Predictor")
            best_predictor = self.predictor_tuner.tune(df=df, scorer=self.scorer,
                                                       match_predictor_factory=self.match_predictor_factory)
        else:
            best_predictor = self.match_predictor_factory.predictor

        best_match_predictor = MatchPredictor(
            rating_generators=best_rating_generators,
            performances_generator=best_performances_generator,
            post_rating_transformers=best_post_transformers,
            post_prediction_transformers=self.match_predictor_factory.post_prediction_transformers,
            predictor=best_predictor)

        if self.fit_best:
            logging.info("Retraining best match predictor with all data")
            best_match_predictor.generate_historical(df=original_df, store_ratings=True)

        return best_match_predictor
