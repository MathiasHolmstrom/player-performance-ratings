import copy
import logging
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.ratings.performance_generator import Performance
from player_performance_ratings.tuner.utils import (
    get_default_lgbm_classifier_search_range,
)

from player_performance_ratings.cross_validator.cross_validator import CrossValidator

from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.ratings.performance_generator import (
    PerformancesGenerator,
    ColumnWeight,
)
from player_performance_ratings.ratings.match_generator import convert_df_to_matches

from player_performance_ratings import PipelineFactory
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner
from player_performance_ratings.tuner.rating_generator_tuner import RatingGeneratorTuner
from player_performance_ratings.tuner.performances_generator_tuner import (
    PerformancesGeneratorTuner,
)

logging.basicConfig(level=logging.INFO)


class PipelineTuner:
    """
    Given a search range, it will identify the best combination of pre_transformers, rating_generators and post_transformers.
    Using optuna, it will tune the hyperparameters of each transformer.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        cross_validator: Optional[CrossValidator] = None,
        performances_generator_tuners: Optional[
            Union[
                list[Optional[PerformancesGeneratorTuner]], PerformancesGeneratorTuner
            ]
        ] = None,
        rating_generator_tuners: Optional[
            Union[list[Optional[RatingGeneratorTuner]], RatingGeneratorTuner]
        ] = None,
        predictor_tuner: Optional[PredictorTuner] = None,
        fit_best: bool = True,
    ):
        """
        :param scorer: The scorer to use to evaluate the performance of the match_predictor
        :param pipeline:
            The factory that creates the MatchPredictor.
            Contains the parameters to create the MatchPredictor if no parameter-tuning was done.
            Based on hyperparameter-tuning the final way the match-predictor is generated will be based on a combination of the parameters in the factory and the tuned parameters.
        :param performances_generator_tuners:
            The tuner that tunes the hyperparameters of the pre_transformers.
            If left none or as [], the pre_transformers in the pipeline_factory will be used.
        :param rating_generator_tuners:
            The tuner that tunes the hyperparameters of the rating_generators.
            If left none or as [], the rating_generators in the pipeline_factory will be used.

        :param predictor_tuner:
            The tuner that tunes the hyperparameters of the predictor model.

        :param fit_best: Whether to refit the match_predictor with the entire training use data using best hyperparameters
        """

        self.pipeline = pipeline
        self._pipeline_factory = PipelineFactory(
            rating_generators=pipeline.rating_generators,
            performances_generator=pipeline.performances_generator,
            post_lag_transformers=pipeline.post_lag_transformers,
            lag_generators=pipeline.lag_generators,
            pre_lag_transformers=pipeline.pre_lag_transformers,
            predictor=pipeline.predictor,
            column_names=pipeline.column_names,
        )

        self.performances_generator_tuners = performances_generator_tuners or []
        if isinstance(self.performances_generator_tuners, PerformancesGeneratorTuner):
            self.performances_generator_tuners = [self.performances_generator_tuners]
        self.rating_generator_tuners = rating_generator_tuners or []
        self.predictor_tuner = predictor_tuner
        if isinstance(self.rating_generator_tuners, RatingGeneratorTuner):
            self.rating_generator_tuners = [self.rating_generator_tuners]
        self.fit_best = fit_best

        if (
            len(self.rating_generator_tuners)
            != len(self._pipeline_factory.rating_generators)
            and self.rating_generator_tuners
        ):
            raise ValueError(
                "Number of rating_generator_tuners must match number of rating_generators"
            )

        if (
            not self.pipeline.rating_generators
            and not self.pipeline.performances_generator
            and not self.predictor_tuner
        ):

            if "LGBM" in self.pipeline.predictor.deepest_estimator.__class__.__name__:
                self.predictor_tuner = PredictorTuner(
                    search_ranges=get_default_lgbm_classifier_search_range()
                )
                logging.info(
                    "No tuning has been provided in config, using default LGBM search range"
                )
            else:
                raise ValueError("No tuning has been provided in config")

        if self.performances_generator_tuners:
            performances = []
            for idx, tuner in enumerate(self.performances_generator_tuners):
                column_weights = []
                for parameter_search_range in tuner.performances_search_ranges[
                    idx
                ].search_ranges:
                    column_weights.append(
                        ColumnWeight(
                            name=parameter_search_range.name,
                            weight=0,
                            lower_is_better=parameter_search_range.lower_is_better,
                        )
                    )
                performances.append(
                    Performance(
                        name=self._pipeline_factory.rating_generators[
                            idx
                        ].performance_column,
                        weights=column_weights,
                    )
                )

            if self._pipeline_factory.performances_generator:
                original_pre_transformations = (
                    self._pipeline_factory.performances_generator.original_transformers
                )
                other_params = {
                    k: v
                    for k, v in self._pipeline_factory.performances_generator.__dict__.items()
                    if k
                    not in ["performances", "transformers", "original_transformers"]
                }

            else:
                original_pre_transformations = []
                other_params = {}

            self._pipeline_factory.performances_generator = PerformancesGenerator(
                performances=performances,
                transformers=original_pre_transformations,
                **other_params
            )

        self.cross_validator = cross_validator

    def tune(
        self,
        df: pd.DataFrame,
        return_df: bool = False,
        return_cross_validated_predictions: bool = False,
        cross_validator: Optional[CrossValidator] = None,
    ) -> Union[Pipeline, tuple[Pipeline, pd.DataFrame]]:

        self.cross_validator = cross_validator or self.cross_validator
        if self.cross_validator is None:
            self.cross_validator = self.pipeline._create_default_cross_validator(df)

        original_df = df.copy()

        best_performances_generator: PerformancesGenerator = copy.deepcopy(
            self._pipeline_factory.performances_generator
        )

        untrained_best_performances_generator = copy.deepcopy(
            best_performances_generator
        )

        if self.performances_generator_tuners:
            for rating_idx, performances_generator_tuner in enumerate(
                self.performances_generator_tuners
            ):
                if performances_generator_tuner is None:
                    continue
                logging.info("Tuning PreTransformers")
                best_performances_generator = performances_generator_tuner.tune(
                    df=df,
                    rating_idx=rating_idx,
                    pipeline_factory=self._pipeline_factory,
                    cross_validator=self.cross_validator,
                )
            untrained_best_performances_generator = copy.deepcopy(
                best_performances_generator
            )
        if best_performances_generator:
            df = best_performances_generator.generate(df)

        best_rating_generators = copy.deepcopy(self._pipeline_factory.rating_generators)

        matches = []
        for rating_generator in self.pipeline.rating_generators:
            rating_matches = convert_df_to_matches(
                df=df,
                column_names=self.pipeline.column_names,
                performance_column_name=rating_generator.performance_column,
            )
            matches.append(rating_matches)

        rating_generators = self._pipeline_factory.rating_generators
        untrained_best_rating_generators = copy.deepcopy(best_rating_generators)
        for rating_idx, rating_generator in enumerate(rating_generators):

            if self.rating_generator_tuners:
                rating_generator_tuner = self.rating_generator_tuners[rating_idx]
                if rating_generator_tuner is None:
                    continue

                rating_matches = convert_df_to_matches(
                    df=df,
                    column_names=self.pipeline.column_names,
                    performance_column_name=rating_generator.performance_column,
                )
                matches.append(rating_matches)
                tuned_rating_generator = rating_generator_tuner.tune(
                    df=df,
                    matches=matches[rating_idx],
                    rating_idx=rating_idx,
                    cross_validator=self.cross_validator,
                    pipeline_factory=self._pipeline_factory,
                )

                best_rating_generators[rating_idx] = tuned_rating_generator

                untrained_best_rating_generators[rating_idx] = copy.deepcopy(
                    tuned_rating_generator
                )
            match_ratings = best_rating_generators[
                rating_idx
            ].generate_historical_by_matches(
                matches=matches[rating_idx],
                column_names=self.pipeline.column_names,
            )

            for rating_feature in best_rating_generators[
                rating_idx
            ].known_features_return:
                values = match_ratings[rating_feature]

                if len(self.rating_generator_tuners) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature

                df = df.assign(**{rating_feature_str: values})

        #     best_post_transformers = copy.deepcopy(self._pipeline_factory.post_lag_transformers)
        #   untrained_best_post_transformers = copy.deepcopy(best_post_transformers)

        if self.predictor_tuner:
            logging.info("Tuning Predictor")
            best_predictor = self.predictor_tuner.tune(
                df=df,
                cross_validator=self.cross_validator,
                pipeline_factory=self._pipeline_factory,
            )
            untrained_best_predictor = copy.deepcopy(best_predictor)
        else:
            untrained_best_predictor = copy.deepcopy(self._pipeline_factory.predictor)

        best_match_predictor = Pipeline(
            rating_generators=[
                copy.deepcopy(rating_generator)
                for rating_generator in untrained_best_rating_generators
            ],
            performances_generator=copy.deepcopy(untrained_best_performances_generator),
            lag_generators=[
                copy.deepcopy(t.reset()) for t in self._pipeline_factory.lag_generators
            ],
            post_lag_transformers=[
                copy.deepcopy(t.reset())
                for t in self._pipeline_factory.post_lag_transformers
            ],
            pre_lag_transformers=[
                copy.deepcopy(t.reset())
                for t in self._pipeline_factory.pre_lag_transformers
            ],
            predictor=untrained_best_predictor,
            column_names=self.pipeline.column_names,
        )

        if self.fit_best:
            logging.info("Retraining best match predictor with all data")
            df = best_match_predictor.train_predict(
                df=original_df,
                return_features=True,
                cross_validate_predict=return_cross_validated_predictions,
                cross_validator=self.cross_validator,
            )

        if return_df:
            return best_match_predictor, df
        return best_match_predictor

    @property
    def untrained_best_model(self):
        return self._untrained_best_model
