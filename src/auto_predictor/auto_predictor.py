
from typing import Any, Optional, Tuple

import pandas as pd

from src.auto_predictor.tuner.pre_transformer_tuner import ParameterSearchRange, PreTransformerTuner
from src.predictor.match_predictor import MatchPredictor
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper

from src.ratings.data_structures import ColumnNames, Match

from src.ratings.factory.match_generator_factory import RatingGeneratorFactory
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.transformers import BaseTransformer


class ParameterTuner():

    def __init__(self,
                 column_names: ColumnNames,
                 features: list[str],
                 target: str,
                 pre_transformer_search_ranges: list[Tuple[BaseTransformer, list[ParameterSearchRange]]],
                 player_rating_generator_search_ranges: Optional[list[ParameterSearchRange]] = None,
                 ):

        self.column_names = column_names
        self.features = features
        self.target = target
        self.pre_transformer_search_ranges = pre_transformer_search_ranges
        self.player_rating_generator_search_ranges = player_rating_generator_search_ranges


    def tune(self, df: pd.DataFrame):
        team_rating_generator = TeamRatingGenerator(
            player_rating_generator=PlayerRatingGenerator(rating_change_multiplier=80))
        rating_generator = RatingGenerator()
        predictor = SKLearnClassifierWrapper(features=self.features, target=self.target)
        pre_transformer_tuner = PreTransformerTuner(column_names=self.column_names,
                                                    pre_transformer_search_ranges=self.pre_transformer_search_ranges,
                                                    rating_generator=rating_generator,
                                                    predictor=predictor,
                                                    n_trials=30
                                                    )

        df = match_predictor.generate(df=df)

    @property
    def best_model(self) -> MatchPredictor:
        return


class AutoPredictor():

    def __init__(self,
                 column_names: ColumnNames,
                 hyperparameter_tuning: bool = True

                 ):
        self.column_names = column_names
        self.hyperparameter_tuning = hyperparameter_tuning

    def predict(self, df: pd.DataFrame):
        if self.hyperparameter_tuning:
            tuner = ParameterTuner()

        parameter_tuning.tune(df=df)
        parameter_tuning.best_params
