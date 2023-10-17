import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import pendulum

from src.predictor.ml_wrappers.base_wrapper import BaseMLWrapper
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from src.ratings.data_prepararer import MatchGenerator
from src.ratings.data_structures import ColumnNames, Match
from src.ratings.enums import RatingColumnNames
from src.ratings.factory.match_generator_factory import RatingGeneratorFactory
from src.ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator
from src.transformers import BaseTransformer


class MatchPredictor():

    def __init__(self,
                 column_names: ColumnNames,
                 rating_features: Optional[list[RatingColumnNames]] = None,
                 pre_rating_transformers: Optional[List[BaseTransformer]] = None,
                 player_performance_rating: Optional[RatingGenerator] = None,
                 post_rating_transformers: Optional[List[BaseTransformer]] = None,
                 target: Optional[str] = None,
                 predictor: [Optional[BaseMLWrapper]] = None,
                 train_split_date: Optional[pendulum.datetime] = None,
                 start_rating_generator: Optional[StartRatingGenerator] = None,
                 performance_predictor: Optional[PerformancePredictor] = None,
                 team_rating_generator: Optional[TeamRatingGenerator] = None,
                 player_rating_generator: Optional[PlayerRatingGenerator] = None,
                 rating_generator: Optional[RatingGenerator] = None,
                 ):
        self.column_names = column_names
        self.rating_features = rating_features or [RatingColumnNames.rating_difference]
        self.pre_rating_transformers = pre_rating_transformers or []
        self.player_performance_rating = player_performance_rating
        self.post_rating_transformers = post_rating_transformers or []
        if predictor is None:
            logging.warning(
                f"predictor is set to warn, will use rating-difference as feature and {self.column_names.performance} as target")
            self.target = target or self.column_names.performance
            self.predictor = predictor or SKLearnClassifierWrapper(
                features=[RatingColumnNames.rating_difference],
                target=self.target
            )
        else:
            self.predictor = predictor

        self.start_rating_generator = start_rating_generator
        self.performance_predictor = performance_predictor
        self.team_rating_generator = team_rating_generator
        self.player_rating_generator = player_rating_generator
        self.train_split_date = train_split_date
        self.rating_generator = rating_generator
        if self.rating_generator is None:
            match_generator_factory = RatingGeneratorFactory(
                start_rating_generator=self.start_rating_generator,
                team_rating_generator=self.team_rating_generator,
                player_rating_generator=self.player_rating_generator,
                performance_predictor=self.performance_predictor,
            )
            self.rating_generator = match_generator_factory.create()



    def generate(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> pd.DataFrame:

        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.transform(df)

        if self.train_split_date is None:
            self.train_split_date = df.iloc[int(len(df) / 1.3)][self.column_names.start_date]

        if not matches:
            match_generator = MatchGenerator(column_names=self.column_names)
            _matches = match_generator.generate(df=df)

        match_ratings = self.rating_generator.generate(_matches)
        for rating_feature, values in match_ratings.items():
            df[rating_feature] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        train_df = df[df[self.column_names.start_date] <= self.train_split_date]

        self.predictor.fit(train_df)
        self.predictor.add_prediction(df)
        return df
