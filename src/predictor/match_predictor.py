import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import pendulum
import pendulum as pendulumf

from src.predictor.ml_wrappers.base_wrapper import BaseMLWrapper
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from src.ratings.data_prepararer import MatchGenerator
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.factory.match_generator_factory import RatingGeneratorFactory
from src.ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator


class PredictorTransformer(ABC):

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class MatchPredictor():

    def __init__(self,
                 column_names: ColumnNames,
                 pre_rating_transformers: Optional[List[PredictorTransformer]] = None,
                 player_performance_rating: Optional[RatingGenerator] = None,
                 post_rating_transformers: Optional[List[PredictorTransformer]] = None,
                 predictor: [Optional[BaseMLWrapper]] = None,
                 train_split_date: Optional[pendulum.datetime] = None,
                 start_rating_generator: Optional[StartRatingGenerator] = None,
                 performance_predictor: Optional[PerformancePredictor] = None,
                 team_rating_generator: Optional[TeamRatingGenerator] = None,
                 player_rating_generator: Optional[PlayerRatingGenerator] = None,
                 ):
        self.column_names = column_names
        self.pre_rating_transformers = pre_rating_transformers or []
        self.player_performance_rating = player_performance_rating
        self.post_rating_transformers = post_rating_transformers or []
        if predictor is None:
            logging.warning(
                f"predictor is set to warn, will use rating-difference as feature and {self.column_names.performance} as target")
        self.predictor = predictor or SKLearnClassifierWrapper(
            features=[RatingColumnNames.rating_difference],
            target=self.column_names.performance
        )
        self.start_rating_generator = start_rating_generator
        self.performance_predictor = performance_predictor
        self.team_rating_generator = team_rating_generator
        self.player_rating_generator = player_rating_generator
        self.train_split_date = train_split_date

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.train_split_date is None:
            self.train_split_date = df.iloc[int(len(df) / 1.3)][self.column_names.start_date]

        match_generator = MatchGenerator(column_names=self.column_names)
        matches = match_generator.generate(df=df)

        match_generator_factory = RatingGeneratorFactory(
            start_rating_generator=self.start_rating_generator,
            team_rating_generator=self.team_rating_generator,
            player_rating_generator=self.player_rating_generator,
            performance_predictor=self.performance_predictor,
        )
        rating_generator = match_generator_factory.create()

        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.transform(df)

        match_ratings = rating_generator.generate(matches)
        rating_difference = np.array(match_ratings.pre_match_team_rating_values) - (
            match_ratings.pre_match_opponent_rating_values)

        df[RatingColumnNames.rating_difference] = rating_difference

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        train_df = df[df[self.column_names.start_date] <= self.train_split_date]

        self.predictor.fit(train_df)
        self.predictor.add_prediction(df)
        return df
