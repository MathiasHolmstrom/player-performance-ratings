import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import pendulum as pendulum
from sklearn.linear_model import LogisticRegression

from src.predictor.ml_wrappers.base_wrapper import BaseMLWrapper
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from src.ratings.data_prepararer import get_matches_from_df
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.rating_generator import RatingGenerator


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
                 ):
        self.column_names = column_names
        self.pre_rating_transformers = pre_rating_transformers or []
        self.player_performance_rating = player_performance_rating
        self.post_rating_transformers = post_rating_transformers or []
        if predictor is None:
            logging.warning(f"predictor is set to warn, will use rating-difference as feature and {self.column_names.performance} as target")
        self.predictor = predictor or SKLearnClassifierWrapper(
            features=[RatingColumnNames.rating_difference],
            target=self.column_names.performance
        )
        self.train_split_date = train_split_date

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.train_split_date is None:
            self.train_split_date = df.iloc[int(len(df)/1.3)][self.column_names.start_date]

        matches = get_matches_from_df(df=df, column_names=self.column_names)

        rating_generator = RatingGenerator(generate_leagues=False)

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