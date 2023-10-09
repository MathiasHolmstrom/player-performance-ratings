from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from src.ratings.player_performance_rating import PlayerRatingGenerator


class PredictorTransformer(ABC):

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class Predictor(ABC):

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class MatchPredictor():

    def __init__(self,
                 pre_rating_transformers: List[PredictorTransformer],
                 player_performance_rating: PlayerRatingGenerator,
                 post_rating_transformers: List[PredictorTransformer],
                 predictor: Predictor
                 ):
        self.pre_rating_transformers = pre_rating_transformers
        self.player_performance_rating = player_performance_rating
        self.post_rating_transformers = post_rating_transformers
        self.predictor = predictor

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:

        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.transform(df)

        df = self.player_performance_rating.generate(df)

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        return self.predictor.predict(df)
