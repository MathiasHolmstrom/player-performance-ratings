import logging
from typing import List, Optional

import pandas as pd
import pendulum

from player_performance_ratings.predictor.ml_wrappers.base_wrapper import BaseMLWrapper
from player_performance_ratings.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from player_performance_ratings.ratings.data_prepararer import MatchGenerator
from player_performance_ratings.data_structures import ColumnNames, Match
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.transformers.base_transformer import BaseTransformer


class MatchPredictor():

    def __init__(self,
                 column_names: ColumnNames,
                 rating_generator: RatingGenerator,
                 pre_rating_transformers: Optional[List[BaseTransformer]] = None,
                 post_rating_transformers: Optional[List[BaseTransformer]] = None,
                 target: Optional[str] = None,
                 predictor: [Optional[BaseMLWrapper]] = None,
                 train_split_date: Optional[pendulum.datetime] = None,
                 ):
        self.column_names = column_names
        self.pre_rating_transformers = pre_rating_transformers or []
        self.post_rating_transformers = post_rating_transformers or []
        if predictor is None:
            logging.warning(
                f"predictor was not defined. Will use rating-difference as feature and {self.column_names.performance} as target")
            self.target = target or self.column_names.performance
            self.predictor = predictor or SKLearnClassifierWrapper(
                features=[RatingColumnNames.RATING_DIFFERENCE],
                target=RatingColumnNames.TARGET
            )
        else:
            self.predictor = predictor
            self.target = predictor.target
            self.predictor.set_target(RatingColumnNames.TARGET)
        self.train_split_date = train_split_date
        self.rating_generator = rating_generator

    def generate(self, df: pd.DataFrame, matches: Optional[list[Match]] = None) -> pd.DataFrame:

        df[RatingColumnNames.TARGET] = df[self.target]

        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.transform(df)

        if self.train_split_date is None:
            self.train_split_date = df.iloc[int(len(df) / 1.3)][self.column_names.start_date]

        if not matches:
            match_generator = MatchGenerator(column_names=self.column_names)
            matches = match_generator.generate(df=df)

        match_ratings = self.rating_generator.generate(matches, df=df)
        for rating_feature, values in match_ratings.items():
            df[rating_feature] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        train_df = df[df[self.column_names.start_date] <= self.train_split_date]

        self.predictor.fit(train_df)
        df = self.predictor.add_prediction(df)
        return df
