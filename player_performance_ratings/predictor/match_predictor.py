from typing import List, Optional, Union

import pandas as pd
import pendulum

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper

from player_performance_ratings.predictor.estimators.classifier import SKLearnClassifierWrapper
from player_performance_ratings.data_structures import ColumnNames, Match
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.preprocessing.base_transformer import BaseTransformer


class MatchPredictor():

    def __init__(self,
                 column_names: ColumnNames,
                 rating_generators: Union[RatingGenerator, list[RatingGenerator]],
                 pre_rating_transformers: Optional[List[BaseTransformer]] = None,
                 post_rating_transformers: Optional[List[BaseTransformer]] = None,
                 predictor: [Optional[BaseMLWrapper]] = None,
                 train_split_date: Optional[pendulum.datetime] = None,
                 ):
        self.column_names = column_names
        self.pre_rating_transformers = pre_rating_transformers or []
        self.post_rating_transformers = post_rating_transformers or []

        self.predictor = predictor or SKLearnClassifierWrapper(
            features=[RatingColumnNames.RATING_DIFFERENCE],
            target=PredictColumnNames.TARGET
        )

        self.predictor.set_target(PredictColumnNames.TARGET)
        self.train_split_date = train_split_date
        self.rating_generators = rating_generators if isinstance(rating_generators, list) else [rating_generators]

    def generate_historical(self, df: pd.DataFrame, store_ratings: bool = True) -> pd.DataFrame:

        if self.predictor.target not in df.columns:
            raise ValueError(
                f"Target {self.predictor.target} not in df columns. Target is always overriden to be set to RatingColumnNames.TARGET")

        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.transform(df)

        if self.train_split_date is None:
            self.train_split_date = df.iloc[int(len(df) / 1.3)][self.column_names.start_date]

        matches = convert_df_to_matches(column_names=self.column_names, df=df)

        for rating_generator in self.rating_generators:
            if store_ratings:
                match_ratings = rating_generator.generate(matches, df=df)
            else:
                match_ratings = rating_generator.generate(matches)
            for rating_feature, values in match_ratings.items():
                df[rating_feature] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        train_df = df[df[self.column_names.start_date] <= self.train_split_date]

        self.predictor.train(train_df)
        df = self.predictor.add_prediction(df)
        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.transform(df)

        matches = convert_df_to_matches(column_names=self.column_names, df=df)

        for rating_generator in self.rating_generators:
            match_ratings = rating_generator.generate(matches, df=df)
            for rating_feature, values in match_ratings.items():
                df[rating_feature] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        df = self.predictor.add_prediction(df)
        return df

    @property
    def classes_(self):
        return self.predictor.model.classes_
