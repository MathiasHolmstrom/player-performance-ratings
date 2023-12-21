import logging
from typing import List, Optional, Union

import pandas as pd
import pendulum

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper

from player_performance_ratings.predictor.estimators.classifier import SKLearnClassifierWrapper
from player_performance_ratings.data_structures import ColumnNames, Match
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.league_identifier import LeagueIdentifier
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.transformations.base_transformer import BaseTransformer


class MatchPredictor():

    def __init__(self,
                 column_names: Union[ColumnNames, list[ColumnNames]],
                 rating_generators: Union[RatingGenerator, list[RatingGenerator]],
                 pre_rating_transformers: Optional[List[BaseTransformer]] = None,
                 post_rating_transformers: Optional[List[BaseTransformer]] = None,
                 predictor: [Optional[BaseMLWrapper]] = None,
                 train_split_date: Optional[pendulum.datetime] = None,
                 ):

        """

        :param column_names:
        :param rating_generators: A single or a list of RatingGenerators.
        :param
            pre_rating_transformers: An optional list of transformations that take place rating generation.
            This is generally recommended if a more complex performance-value is used to update ratings.
            Although any type of feature engineering that isn't dependant upon the output of the ratings can be performed here.
        :param post_rating_transformers:
            After rating-generation, additional feature engineering can be performed.
        :param predictor:
        :param train_split_date:
        """

        self.column_names = column_names if isinstance(column_names, list) else [column_names for _ in
                                                                                 range(rating_generators)]
        self.pre_rating_transformers = pre_rating_transformers or []
        self.post_rating_transformers = post_rating_transformers or []

        if predictor is None:
            logging.warning(f"predictor is not set. Will use {RatingColumnNames.RATING_DIFFERENCE} as only feature")
        self.predictor = predictor or SKLearnClassifierWrapper(
            features=[RatingColumnNames.RATING_DIFFERENCE],
            target=PredictColumnNames.TARGET
        )

        self.predictor.set_target(PredictColumnNames.TARGET)
        self.train_split_date = train_split_date
        self.rating_generators = rating_generators if isinstance(rating_generators, list) else [rating_generators]

    def generate_historical(self, df: pd.DataFrame, matches: list[Match] = None,
                            store_ratings: bool = True) -> pd.DataFrame:

        if self.predictor.target not in df.columns:
            raise ValueError(
                f"Target {self.predictor.target} not in df columns. Target always needs to be set equal to {PredictColumnNames.TARGET}")

        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.transform(df)

        if self.train_split_date is None:
            self.train_split_date = df.iloc[int(len(df) / 1.3)][self.column_names[0].start_date]

        for rating_idx, rating_generator in enumerate(self.rating_generators):

            rating_column_names = self.column_names[rating_idx]

            if matches is None:
                matches = convert_df_to_matches(column_names=rating_column_names, df=df,
                                                league_identifier=LeagueIdentifier())

            if store_ratings:
                match_ratings = rating_generator.generate(matches, df=df, column_names=rating_column_names)
            else:
                match_ratings = rating_generator.generate(matches)
            for rating_feature, values in match_ratings.items():
                if len(self.rating_generators) > 0:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature
                df[rating_feature_str] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        train_df = df[df[self.column_names[0].start_date] <= self.train_split_date]

        self.predictor.train(train_df)
        df = self.predictor.add_prediction(df)
        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.transform(df)

        for rating_idx, rating_generator in enumerate(self.rating_generators):
            rating_column_names = self.column_names[rating_idx]

            matches = convert_df_to_matches(column_names=rating_column_names, df=df,
                                            league_identifier=LeagueIdentifier())

            match_ratings = rating_generator.generate(matches, df=df)
            for rating_feature, values in match_ratings.items():

                if len(self.rating_generators) > 0:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature
                df[rating_feature_str] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        df = self.predictor.add_prediction(df)
        return df

    @property
    def classes_(self):
        return self.predictor.model.classes_
