import logging
from typing import List, Optional, Union

import pandas as pd
import pendulum
from player_performance_ratings.transformation import ColumnWeight

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper

from player_performance_ratings.predictor.estimators import SklearnPredictor
from player_performance_ratings.data_structures import ColumnNames, Match
from player_performance_ratings.ratings.league_identifier import LeagueIdentifier
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import RatingGenerator, \
    OpponentAdjustedRatingGenerator
from player_performance_ratings.transformation.base_transformer import BaseTransformer
from player_performance_ratings.transformation.factory import auto_create_pre_transformers


class MatchPredictor():

    def __init__(self,
                 column_names: Union[ColumnNames, list[ColumnNames]],
                 rating_generators: Optional[Union[RatingGenerator, list[RatingGenerator]]] = None,
                 pre_rating_transformers: Optional[List[BaseTransformer]] = None,
                 post_rating_transformers: Optional[List[BaseTransformer]] = None,
                 predictor: [Optional[BaseMLWrapper]] = None,
                 train_split_date: Optional[pendulum.datetime] = None,
                 use_auto_pre_transformers: bool = False,
                 column_weights: Optional[Union[list[list[ColumnWeight]], list[ColumnWeight]]] = None
                 ):

        """

        :param column_names:
        :param rating_generators: A single or a list of RatingGenerators.
        :param pre_rating_transformers: An optional list of transformations that take place rating generation.
            This is generally recommended if a more complex performance-value is used to update ratings.
            Although any type of feature engineering that isn't dependant upon the output of the ratings can be performed here.
        :param post_rating_transformers:
            After rating-generation, additional feature engineering can be performed.
        :param predictor:
        :param train_split_date:
        :param use_auto_pre_transformers:
            If true, the pre_rating_transformers will be automatically generated to ensure the performance-value is done according to good practices.
            For new users, this is recommended.
        :param column_weights:
            If auto_create_pre_transformers is True, column_weights must be set.
            It is generally used when multiple columns are used to calculate ratings and the columns need to be weighted when converting it to a performance_value.
            Even if only 1  feature is used but auto_create_pre_transformers is used,
             then it must still be created in order for auto_create_pre_transformers to know which columns needs to be transformed.


        """

        if rating_generators is None:
            rating_generators = [OpponentAdjustedRatingGenerator()]
            logging.warning(
                "rating generator not set. Uses OpponentAdjustedRatingGenerator. To run match_predictor without rating_generator set rating_generator to []")

        self.rating_generators: list[RatingGenerator] = rating_generators if isinstance(rating_generators, list) else [
            rating_generators]
        self.column_names = column_names if isinstance(column_names, list) else [column_names for _ in
                                                                                 self.rating_generators]

        if len(rating_generators) == 0:
            if isinstance(column_names, list):
                self.column_names = column_names
            else:
                self.column_names = [column_names]

        if len(self.column_names) < len(self.rating_generators):
            while len(self.column_names) < len(self.rating_generators):
                self.column_names.append(self.column_names[0])

        self.use_auto_pre_transformers = use_auto_pre_transformers
        if self.use_auto_pre_transformers and not column_weights:
            raise ValueError("column_weights must be set if auto_create_pre_transformers is True")

        self.column_weights = column_weights
        if self.column_weights and isinstance(self.column_weights[0], ColumnWeight):
            self.column_weights = [self.column_weights]

        if not self.use_auto_pre_transformers and column_weights:
            logging.warning(
                "column_weights is set but auto_create_pre_transformers is False. column_weights will be ignored")

        self.pre_rating_transformers = pre_rating_transformers or []
        if self.use_auto_pre_transformers:
            self.pre_rating_transformers = auto_create_pre_transformers(column_weights=self.column_weights,
                                                                        column_names=self.column_names)

        self.post_rating_transformers = post_rating_transformers or []

        self.predictor = predictor

        if self.predictor is None:
            features = []

            for c in self.post_rating_transformers:
                features += c.features_created
            for c in self.rating_generators:
                features += c.features_created

            logging.warning(f"predictor is not set. Will use {features} as features")

            self.predictor = SklearnPredictor(
                features=features,
                target=PredictColumnNames.TARGET
            )

        self.predictor.set_target(PredictColumnNames.TARGET)
        self.train_split_date = train_split_date

    def generate_historical(self, df: pd.DataFrame, matches: Optional[Union[list[Match], list[list[Match]]]] = None,
                            store_ratings: bool = True) -> pd.DataFrame:

        for pre_rating_transformer in self.pre_rating_transformers:
            df = pre_rating_transformer.fit_transform(df)

        if matches:
            if isinstance(matches[0], Match):
                matches = [matches for _ in self.rating_generators]

        if self.predictor.target not in df.columns:
            raise ValueError(
                f"Target {self.predictor.target} not in df columns. Target always needs to be set equal to {PredictColumnNames.TARGET}")

        if self.train_split_date is None:
            self.train_split_date = df.iloc[int(len(df) / 1.3)][self.column_names[0].start_date]

        for rating_idx, rating_generator in enumerate(self.rating_generators):

            rating_column_names = self.column_names[rating_idx]

            if matches is None:
                rating_matches = convert_df_to_matches(column_names=rating_column_names, df=df,
                                                       league_identifier=LeagueIdentifier())
            else:
                rating_matches = matches[rating_idx]

            if store_ratings:
                match_ratings = rating_generator.generate(matches=rating_matches, df=df,
                                                          column_names=rating_column_names)
            else:
                match_ratings = rating_generator.generate(matches=rating_matches)
            for rating_feature, values in match_ratings.items():
                if len(self.rating_generators) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature
                df[rating_feature_str] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.fit_transform(df)

        train_df = df[df[self.column_names[0].start_date] <= self.train_split_date]

        self.predictor.train(train_df)
        df = self.predictor.add_prediction(df)
        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:

        for rating_idx, rating_generator in enumerate(self.rating_generators):
            rating_column_names = self.column_names[rating_idx]

            matches = convert_df_to_matches(column_names=rating_column_names, df=df,
                                            league_identifier=LeagueIdentifier())

            match_ratings = rating_generator.generate(matches, df=df)
            for rating_feature in rating_generator.features_created:
                values = match_ratings[rating_feature]

                if len(self.rating_generators) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature
                df[rating_feature_str] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        df = self.predictor.add_prediction(df)
        return df

    @property
    def classes_(self) -> Optional[list[str]]:
        if 'classes_' not in dir(self.predictor.model):
            return None
        return self.predictor.model.classes_
