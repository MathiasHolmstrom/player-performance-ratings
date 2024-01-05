import logging
from typing import List, Optional, Union

import pandas as pd
import pendulum
from sklearn.preprocessing import OneHotEncoder

from player_performance_ratings.ratings import PerformancesGenerator, ColumnWeight

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper

from player_performance_ratings.predictor.estimators import Predictor, GameTeamPredictor
from player_performance_ratings.data_structures import Match
from player_performance_ratings.ratings.league_identifier import LeagueIdentifier
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.ratings.rating_generator import RatingGenerator

from player_performance_ratings.transformation.base_transformer import BaseTransformer, BasePostTransformer
from player_performance_ratings.transformation.factory import auto_create_performance_generator

from player_performance_ratings.transformation.pre_transformers import ConvertDataFrameToCategoricalTransformer, \
    SkLearnTransformerWrapper

def create_predictor(
        rating_generators: Optional[Union[RatingGenerator, list[RatingGenerator]]],
        other_features: Optional[list[str]],
        other_categorical_features: Optional[list[str]],
        post_rating_transformers: Optional[List[BasePostTransformer]],
        estimator: Optional,
        group_predictor_by_game_team: bool,
        match_id_column_name: Optional[str],
        team_id_column_name: Optional[str] ,

) -> BaseMLWrapper:
    features = list(set(other_features + other_categorical_features)) or []

    for c in post_rating_transformers:
        features += c.features_out
    for rating_idx, c in enumerate(rating_generators):
        for rating_feature in c.features_out:
            if len(rating_generators) > 1:
                rating_feature_str = rating_feature + str(rating_idx)
            else:
                rating_feature_str = rating_feature
            features.append(rating_feature_str)

    logging.warning(f"predictor is not set. Will use {features} as features")
    if other_categorical_features:
        if estimator.__class__.__name__ in ('LGBMClassifier', 'LGBMRegressor', "XGBClassifier", "XGBRegressor"):
            categorical_transformers = [
                ConvertDataFrameToCategoricalTransformer(features=other_categorical_features)
            ]
        else:
            categorical_transformers = [
                SkLearnTransformerWrapper(
                    transformer=OneHotEncoder(handle_unknown='ignore'), features=other_categorical_features)
            ]
    else:
        categorical_transformers = []

    if group_predictor_by_game_team:
        match_id = rating_generators[
            0].column_names.match_id if rating_generators else match_id_column_name
        team_id = rating_generators[0].column_names.team_id if rating_generators else team_id_column_name
        if match_id is None or team_id is None:
            raise ValueError(
                "match_id and team_id must be set if group_predictor_by_game_team is used to create predictor")

        return  GameTeamPredictor(
            estimator=estimator,
            features=features,
            target=PredictColumnNames.TARGET,
            game_id_colum=match_id,
            team_id_column=team_id,
            categorical_transformers=categorical_transformers
        )


    else:
        return  Predictor(
            estimator=estimator,
            features=features,
            target=PredictColumnNames.TARGET,
            categorical_transformers=categorical_transformers
        )


class MatchPredictor():

    def __init__(self,
                 rating_generators: Optional[Union[RatingGenerator, list[RatingGenerator]]] = None,
                 performances_generator: Optional[PerformancesGenerator] = None,
                 post_rating_transformers: Optional[List[BasePostTransformer]] = None,
                 post_prediction_transformers: Optional[List[BasePostTransformer]] = None,
                 predictor: [Optional[BaseMLWrapper]] = None,
                 estimator: Optional = None,
                 estimator_or_transformers: Optional[Union[BaseMLWrapper, List[BaseTransformer]]] = None,
                 other_features: Optional[list[str]] = None,
                 other_categorical_features: Optional[list[str]] = None,
                 group_predictor_by_game_team: bool = False,
                 train_split_date: Optional[pendulum.datetime] = None,
                 date_column_name: Optional[str] = None,
                 match_id_column_name: Optional[str] = None,
                 team_id_column_name: Optional[str] = None,
                 use_auto_create_performance_calculator: bool = False,
                 column_weights: Optional[Union[list[list[ColumnWeight]], list[ColumnWeight]]] = None,
                 keep_features: bool = False,
                 ):

        """

        :param column_names:
        :param rating_generators:
        A single or a list of RatingGenerators.

        :param performances_generator:
        An optional transformer class that take place in order to convert one or multiple column names into the performance value that is used by the rating model

        :param post_rating_transformers:
            After rating-generation, additional feature engineering can be performed.

        :param predictor:
            The object which trains and returns predictions. Defaults to LGBMClassifier

        :param estimator: Sklearn-like estimator. If predictor is set, estimator will be ignored.
         Because it sometimes can be tricky to identify the names of all the features that must be passed to predictor, the user can decide to only pass in an estimator.
         The features will then be automatically identified based on features_created from the rating_generator, post_rating_transformers and other_features.
         If predictor is set, estimator will be ignored

        :param other_features: If estimator is set and predictor is not,
        other_features allows the user to pass in additional features that are not created by the rating_generator or post_rating_transformers to the predictor.

        :param other_categorical_features: Which of the other_features are categorical.
        It is not required to duplicate the categorical features in other_features and other_categorical_features.
        Simply passing an a categorical_feature in categorical_features will add it to other_features if it doesn't already exist.

        :param train_split_date:
            Date threshold which defines which periods to train the predictor on

        :param date_column_name:
            If rating_generators are not defined and train_split_date is used, then train_column_name must be set.

        :param use_auto_create_performance_calculator:
            If true, the pre_rating_transformers will be automatically generated to ensure the performance-value is done according to good practices.
            For new users, this is recommended.

        :param column_weights:
            If auto_create_pre_transformers is True, column_weights must be set.
            It is generally used when multiple columns are used to calculate ratings and the columns need to be weighted when converting it to a performance_value.
            Even if only 1  feature is used but auto_create_pre_transformers is used,
             then it must still be created in order for auto_create_pre_transformers to know which columns needs to be transformed.

        """

        self.rating_generators: list[RatingGenerator] = rating_generators if isinstance(rating_generators, list) else [
            rating_generators]
        if rating_generators is None:
            self.rating_generators: list[RatingGenerator] = []

        self.auto_create_performance_calculator = use_auto_create_performance_calculator
        if self.auto_create_performance_calculator and not column_weights:
            raise ValueError("column_weights must be set if auto_create_pre_transformers is True")

        self.column_weights = column_weights
        if self.column_weights and isinstance(self.column_weights[0], ColumnWeight):
            self.column_weights = [self.column_weights]

        if not self.auto_create_performance_calculator and column_weights:
            logging.warning(
                "column_weights is set but auto_create_pre_transformers is False. column_weights will be ignored")

        self.performances_generator = performances_generator
        if self.auto_create_performance_calculator:
            if not self.rating_generators:
                raise ValueError("rating_generators must be set if auto_create_pre_transformers is True")
            column_names = [r.column_names for r in self.rating_generators]
            self.performances_generator = auto_create_performance_generator(column_weights=self.column_weights,
                                                                            column_names=column_names)

        self.post_rating_transformers = post_rating_transformers or []
        self.post_prediction_transformers = post_prediction_transformers or []

        self.predictor = predictor
        self.other_features = other_features or []
        self.other_categorical_features = other_categorical_features or []
        self.group_predictor_by_game_team = group_predictor_by_game_team
        self.estimator_or_transformers = estimator_or_transformers
        self.match_id_column_name = match_id_column_name
        self.team_id_column_name = team_id_column_name
        self.keep_features = keep_features

        if self.predictor is not None and estimator is not None:
            logging.warning(
                "predictor and estimator is set. estimator will be ignored. If it's intended to be used, either inject it into predictor or remove predictor")

        if self.predictor is not None and self.other_features:
            logging.warning(
                "predictor and other_features is set. other_features will be ignored. If it's intended to be used, either inject it into predictor or remove predictor")

        if self.predictor is None:
            self.predictor = create_predictor(
                rating_generators=self.rating_generators,
                other_features=self.other_features,
                other_categorical_features=self.other_categorical_features,
                post_rating_transformers=self.post_rating_transformers,
                estimator=estimator,
                group_predictor_by_game_team=self.group_predictor_by_game_team,
                match_id_column_name=self.match_id_column_name,
                team_id_column_name=self.team_id_column_name
            )

        self.predictor.set_target(PredictColumnNames.TARGET)
        self.train_split_date = train_split_date
        self.date_column_name = date_column_name
        if self.train_split_date and date_column_name is None:
            if not self.rating_generators:
                raise ValueError(
                    "date_column_name must be set if train_split_date is set and rating_generators is None")

            self.date_column_name = self.rating_generators[
                0].column_names.start_date

    def generate_historical(self, df: pd.DataFrame, matches: Optional[Union[list[Match], list[list[Match]]]] = None,
                            store_ratings: bool = True) -> pd.DataFrame:

        df = df.copy()

        if self.predictor.pred_column in df.columns:
            raise ValueError(f"Predictor column {self.predictor.pred_column} already in df columns. Remove or rename before generating predictions")

        ori_cols = df.columns.tolist()

        if matches:
            if isinstance(matches[0], Match):
                matches = [matches for _ in self.rating_generators]

        elif self.performances_generator:
            df = self.performances_generator.generate(df)

        if self.predictor.target not in df.columns:
            raise ValueError(
                f"Target {self.predictor.target} not in df columns. Target always needs to be set equal to {PredictColumnNames.TARGET}")

        for rating_idx, rating_generator in enumerate(self.rating_generators):

            rating_column_names = rating_generator.column_names

            if matches is None:
                rating_matches = convert_df_to_matches(column_names=rating_column_names, df=df,
                                                       league_identifier=LeagueIdentifier())
            else:
                rating_matches = matches[rating_idx]

            if store_ratings:
                match_ratings = rating_generator.generate_historical(matches=rating_matches, df=df)
            else:
                match_ratings = rating_generator.generate_historical(matches=rating_matches)
            for rating_feature, values in match_ratings.items():
                if len(self.rating_generators) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature
                df[rating_feature_str] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.fit_transform(df)

        if self.date_column_name:
            if not self.train_split_date:
                self.train_split_date = df.iloc[int(len(df) / 1.3)][self.date_column_name]
            train_df = df[df[self.date_column_name] <= self.train_split_date]
        else:
            logging.warning("train date is not defined. Uses entire dataset to train predictor")
            train_df = df

        self.predictor.train(train_df)
        df = self.predictor.add_prediction(df)
        for post_prediction_transformer in self.post_prediction_transformers:
            df = post_prediction_transformer.transform(df)
        if not self.keep_features:
            df = df[ori_cols + [self.predictor.pred_column]]

        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ori_cols = df.columns.tolist()
        for rating_idx, rating_generator in enumerate(self.rating_generators):
            rating_column_names = rating_generator.column_names

            matches = convert_df_to_matches(column_names=rating_column_names, df=df,
                                            league_identifier=LeagueIdentifier())

            match_ratings = rating_generator.generate_future(matches, df=df)
            for rating_feature in rating_generator.features_out:
                values = match_ratings[rating_feature]

                if len(self.rating_generators) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature
                df[rating_feature_str] = values

        for post_rating_transformer in self.post_rating_transformers:
            df = post_rating_transformer.transform(df)

        df = self.predictor.add_prediction(df)
        for post_prediction_transformer in self.post_prediction_transformers:
            df = post_prediction_transformer.transform(df)
        if not self.keep_features:
            df = df[ori_cols + [self.predictor.pred_column]]
        return df

    @property
    def classes_(self) -> Optional[list[str]]:
        if 'classes_' not in dir(self.predictor.estimator):
            return None
        return self.predictor.estimator.classes_
