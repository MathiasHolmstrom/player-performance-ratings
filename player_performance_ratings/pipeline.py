import logging
from typing import List, Optional, Union

import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error

from player_performance_ratings.scorer import SklearnScorer, OrdinalLossScorer

from player_performance_ratings.cross_validator.cross_validator import CrossValidator, MatchKFoldCrossValidator
from player_performance_ratings.ratings import PerformancesGenerator

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor._base import BasePredictor

from player_performance_ratings.data_structures import Match, ColumnNames
from player_performance_ratings.ratings.league_identifier import LeagueIdentifier
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.ratings.rating_generator import RatingGenerator

from player_performance_ratings.transformation.base_transformer import BasePostTransformer


class Pipeline():

    def __init__(self,
                 predictor: BasePredictor,
                 column_names: ColumnNames,
                 rating_generators: Optional[Union[RatingGenerator, list[RatingGenerator]]] = None,
                 performances_generator: Optional[PerformancesGenerator] = None,
                 post_rating_transformers: Optional[List[BasePostTransformer]] = None,
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


        """

        self._estimator_features = predictor._estimator_features
        self.rating_generators: list[RatingGenerator] = rating_generators if isinstance(rating_generators, list) else [
            rating_generators]
        if rating_generators is None:
            self.rating_generators: list[RatingGenerator] = []

        self.post_rating_transformers = post_rating_transformers or []
        self.column_names = column_names

        for c in self.post_rating_transformers:
            self._estimator_features += [f for f in c.estimator_features_out if f not in self._estimator_features]
        for rating_idx, c in enumerate(self.rating_generators):
            for rating_feature in c.estimator_features_out:
                if len(self.rating_generators) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)

                else:
                    rating_feature_str = rating_feature
                if rating_feature_str not in self._estimator_features:
                    self._estimator_features.append(rating_feature_str)

        self.performances_generator = performances_generator
        self.predictor = predictor
        self.predictor.set_target(PredictColumnNames.TARGET)

    def cross_validate_score(self,
                             df: pd.DataFrame,
                             cross_validator: Optional[CrossValidator] = None,
                             matches: Optional[list[Match]] = None,
                             create_performance: bool = True,
                             create_rating_features: bool = True) -> float:

        if cross_validator is None:
            cross_validator = self.create_default_cross_validator(df=df)

        if create_performance:
            df = self._add_performance(df=df)

        for rating_generator in self.rating_generators:
            create_rating_features = any(
                feature not in df.columns for feature in rating_generator.estimator_features_return)
            if create_rating_features:
                break

        if create_rating_features and self.rating_generators:
            df = self._add_rating(matches=matches, df=df, store_ratings=False)

        validation_df = cross_validator.generate_validation_df(df=df, predictor=self.predictor,
                                                               column_names=self.column_names,
                                                               post_transformers=self.post_rating_transformers,
                                                               estimator_features=self._estimator_features,
                                                               return_features=False)

        if cross_validator.scorer is None:
            scorer = self._create_default_scorer(df)
            return cross_validator.cross_validation_score(validation_df=validation_df, scorer=scorer)

        return cross_validator.cross_validation_score(validation_df=validation_df)

    def cross_validate_predict(self,
                               df: pd.DataFrame,
                               cross_validator: Optional[CrossValidator] = None,
                               matches: Optional[list[Match]] = None,
                               create_performance: bool = True,
                               create_rating_features: bool = True,
                               return_features: bool = False,
                               add_train_prediction: bool = False
                               ) -> pd.DataFrame:

        cross_validated_df = df.copy()
        if cross_validator is None:
            cross_validator = self.create_default_cross_validator(df=cross_validated_df)

        if self.predictor.target not in cross_validated_df.columns:
            raise ValueError(
                f"Target {self.predictor.target} not in df columns. Target always needs to be set equal to {PredictColumnNames.TARGET}")

        if create_performance:
            cross_validated_df = self._add_performance(df=cross_validated_df)

        for rating_generator in self.rating_generators:
            create_rating_features = any(
                feature not in df.columns for feature in rating_generator.estimator_features_return)
            if create_rating_features:
                break

        if create_rating_features and self.rating_generators:
            cross_validated_df = self._add_rating(matches=matches, df=cross_validated_df, store_ratings=False)

        cross_validated_df =  cross_validator.generate_validation_df(df=cross_validated_df,
                                                                     predictor=self.predictor,
                                                                     column_names=self.column_names,
                                                                     post_transformers=self.post_rating_transformers,
                                                                     estimator_features=self._estimator_features,
                                                                     return_features=return_features,
                                                                     add_train_prediction=add_train_prediction)

        cn = self.column_names
        for _, row in df[[cn.match_id, cn.team_id, cn.player_id]].dtypes.reset_index().iterrows():
            cross_validated_df[row['index']] = cross_validated_df[row['index']].astype(row[0])
        if return_features:
            new_feats = [f for f in cross_validated_df.columns if f not in df.columns]
            return df.merge(
                cross_validated_df[new_feats + [cn.match_id, cn.team_id, cn.player_id]],
                on=[cn.match_id, cn.team_id, cn.player_id], how='left')

        return df.merge(
            cross_validated_df[self.predictor.columns_added + [cn.match_id, cn.team_id, cn.player_id]],
            on=[cn.match_id, cn.team_id, cn.player_id], how='left')

    def create_default_cross_validator(self, df: pd.DataFrame) -> CrossValidator:

        scorer = self._create_default_scorer(df)

        return MatchKFoldCrossValidator(
            date_column_name=self.column_names.start_date,
            match_id_column_name=self.column_names.rating_update_match_id,
            scorer=scorer,
        )

    def _create_default_scorer(self, df: pd.DataFrame):
        if self.predictor.estimator_type == "regressor":
            scorer = SklearnScorer(scorer_function=mean_absolute_error, pred_column=self.predictor.pred_column)
            logging.info("Using mean_absolute_error as scorer")
        else:
            if len(df[PredictColumnNames.TARGET].unique()) > 2:
                scorer = OrdinalLossScorer(pred_column=self.predictor.pred_column)
                logging.info("Using ordinal loss as scorer")
            else:
                scorer = SklearnScorer(scorer_function=log_loss, pred_column=self.predictor.pred_column)
                logging.info("Using log_loss as scorer")

        return scorer

    def train(self, df: pd.DataFrame, matches: Optional[Union[list[Match], list[list[Match]]]] = None,
              store_ratings: bool = True, return_features: bool = False) -> pd.DataFrame:

        df_with_predict = df.copy()

        if self.predictor.target not in df_with_predict.columns:
            raise ValueError(
                f"Target {self.predictor.target} not in df columns. Target always needs to be set equal to {PredictColumnNames.TARGET}")

        ori_cols = df.columns.tolist()
        df_with_predict = self._add_performance(df=df_with_predict)
        if self.rating_generators:
            df_with_predict = self._add_rating(matches=matches, df=df_with_predict, store_ratings=store_ratings)

        for post_rating_transformer in self.post_rating_transformers:
            df_with_predict = post_rating_transformer.fit_transform(df_with_predict, column_names=self.column_names)

        self.predictor.train(df=df_with_predict, estimator_features=self._estimator_features)
        df_with_predict = self.predictor.add_prediction(df=df_with_predict)
        cn = self.column_names
        for _, row in df[[cn.match_id,cn.team_id, cn.player_id]].dtypes.reset_index().iterrows():
            df_with_predict[row['index']] = df_with_predict[row['index']].astype(row[0])

        if return_features:
            new_feats = [f for f in df.columns if f not in ori_cols]
            return df.merge(
                df_with_predict[new_feats + [cn.match_id, cn.team_id, cn.player_id]],
                on=[cn.match_id, cn.team_id, cn.player_id], how='left')

        return df.merge(
            df_with_predict[self.predictor.columns_added + [cn.match_id, cn.team_id, cn.player_id]],
            on=[cn.match_id, cn.team_id, cn.player_id], how='left')

    def _add_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.predictor.pred_column in df.columns:
            raise ValueError(
                f"Predictor column {self.predictor.pred_column} already in df columns. Remove or rename before generating predictions")


        elif self.performances_generator:
            df = self.performances_generator.generate(df)

        if self.predictor.target not in df.columns:
            raise ValueError(
                f"Target {self.predictor.target} not in df columns. Target always needs to be set equal to {PredictColumnNames.TARGET}")

        return df

    def _add_rating(self, matches: Optional[Union[list[Match], Match]], df: pd.DataFrame,
                    store_ratings: bool = True):

        if matches:
            if isinstance(matches[0], Match):
                matches = [matches for _ in self.rating_generators]

        rg = self.rating_generators[0]
        ratings_df = rg.ratings_df

        if isinstance(ratings_df, pd.DataFrame):
            rating_game_ids = ratings_df[rg.column_names.match_id].unique()
            new_game_ids = df[rg.column_names.match_id].unique()
        else:
            rating_game_ids = []

        df_no_ratings = df[~df[self.column_names.match_id].isin(rating_game_ids)]
        df_calculated_ratings = df[df[self.column_names.match_id].isin(rating_game_ids)]
        for rating_idx, rating_generator in enumerate(self.rating_generators):
            if len(df_no_ratings) > 0:

                if matches is None:
                    rating_matches = convert_df_to_matches(column_names=self.column_names,
                                                           df=df_no_ratings,
                                                           league_identifier=LeagueIdentifier(),
                                                           performance_column_name=rating_generator.performance_column)
                else:
                    rating_matches = matches[rating_idx]
                    if len(df_no_ratings) != len(df):
                        rating_matches = [m for m in rating_matches if m.id in new_game_ids]

                if store_ratings:
                    match_ratings = rating_generator.generate_historical(matches=rating_matches,
                                                                         column_names=self.column_names, df=df)
                else:
                    match_ratings = rating_generator.generate_historical(df=None, matches=rating_matches,
                                                                         column_names=self.column_names)

                for rating_feature, values in match_ratings.items():
                    if len(self.rating_generators) > 1:
                        rating_feature_str = rating_feature + str(rating_idx)
                    else:
                        rating_feature_str = rating_feature
                    df_no_ratings[rating_feature_str] = values

            if len(df_calculated_ratings) > 0:
                ratings_df = rating_generator.ratings_df
                rating_cols = rating_generator.estimator_features_return + [
                    rating_generator.column_names.match_id, rating_generator.column_names.team_id,
                    rating_generator.column_names.player_id]
                df_calculated_ratings = df_calculated_ratings[
                    [f for f in df_calculated_ratings.columns if
                     f not in rating_generator.estimator_features_return]].merge(
                    ratings_df[rating_cols], on=[
                        rating_generator.column_names.match_id, rating_generator.column_names.team_id,
                        rating_generator.column_names.player_id], how='inner')

        if len(df_no_ratings) != len(df):
            col = rg.column_names
            df = pd.concat([df_calculated_ratings, df_no_ratings])
            df = df.drop_duplicates(subset=[col.match_id, col.team_id, col.player_id])
        else:
            df = df_no_ratings

        return df

    def future_predict(self, df: pd.DataFrame, return_features: bool = False) -> pd.DataFrame:
        df_with_predict = df.copy()

        for rating_idx, rating_generator in enumerate(self.rating_generators):
            rating_column_names = rating_generator.column_names

            matches = convert_df_to_matches(column_names=rating_column_names, df=df_with_predict,
                                            league_identifier=LeagueIdentifier(),
                                            performance_column_name=rating_generator.performance_column)

            match_ratings = rating_generator.generate_future(matches=matches, df=df_with_predict)
            for rating_feature in rating_generator.features_out:
                values = match_ratings[rating_feature]

                if len(self.rating_generators) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature
                df_with_predict[rating_feature_str] = values

        for post_rating_transformer in self.post_rating_transformers:
            df_with_predict = post_rating_transformer.transform(df_with_predict)

        df_with_predict = self.predictor.add_prediction(df_with_predict)

        cn = self.column_names
        for _, row in df[[cn.match_id, cn.team_id, cn.player_id]].dtypes.reset_index().iterrows():
            df_with_predict[row['index']] = df_with_predict[row['index']].astype(row[0])
        if return_features:
            new_feats = [f for f in df_with_predict.columns if f not in df.columns]
            return df.merge(
                df_with_predict[new_feats + [cn.match_id, cn.team_id, cn.player_id]],
                on=[cn.match_id, cn.team_id, cn.player_id], how='left')

        return df.merge(
            df_with_predict[self.predictor.columns_added + [cn.match_id, cn.team_id, cn.player_id]],
            on=[cn.match_id, cn.team_id, cn.player_id], how='left')

    @property
    def classes_(self) -> Optional[list[str]]:
        if 'classes_' not in dir(self.predictor.estimator):
            return None
        return self.predictor.estimator.classes_
