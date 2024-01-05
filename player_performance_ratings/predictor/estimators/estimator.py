import logging
import warnings
import pandas as pd
from lightgbm import LGBMClassifier
from pandas.errors import SettingWithCopyWarning
from player_performance_ratings.transformation.base_transformer import BaseTransformer


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from typing import Optional, Any

from sklearn.linear_model import LogisticRegression

from player_performance_ratings.consts import PredictColumnNames

from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper


class GameTeamPredictor(BaseMLWrapper):

    def __init__(self,
                 game_id_colum: str,
                 team_id_column: str,
                 features: list[str],
                 weight_column: Optional[str] = None,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 estimator: Optional = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = None,
                 categorical_transformers: Optional[list[BaseTransformer]] = None

                 ):
        """
        Wrapper for sklearn models that predicts game results.

        The GameTeam Predictor is intended to transform predictions from a lower granularity into a GameTeam level.
        So if input data is at game-player, data is converted to game_team before being trained
        Similar concept if it is at a granularity below game level.

        The weight_column makes it possible to weight certain rows more than others.
        For instance, if data is on game-player level and the participation rate of players is not equal -->
         setting participation_weight equal to weight_column will the feature values of the players with high participation_rates have higher weight before aggregating to game_team.


        :param game_id_colum:
        :param team_id_column:
        :param features:
        :param weight_column:
        :param target:
        :param estimator:
        :param multiclassifier:
        :param pred_column:
        """

        self.weight_column = weight_column
        self.game_id_colum = game_id_colum
        self.team_id_column = team_id_column
        self._target = target

        self.multiclassifier = multiclassifier
        super().__init__(target=self._target, features=features, pred_column=pred_column,
                         estimator=estimator or LogisticRegression(), categorical_transformers=categorical_transformers)

    def train(self, df: pd.DataFrame) -> None:
        df = self.fit_transform_categorical_transformers(df=df)
        if len(df[self._target].unique()) > 2 and hasattr(self.estimator, "predict_proba"):
            logging.warning("target has more than 2 unique values, multiclassifier has therefore been set to True")
            self.multiclassifier = True

        if self._target not in df.columns:
            raise ValueError(f"target {self._target} not in df")
        grouped = self._create_grouped(df)
        self.estimator.fit(grouped[self.estimator_features], grouped[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds prediction to df

        :param df:
        :return: Input df with prediction column
        """
        df = self.transform_categorical_transformers(df=df)
        grouped = self._create_grouped(df)

        if self.multiclassifier:
            grouped[self._pred_column] = self.estimator.predict_proba(grouped[self.estimator_features]).tolist()
        elif not hasattr(self.estimator, "predict_proba"):
            grouped[self._pred_column] = self.estimator.predict(grouped[self.estimator_features])
        else:
            grouped[self._pred_column] = self.estimator.predict_proba(grouped[self.estimator_features])[:, 1]

        if self.pred_column in df.columns:
            df = df.drop(columns=[self.pred_column])

        df = df.merge(grouped[[self.game_id_colum, self.team_id_column] + [self._pred_column]],
                      on=[self.game_id_colum, self.team_id_column])

        return df

    def _create_grouped(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_features = [feature for feature in self.estimator_features if
                            feature not in self.estimator_categorical_features]

        if self._target in df.columns:
            if df[self._target].dtype == 'object':
                df.loc[:, self._target] = df[self._target].astype('int')

        if self.weight_column:
            for feature in numeric_features:
                df = df.assign(**{feature: df[self.weight_column] * df[feature]})

        if self.weight_column:
            grouped = df.groupby([self.game_id_colum, self.team_id_column]).agg({
                **{feature: 'sum' for feature in numeric_features},
                self._target: 'mean',
                self.weight_column: 'sum',
            }).reset_index()

            for feature in numeric_features:
                grouped[feature] = grouped[feature] / grouped[self.weight_column]

            grouped.drop(columns=[self.weight_column], inplace=True)

        else:
            if self._target in df.columns:
                grouped = df.groupby([self.game_id_colum, self.team_id_column]).agg({
                    **{feature: 'mean' for feature in numeric_features},
                    self._target: 'mean',
                }).reset_index()
            else:
                grouped = df.groupby([self.game_id_colum, self.team_id_column]).agg({
                    **{feature: 'mean' for feature in numeric_features}
                }).reset_index()

        if self._target in df.columns:
            grouped[self._target] = grouped[self._target].astype('int')

        grouped = grouped.merge(df[[self.game_id_colum, self.team_id_column, *self.estimator_categorical_features]].drop_duplicates(
                                    subset=[self.game_id_colum, self.team_id_column]),
                                on=[self.game_id_colum, self.team_id_column], how='inner')
        return grouped


class Predictor(BaseMLWrapper):

    def __init__(self,
                 features: list[str],
                 target: Optional[str] = PredictColumnNames.TARGET,
                 estimator: Optional = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = None,
                 column_names: Optional[ColumnNames] = None,
                 categorical_transformers: Optional[list[BaseTransformer]] = None
                 ):
        self._target = target
        self.multiclassifier = multiclassifier
        self.column_names = column_names

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=400, learning_rate=0.05)")

        super().__init__(target=self._target, features=features, pred_column=pred_column,
                         estimator=estimator or LGBMClassifier(max_depth=2, n_estimators=300, learning_rate=0.05, verbose=-100),
                         categorical_transformers=categorical_transformers)

    def train(self, df: pd.DataFrame) -> None:
        df = self.fit_transform_categorical_transformers(df=df)
        if self.multiclassifier is False and len(df[self._target].unique()) > 2 and hasattr(self.estimator,
                                                                                            "predict_proba"):
            logging.warning("target has more than 2 unique values, multiclassifier has therefore been set to True")
            self.multiclassifier = True
            if len(df[self._target].unique()) > 50:
                logging.warning(
                    f"target has {len(df[self._target].unique())} unique values. This may machine-learning model to not function properly."
                    f" It is recommended to limit max and min values to ensure less than 50 unique targets")

        if hasattr(self.estimator, "predict_proba"):
            df = df.assign(**{self._target: df[self._target].astype('int')})

        self.estimator.fit(df[self.estimator_features], df[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.transform_categorical_transformers(df=df)
        df = df.copy()

        if self.multiclassifier:
            df[self._pred_column] = self.estimator.predict_proba(df[self.estimator_features]).tolist()
            if len(set(df[self.pred_column].iloc[0])) == 2:
                raise ValueError(
                    "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly")

        elif not hasattr(self.estimator, "predict_proba"):
            df[self._pred_column] = self.estimator.predict(df[self.estimator_features])
        else:
            df[self._pred_column] = self.estimator.predict_proba(df[self.estimator_features])[:, 1]
        return df
