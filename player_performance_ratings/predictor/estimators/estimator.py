import logging
import warnings
import pandas as pd
from lightgbm import LGBMClassifier
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from typing import Optional

from sklearn.linear_model import LogisticRegression

from player_performance_ratings.consts import PredictColumnNames

from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper


class SkLearnGameTeamPredictor(BaseMLWrapper):

    def __init__(self,
                 game_id_colum: str,
                 team_id_column: str,
                 features: list[str],
                 weight_column: Optional[str] = None,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 model: Optional = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = "prob",
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
        :param model:
        :param multiclassifier:
        :param pred_column:
        """

        self.weight_column = weight_column
        self.game_id_colum = game_id_colum
        self.team_id_column = team_id_column
        self._target = target
        self.multiclassifier = multiclassifier
        super().__init__(target=self._target, features=features, pred_column=pred_column,
                         model=model or LogisticRegression())

    def train(self, df: pd.DataFrame) -> None:

        if len(df[self._target].unique()) > 2 and hasattr(self.model, "predict_proba"):
            logging.warning("target has more than 2 unique values, multiclassifier has therefore been set to True")
            self.multiclassifier = True

        if self._target not in df.columns:
            raise ValueError(f"target {self._target} not in df")
        grouped = self._create_grouped(df)
        self.model.fit(grouped[self.features], grouped[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds prediction to df

        :param df:
        :return: Input df with prediction column
        """

        grouped = self._create_grouped(df)

        if self.multiclassifier:
            grouped[self._pred_column] = self.model.predict_proba(grouped[self.features]).tolist()
        elif not hasattr(self.model, "predict_proba"):
            grouped[self._pred_column] = self.model.predict(grouped[self.features])
        else:
            grouped[self._pred_column] = self.model.predict_proba(grouped[self.features])[:, 1]

        if self.pred_column in df.columns:
            df = df.drop(columns=[self.pred_column])

        df = df.merge(grouped[[self.game_id_colum, self.team_id_column] + [self._pred_column]],
                      on=[self.game_id_colum, self.team_id_column])

        return df

    def _create_grouped(self, df: pd.DataFrame) -> pd.DataFrame:

        if df[self._target].dtype == 'object':
            df.loc[:, self._target] = df[self._target].astype('int')

        if self.weight_column:
            for feature in self.features:
                df = df.assign(**{feature: df[self.weight_column] * df[feature]})

        if self.weight_column:
            grouped = df.groupby([self.game_id_colum, self.team_id_column]).agg({
                **{feature: 'sum' for feature in self.features},
                self._target: 'mean',
                self.weight_column: 'sum',
            }).reset_index()
            for feature in self.features:
                grouped[feature] = grouped[feature] / grouped[self.weight_column]

            grouped.drop(columns=[self.weight_column], inplace=True)

        else:
            grouped = df.groupby([self.game_id_colum, self.team_id_column]).agg({
                **{feature: 'sum' for feature in self.features},
                self._target: 'mean',
            }).reset_index()

        grouped[self._target] = grouped[self._target].astype('int')
        return grouped


class SklearnPredictor(BaseMLWrapper):

    def __init__(self,
                 features: list[str],
                 target: Optional[str] = PredictColumnNames.TARGET,
                 model: Optional = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = "prob",
                 column_names: Optional[ColumnNames] = None,
                 categorical_features: Optional[list[str]] = None
                 ):
        self._target = target
        self.multiclassifier = multiclassifier
        self.column_names = column_names
        self.categorical_features = categorical_features or []

        if model is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=400, learning_rate=0.05)")

        super().__init__(target=self._target, features=features, pred_column=pred_column,
                         model=model or LGBMClassifier(max_depth=2, n_estimators=300, learning_rate=0.05, verbose=-100))

    def train(self, df: pd.DataFrame) -> None:
        for cat_feature in self.categorical_features:
            df = df.assign(**{cat_feature: df[cat_feature].astype('category')})

        if self.multiclassifier is False and len(df[self._target].unique()) > 2 and hasattr(self.model,
                                                                                            "predict_proba"):
            logging.warning("target has more than 2 unique values, multiclassifier has therefore been set to True")
            self.multiclassifier = True
            if len(df[self._target].unique()) > 50:
                logging.warning(
                    f"target has {len(df[self._target].unique())} unique values. This may machine-learning model to not function properly."
                    f" It is recommended to limit max and min values to ensure less than 50 unique targets")

        if hasattr(self.model, "predict_proba"):
            df = df.assign(**{self._target: df[self._target].astype('int')})

        self.model.fit(df[self.features], df[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        for cat_feature in self.categorical_features:
            df[cat_feature] = df[cat_feature].astype('category')

        if self.multiclassifier:
            df[self._pred_column] = self.model.predict_proba(df[self.features]).tolist()
            if len(set(df[self.pred_column].iloc[0])) == 2:
                raise ValueError(
                    "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly")

        elif not hasattr(self.model, "predict_proba"):
            df[self._pred_column] = self.model.predict(df[self.features])
        else:
            df[self._pred_column] = self.model.predict_proba(df[self.features])[:, 1]
        return df
