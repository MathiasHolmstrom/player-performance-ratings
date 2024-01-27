from typing import Optional

import pandas as pd
from lightgbm import LGBMRegressor

from player_performance_ratings.predictor import Predictor, GameTeamPredictor
from player_performance_ratings.transformation.base_transformer import BaseTransformer
from player_performance_ratings.transformation.post_transformers import NormalizerTargetColumnTransformer


class PredictionNormalizerTransformer(BaseTransformer):

    def __init__(self,
                 features: list[str],
                 parent_predictor_features: list[str],
                 parent_target: str,
                 child_target: str,
                 game_id_colum: str,
                 team_id_column: str,
                 child_estimator: Optional = None,
                 parent_estimator: Optional = None,
                 ):
        super().__init__(features=features)
        self.parent_predictor_features = parent_predictor_features
        self.parent_target = parent_target
        self.game_id_colum = game_id_colum
        self.team_id_column = team_id_column
        self.child_target = child_target
        self.child_estimator = child_estimator or LGBMRegressor(verbose=-100)
        self.parent_estimator = parent_estimator or LGBMRegressor(verbose=-100)

        self._child_predictor = Predictor(
            estimator=self.child_estimator,
            features=self.features,
            target =self.child_target,
            pred_column ="__child_prediction",
        )
        self._parent_predictor = GameTeamPredictor(
            estimator=self.parent_estimator,
            features=parent_predictor_features,
            target=self.parent_target,
            team_id_column=self.team_id_column,
            game_id_colum=self.game_id_colum,
            pred_column="__parent_prediction",
        )

        self._normalizer = NormalizerTargetColumnTransformer(features=['__child_prediction'], granularity=[self.game_id_colum, self.team_id_column],
                                                            target_sum_column_name='__parent_prediction', prefix='__normalized_')
        self._features_out = self._normalizer.features_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._parent_predictor.train(df)
        self._child_predictor.train(df)
        return self.transform(df=df).drop(columns=[self._child_predictor.pred_column, self._parent_predictor.pred_column])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._parent_predictor.add_prediction(df=df)
        df = self._child_predictor.add_prediction(df=df)
        df = self._normalizer.fit_transform(df=df)
        return df.drop(columns=[self._child_predictor.pred_column, self._parent_predictor.pred_column])

    @property
    def features_out(self):
        return self._features_out
