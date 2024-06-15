import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from player_performance_ratings.predictor import BasePredictor

from player_performance_ratings import ColumnNames

from player_performance_ratings.transformers.base_transformer import BaseTransformer, BaseLagGenerator


class NetOverPredictedPostTransformer(BaseTransformer):

    def __init__(self,
                 predictor: BasePredictor,
                 features: list[str] = None,
                 lag_generators: Optional[list[BaseLagGenerator]] = None,
                 prefix: str = "net_over_predicted_",
                 are_estimator_features: bool = False,
                 ):
        super().__init__(features=features, are_estimator_features=are_estimator_features, features_out=[])
        self.prefix = prefix
        self.predictor = predictor
        self._features_out = []
        self.lag_generators = lag_generators or []
        self.column_names = None
        new_feature_name = self.prefix + self.predictor.pred_column
        self._features_out.append(new_feature_name)
        self._estimator_features_out = []
        for lag_generator in self.lag_generators:
            if not lag_generator.features:
                lag_generator.features = [new_feature_name]
                for iteration in lag_generator.iterations:
                    lag_generator._features_out = [f"{lag_generator.prefix}{iteration}_{new_feature_name}"]
                    self.features_out.extend(lag_generator._features_out.copy())
                    self._estimator_features_out.extend(lag_generator._features_out.copy())

        if self._are_estimator_features:
            self._estimator_features_out.append(self.predictor.pred_column)
            self.features_out.append(self.predictor.pred_column)
        if self.prefix is "":
            raise ValueError("Prefix must not be empty")

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
        ori_cols = df.columns.tolist()
        self.column_names = column_names
        self.predictor.train(df, estimator_features=self.features)
        df = self.predictor.add_prediction(df)
        new_feature_name = self.prefix + self.predictor.pred_column
        if self.predictor.target not in df.columns:
            df = df.assign(**{new_feature_name: np.nan})
        else:
            df = df.assign(**{new_feature_name: df[self.predictor.target] - df[self.predictor.pred_column]})
        for lag_generator in self.lag_generators:
            df = lag_generator.generate_historical(df, column_names=self.column_names)
        return df[list(set(ori_cols + self.features_out))]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ori_cols = df.columns.tolist()
        df = self.predictor.add_prediction(df)
        new_feature_name = self.prefix + self.predictor.pred_column
        if self.predictor.target not in df.columns:
            df = df.assign(**{new_feature_name: np.nan})
        else:
            df = df.assign(**{new_feature_name: df[self.predictor.target] - df[self.predictor.pred_column]})

        for lag_generator in self.lag_generators:
            df = lag_generator.generate_future(df)

        return df[list(set(ori_cols + self.features_out))]

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def reset(self) -> "BaseTransformer":
        for lag_generator in self.lag_generators:
            lag_generator.reset()
        return self


class Operation(Enum):
    SUBTRACT = "subtract"


@dataclass
class ModifyOperation:
    feature1: str
    operation: Operation
    feature2: str
    new_column_name: Optional[str] = None

    def __post_init__(self):
        if self.operation == Operation.SUBTRACT and not self.new_column_name:
            self.new_column_name = f"{self.feature1}_minus_{self.feature2}"


class ModifierTransformer(BaseTransformer):

    def __init__(self,
                 modify_operations: list[ModifyOperation],
                 features: list[str] = None,
                 are_estimator_features: bool = True,
                 ):
        self.modify_operations = modify_operations
        _features_out = [operation.new_column_name for operation in self.modify_operations]
        super().__init__(features=features, are_estimator_features=are_estimator_features, features_out=_features_out)

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames]) -> pd.DataFrame:
        self.column_names = column_names
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for operation in self.modify_operations:
            if operation.operation == Operation.SUBTRACT:
                if operation.feature1 not in df.columns or operation.feature2 not in df.columns:
                    df = df.assign(**{operation.new_column_name: np.nan})

                else:
                    df = df.assign(**{operation.new_column_name: df[operation.feature1] - df[operation.feature2]})

        return df


class PredictorTransformer(BaseTransformer):

    def __init__(self, predictor: BasePredictor, features: list[str] = None):
        self.predictor = predictor
        super().__init__(features=features, features_out=[f'{self.predictor.pred_column}'])

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[None] = None) -> pd.DataFrame:
        self.predictor.train(df=df, estimator_features=self.features)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.predictor.add_prediction(df=df)
        return df


class RatioTeamPredictorTransformer(BaseTransformer):
    def __init__(self,
                 features: list[str],
                 predictor: BasePredictor,
                 team_total_prediction_column: Optional[str] = None,
                 lag_generators: Optional[list[BaseLagGenerator]] = None,
                 prefix: str = "_ratio_team"
                 ):
        self.predictor = predictor
        self.team_total_prediction_column = team_total_prediction_column
        self.prefix = prefix
        self.predictor._pred_column = f"__prediction__{self.predictor.target}"
        self.lag_generators = lag_generators or []
        super().__init__(features=features, features_out=[self.predictor.target + prefix, self.predictor._pred_column])

        if self.team_total_prediction_column:
            self._features_out.append(self.predictor.target + prefix + "_team_total_multiplied")
        for lag_generator in self.lag_generators:
            self._features_out.extend(lag_generator.features_out)

        if self._are_estimator_features:
            self._estimator_features_out = self._features_out.copy()

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
        ori_cols = df.columns.tolist()
        self.column_names = column_names
        self.predictor.train(df=df, estimator_features=self.features)
        transformed_df = self.transform(df)
        for lag_generator in self.lag_generators:
            transformed_df = lag_generator.generate_historical(transformed_df, column_names=self.column_names)

        return transformed_df[list(set(ori_cols + self.features_out))]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.predictor.add_prediction(df=df)

        df[self.predictor.pred_column + "_sum"] = df.groupby([self.column_names.match_id, self.column_names.team_id])[
            self.predictor.pred_column].transform('sum')
        df[self._features_out[0]] = df[self.predictor.pred_column] / df[self.predictor.pred_column + "_sum"]
        if self.team_total_prediction_column:
            df = df.assign(**{self.predictor.target + self.prefix + "_team_total_multiplied": df[self._features_out[
                0]] * df[
                                                                                                  self.team_total_prediction_column]})

        for lag_transformer in self.lag_generators:
            df = lag_transformer.generate_historical(df, column_names=self.column_names)

        return df.drop(columns=[self.predictor.pred_column + "_sum"])

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def reset(self) -> "BaseTransformer":
        for lag_generator in self.lag_generators:
            lag_generator.reset()
        return self
