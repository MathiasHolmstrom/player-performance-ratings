import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from player_performance_ratings.predictor import GameTeamPredictor, BasePredictor

from player_performance_ratings import ColumnNames

from player_performance_ratings.transformers.base_transformer import BaseTransformer, BaseLagGenerator


class NormalizerTransformer(BaseTransformer):

    def __init__(self, features: list[str], granularity, target_mean: Optional[float] = None,
                 create_target_as_mean: bool = False):
        super().__init__(features=features)
        self.granularity = granularity
        self.target_mean = target_mean
        self.create_target_as_mean = create_target_as_mean
        self._features_to_normalization_target = {}

        if self.target_mean is None and not self.create_target_as_mean:
            raise ValueError("Either target_sum or create_target_as_mean must be set")

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
        self._features_to_normalization_target = {f: self.target_mean for f in self.features} if self.target_mean else {
            f: df[f].mean() for f in self.features}
        return self.transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{f'__mean_{feature}': df.groupby(self.granularity)[feature].transform('mean') for feature in
                          self.features})
        for feature, target_sum in self._features_to_normalization_target.items():
            df = df.assign(**{feature: df[feature] / df[f'__mean_{feature}'] * target_sum})
        return df.drop(columns=[f'__mean_{feature}' for feature in self.features])

    @property
    def features_out(self) -> list[str]:
        return self.features

    def reset(self):
        pass


class NetOverPredictedPostTransformer(BaseTransformer):

    def __init__(self,
                 predictor: BasePredictor,
                 features: list[str] = None,
                 lag_generators: Optional[list[BaseLagGenerator]] = None,
                 prefix: str = "net_over_predicted_",
                 are_estimator_features: bool = False,
                 ):
        super().__init__(features=features, are_estimator_features=are_estimator_features)
        self.prefix = prefix
        self.predictor = predictor
        self._features_out = []
        self.lag_generators = lag_generators or []
        self.column_names = None
        new_feature_name = self.prefix + self.predictor.pred_column
        self._features_out.append(new_feature_name)
        for lag_generator in self.lag_generators:
            if not lag_generator.features:
                lag_generator.features = [self.predictor.pred_column]
                for iteration in lag_generator.iterations:
                    lag_generator._features_out = [f"{lag_generator.prefix}{iteration}_{self.predictor.pred_column}"]
                    self.features_out.extend(lag_generator._features_out.copy())
                    self._estimator_features_out.extend(lag_generator._features_out.copy())
        if self.prefix is "":
            raise ValueError("Prefix must not be empty")

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
        self.column_names = column_names
        self.predictor.train(df, estimator_features=self.features)
        if self.lag_generators:
            df = self.predictor.add_prediction(df)
        for lag_generator in self.lag_generators:
            df = lag_generator.generate_historical(df, column_names=self.column_names)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.predictor.add_prediction(df)
        new_feature_name = self.prefix + self.predictor.pred_column
        if self.predictor.target not in df.columns:
            df = df.assign(**{new_feature_name: np.nan})
        else:
            df = df.assign(**{new_feature_name: df[self.predictor.target] - df[self.predictor.pred_column]})

        for lag_generator in self.lag_generators:
            fitted_game_ids = lag_generator._df[self.column_names.match_id].unique()
            if df[self.column_names.match_id].nunique() != len(
                    fitted_game_ids) + df[self.column_names.match_id].nunique():
                df = copy.deepcopy(lag_generator).generate_historical(df, column_names=self.column_names)

        return df.drop(columns=[self.predictor.pred_column])

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
        super().__init__(features=features, are_estimator_features=are_estimator_features)
        self.modify_operations = modify_operations
        self._features_out = [operation.new_column_name for operation in self.modify_operations]

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
        super().__init__(features=features)

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[None] = None) -> pd.DataFrame:
        self.predictor.train(df=df, estimator_features=self.features)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.predictor.add_prediction(df=df)
        return df

    @property
    def features_out(self) -> list[str]:
        return [f'{self.predictor.pred_column}']


class RatioTeamPredictorTransformer(BaseTransformer):
    def __init__(self,
                 features: list[str],
                 predictor: BasePredictor,
                 team_total_prediction_column: Optional[str] = None,
                 lag_generators: Optional[list[BaseLagGenerator]] = None,
                 prefix: str = "_ratio_team"
                 ):
        super().__init__(features=features)
        self.predictor = predictor
        self.team_total_prediction_column = team_total_prediction_column
        self.prefix = prefix
        self.predictor._pred_column = f"__prediction__{self.predictor.target}"
        self.lag_generators = lag_generators or []
        self._features_out = [self.predictor.target + prefix, self.predictor._pred_column]
        if self.team_total_prediction_column:
            self._features_out.append(self.predictor.target + prefix + "_team_total_multiplied")
        for lag_generator in self.lag_generators:
            self._features_out.extend(lag_generator.features_out)

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
        self.column_names = column_names
        self.predictor.train(df=df, estimator_features=self.features)
        transformed_df = self.transform(df)
        for lag_generator in self.lag_generators:
            transformed_df = lag_generator.generate_historical(transformed_df, column_names=self.column_names)

        return transformed_df

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


class NormalizerTargetColumnTransformer(BaseTransformer):

    def __init__(self, features: list[str], granularity, target_sum_column_name: str, prefix: str = "__normalized_"):
        super().__init__(features=features)
        self.granularity = granularity
        self.prefix = prefix
        self.target_sum_column_name = target_sum_column_name
        self._features_to_normalization_target = {}
        self._features_out = []
        for feature in self.features:
            self._features_out.append(f'{self.prefix}{feature}')

    def fit_transform(self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None) -> pd.DataFrame:
        self.column_names = column_names
        return self.transform(df=df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            df[f"{feature}_sum"] = df.groupby(self.granularity)[feature].transform('sum')
            df = df.assign(
                **{self.prefix + feature: df[feature] / df[f"{feature}_sum"] * df[self.target_sum_column_name]})
            df = df.drop(columns=[f"{feature}_sum"])
        return df

    @property
    def features_out(self) -> list[str]:
        return self._features_out