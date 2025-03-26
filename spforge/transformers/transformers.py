import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from spforge.predictor import BasePredictor

from spforge import ColumnNames

from spforge.transformers.base_transformer import (
    BaseTransformer,
    BaseLagGenerator,
)
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT


class NetOverPredictedPostTransformer(BaseTransformer):

    def __init__(
        self,
        predictor: BasePredictor,
        features: list[str] = None,
        lag_generators: Optional[list[BaseLagGenerator]] = None,
        prefix: str = "net_over_predicted_",
        are_estimator_features: bool = False,
    ):
        super().__init__(
            features=features,
            are_estimator_features=are_estimator_features,
            features_out=[],
        )
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
                    lag_generator._features_out = [
                        f"{lag_generator.prefix}{iteration}_{new_feature_name}"
                    ]
                    self.features_out.extend(lag_generator._features_out.copy())
                    self._estimator_features_out.extend(
                        lag_generator._features_out.copy()
                    )

        if self._are_estimator_features:
            self._estimator_features_out.append(self.predictor.pred_column)
            self.features_out.append(self.predictor.pred_column)
        if self.prefix is "":
            raise ValueError("Prefix must not be empty")

    def fit_transform(
        self, df: pd.DataFrame, column_names: Optional[ColumnNames] = None
    ) -> pd.DataFrame:
        ori_cols = df.columns.tolist()
        self.column_names = column_names
        self.predictor.train(df, features=self.features)
        df = self.predictor.predict(df)
        new_feature_name = self.prefix + self.predictor.pred_column
        if self.predictor.target not in df.columns:
            df = df.assign(**{new_feature_name: np.nan})
        else:
            df = df.assign(
                **{
                    new_feature_name: df[self.predictor.target]
                    - df[self.predictor.pred_column]
                }
            )
        for lag_generator in self.lag_generators:
            df = lag_generator.transform_historical(df, column_names=self.column_names)
        return df[list(set(ori_cols + self.features_out))]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ori_cols = df.columns.tolist()
        df = self.predictor.predict(df)
        new_feature_name = self.prefix + self.predictor.pred_column
        if self.predictor.target not in df.columns:
            df = df.assign(**{new_feature_name: np.nan})
        else:
            df = df.assign(
                **{
                    new_feature_name: df[self.predictor.target]
                    - df[self.predictor.pred_column]
                }
            )

        for lag_generator in self.lag_generators:
            df = lag_generator.transform_future(df)

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
    MULTIPLY = "multiply"


@dataclass
class ModifyOperation:
    feature1: str
    operation: Operation
    feature2: str
    new_column_name: Optional[str] = None

    def __post_init__(self):
        if self.operation == Operation.SUBTRACT and not self.new_column_name:
            self.new_column_name = f"{self.feature1}_minus_{self.feature2}"


class OperatorTransformer(BaseTransformer):
    """
    Performs operations on two columns and stores the result in a new column.
    An operation can be subtraction, addition, multiplication, or division.
    """

    def __init__(
        self,
        feature1: str,
        operation: Operation,
        feature2: str,
        new_column_name: Optional[str] = None,
        features: Optional[list[str]] = None,
        are_estimator_features: bool = True,
    ):
        """
        :param feature1: The first feature to perform the operation on
        :param operation: The operation to perform
        :param feature2: The second feature to perform the operation on
        :param new_column_name: The name of the new column to store the result in
        """
        self.feature1 = feature1
        self.operation = operation
        self.feature2 = feature2
        self.new_column_name = new_column_name

        if not self.new_column_name:

            if self.operation == Operation.SUBTRACT:
                self.new_column_name = f"{self.feature1}_minus_{self.feature2}"
            elif self.operation == Operation.MULTIPLY:
                self.new_column_name = f"{self.feature1}_times_{self.feature2}"

        super().__init__(
            features=features,
            are_estimator_features=are_estimator_features,
            features_out=[self.new_column_name],
        )

    def fit_transform(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        self.column_names = column_names
        return self.transform(df)

    @nw.narwhalify
    def transform(self, df: FrameT) -> FrameT:
        if self.feature1 not in df.columns or self.feature2 not in df.columns:
            return df
        if self.operation == Operation.SUBTRACT:
            df = df.with_columns(
                (nw.col(self.feature1) - nw.col(self.feature2)).alias(
                    self.new_column_name
                )
            )

        elif self.operation == Operation.MULTIPLY:
            df = df.with_columns(
                (nw.col(self.feature1) * nw.col(self.feature2)).alias(
                    self.new_column_name
                )
            )
        else:
            logging.warning(f"Operation {self.operation} not implemented")
            raise NotImplementedError

        return df


class PredictorTransformer(BaseTransformer):
    """
    Transformer that uses a predictor to generate predictions on the dataset
    This is useful if you want to use the output of a feature as input for another model
    """

    def __init__(self, predictor: BasePredictor, features: list[str] = None):
        """
        :param predictor: The predictor to use to add add new prediction-columns to the dataset
        :param features: The features to use for the predictor
        """
        self.predictor = predictor
        super().__init__(
            features=features, features_out=[f"{self.predictor.pred_column}"]
        )

    @nw.narwhalify
    def fit_transform(
        self, df: FrameT, column_names: Optional[None] = None
    ) -> IntoFrameT:
        self.predictor.train(df=df, features=self.features)
        return self.transform(df)

    @nw.narwhalify
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.predictor.predict(df=df)
        return df


class RatioTeamPredictorTransformer(BaseTransformer):
    """
    Transformer that trains and uses the output of a predictor and divides it by the sum of the predictions for all the players within the team
    If team_total_prediction_column is passed in, it will also multiply the ratio by the team_total_prediction_column
    This is useful to provide a normalized point-estimate for a player for the given feature.

    """

    def __init__(
        self,
        features: list[str],
        predictor: BasePredictor,
        team_total_prediction_column: Optional[str] = None,
        lag_generators: Optional[list[BaseLagGenerator]] = None,
        prefix: str = "_ratio_team",
    ):
        """
        :param features: The features to use for the predictor
        :param predictor: The predictor to use to add add new prediction-columns to the dataset
        :param team_total_prediction_column: If passed, The column to multiply the ratio by.
        :param lag_generators: Additional lag-generators (such as rolling-mean) can be performed after the ratio is calculated is passed
        :param prefix: The prefix to use for the new columns
        """

        self.predictor = predictor
        self.team_total_prediction_column = team_total_prediction_column
        self.prefix = prefix
        # self.predictor._pred_column = f"__prediction__{self.predictor.target}"
        self.lag_generators = lag_generators or []
        super().__init__(
            features=features,
            features_out=[self.predictor.target + prefix, self.predictor._pred_column],
        )

        if self.team_total_prediction_column:
            self._features_out.append(
                self.predictor.target + prefix + "_team_total_multiplied"
            )
        for lag_generator in self.lag_generators:
            self._features_out.extend(lag_generator.features_out)

        if self._are_estimator_features:
            self._estimator_features_out = self._features_out.copy()

    @nw.narwhalify
    def fit_transform(
        self, df: FrameT, column_names: Optional[ColumnNames]
    ) -> IntoFrameT:
        ori_cols = df.columns
        self.column_names = column_names
        self.predictor.train(df=df, features=self.features)
        transformed_df = nw.from_native(self.transform(df))
        for lag_generator in self.lag_generators:
            transformed_df = nw.from_native(
                lag_generator.transform_historical(
                    transformed_df, column_names=self.column_names
                )
            )

        return transformed_df.select(list(set(ori_cols + self.features_out)))

    def transform(self, df: FrameT) -> IntoFrameT:
        input_features = df.columns
        df = nw.from_native(self.predictor.predict(df=df))

        df = df.with_columns(
            [
                nw.col(self.predictor.pred_column)
                .sum()
                .over([self.column_names.match_id, self.column_names.team_id])
                .alias(f"{self.predictor.pred_column}_sum")
            ]
        )

        df = df.with_columns(
            [
                (
                    nw.col(self.predictor.pred_column)
                    / nw.col(f"{self.predictor.pred_column}_sum")
                ).alias(self._features_out[0])
            ]
        )

        if self.team_total_prediction_column:
            df = df.with_columns(
                [
                    (
                        nw.col(self._features_out[0])
                        * nw.col(self.team_total_prediction_column)
                    ).alias(
                        f"{self.predictor.target}{self.prefix}_team_total_multiplied"
                    )
                ]
            )
        for lag_transformer in self.lag_generators:
            df = lag_transformer.transform_historical(
                df, column_names=self.column_names
            )

        return df.select(list(set(input_features + self.features_out)))

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def reset(self) -> "BaseTransformer":
        for lag_generator in self.lag_generators:
            lag_generator.reset()
        return self
