import logging
from enum import Enum
from typing import Optional

from spforge import ColumnNames

from spforge.transformers.base_transformer import (
    BaseTransformer,
)
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from spforge.transformers.lag_transformers import BaseLagTransformer


class Operation(Enum):
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


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
        lag_transformers: Optional[list[BaseLagTransformer]] = None,
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
        self.lag_transformers = lag_transformers or []

        if not self.new_column_name:

            if self.operation == Operation.SUBTRACT:
                self.new_column_name = f"{self.feature1}_minus_{self.feature2}"
            elif self.operation == Operation.MULTIPLY:
                self.new_column_name = f"{self.feature1}_times_{self.feature2}"
            elif self.operation == Operation.DIVIDE:
                self.new_column_name = f"{self.feature1}_divided_by_{self.feature2}"

        super().__init__(
            features=features,
            are_estimator_features=are_estimator_features,
            features_out=[self.new_column_name],
        )
        for lag_transformer in self.lag_transformers:
            self._features_out.extend(lag_transformer.features_out)
            self.predictor_features_out.extend(lag_transformer.predictor_features_out)

    def fit_transform(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        self.column_names = column_names

        df = self._transform(df)
        for lag_transformer in self.lag_transformers:
            df = nw.from_native(
                lag_transformer.transform_historical(df, column_names=self.column_names)
            )
        return df

    @nw.narwhalify
    def transform(self, df: FrameT, cross_validate: bool = False) -> IntoFrameT:
        df = self._transform(df)
        for lag_transformer in self.lag_transformers:
            if cross_validate:
                df = nw.from_native(lag_transformer.transform_historical(df))
            else:
                df = nw.from_native(lag_transformer.transform_future(df))
        return df

    def _transform(self, df: FrameT) -> FrameT:
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
        elif self.operation == Operation.DIVIDE:
            df = df.with_columns(
                (nw.col(self.feature1) / nw.col(self.feature2)).alias(
                    self.new_column_name
                )
            )

        else:
            logging.warning(f"Operation {self.operation} not implemented")
            raise NotImplementedError

        return df

    def reset(self) -> "BaseTransformer":
        for lag_generator in self.lag_transformers:
            lag_generator.reset()
        return self
