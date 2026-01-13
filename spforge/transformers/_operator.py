import logging
from enum import Enum

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge.transformers._base import PredictorTransformer


class Operation(Enum):
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class OperatorTransformer(PredictorTransformer):
    """
    Performs operations on two columns and stores the result in a new column.
    An operation can be subtraction, addition, multiplication, or division.
    """

    def __init__(
        self,
        feature1: str,
        operation: Operation,
        feature2: str,
        new_column_name: str | None = None,
        features: list[str] | None = None,
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
            elif self.operation == Operation.DIVIDE:
                self.new_column_name = f"{self.feature1}_divided_by_{self.feature2}"

        super().__init__(
            features=features,
            are_estimator_features=are_estimator_features,
            features_out=[self.new_column_name],
        )

    def fit(self, df: IntoFrameT):
        return self

    @nw.narwhalify
    def transform(self, df: IntoFrameT, cross_validate: bool = False) -> IntoFrameT:
        df = self._transform(df)
        return df

    def _transform(self, df: IntoFrameT) -> IntoFrameT:
        if self.feature1 not in df.columns or self.feature2 not in df.columns:
            raise ValueError(f"{self.feature1} and {self.feature2} are both missing indf")

        if self.operation == Operation.SUBTRACT:
            df = df.with_columns(
                (nw.col(self.feature1) - nw.col(self.feature2)).alias(self.new_column_name)
            )

        elif self.operation == Operation.MULTIPLY:
            df = df.with_columns(
                (nw.col(self.feature1) * nw.col(self.feature2)).alias(self.new_column_name)
            )
        elif self.operation == Operation.DIVIDE:
            df = df.with_columns(
                (nw.col(self.feature1) / nw.col(self.feature2)).alias(self.new_column_name)
            )

        else:
            logging.warning(f"Operation {self.operation} not implemented")
            raise NotImplementedError

        return df
