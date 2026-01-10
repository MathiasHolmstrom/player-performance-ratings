import logging
from typing import Literal

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT
from sklearn.base import TransformerMixin


class OperatorTransformer(TransformerMixin):
    """
    Performs operations on two columns and stores the result in a new column.
    An operation can be subtraction, addition, multiplication, or division.
    """

    def __init__(
        self,
        feature1: str,
        operation: Literal["subtract", "multiply", "divide", "add"],
        feature2: str,
        alias: str | None = None,
        drop_cols: list[str] | None = None,
    ):
        """
        :param feature1: The first feature to perform the operation on
        :param operation: The operation to perform
        :param feature2: The second feature to perform the operation on
        :param alias: The name of the new column to store the result in
        :param drop_cols: Columns to drop after transformation
        """
        self.feature1 = feature1
        self.operation = operation
        self.feature2 = feature2
        self.drop_cols = drop_cols or []

        if not alias:
            if self.operation == "subtract":
                alias = f"{self.feature1}_minus_{self.feature2}"
            elif self.operation == "multiply":
                alias = f"{self.feature1}_times_{self.feature2}"
            elif self.operation == "add":
                alias = f"{self.feature1}_plus_{self.feature2}"
            elif self.operation == "divide":
                alias = f"{self.feature1}_divided_by_{self.feature2}"

        self.alias = alias
        super().__init__(features=[feature1, feature2], features_out=[alias])

    @nw.narwhalify
    def fit(self, df: IntoFrameT, column_names=None) -> "OperatorTransformer":
        return self

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        if self.feature1 not in df.columns or self.feature2 not in df.columns:
            return df
        if self.operation == "subtract":
            df = df.with_columns((nw.col(self.feature1) - nw.col(self.feature2)).alias(self.alias))
        elif self.operation == "multiply":
            df = df.with_columns((nw.col(self.feature1) * nw.col(self.feature2)).alias(self.alias))
        elif self.operation == "divide":
            df = df.with_columns((nw.col(self.feature1) / nw.col(self.feature2)).alias(self.alias))
        elif self.operation == "add":
            df = df.with_columns((nw.col(self.feature1) + nw.col(self.feature2)).alias(self.alias))
        else:
            logging.warning(f"Operation {self.operation} not implemented")
            raise NotImplementedError

        if self.drop_cols:
            df = df.drop(self.drop_cols)
        return df


class AggregatorTransformer(TransformerMixin):

    def __init__(
        self,
        columns: list[str],
        column_to_alias: dict[str, str] | None = None,
        granularity: list | None = None,
        aggregator: Literal["sum", "mean"] = "sum",
        drop_cols: list[str] | None = None,
    ):
        self.columns = columns
        self.column_to_alias = column_to_alias or {f: f"{f}_{aggregator}" for f in columns}
        self.granularity = granularity
        self.aggregator = aggregator
        self.drop_cols = drop_cols or []

    @nw.narwhalify
    def fit(self, df: IntoFrameT) -> "AggregatorTransformer":
        return self

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        if self.aggregator == "sum":
            if self.granularity:
                result = df.with_columns(
                    nw.col(column).sum().over(self.granularity).alias(self.column_to_alias[column])
                    for column in self.columns
                )
            else:
                result = df.with_columns(
                    nw.col(column).sum().alias(self.column_to_alias[column])
                    for column in self.columns
                )
        elif self.aggregator == "mean":
            if self.granularity:
                result = df.with_columns(
                    nw.col(column).mean().over(self.granularity).alias(self.column_to_alias[column])
                    for column in self.columns
                )
            else:
                result = df.with_columns(
                    nw.col(column).mean().alias(self.column_to_alias[column])
                    for column in self.columns
                )
        else:
            raise NotImplementedError(f"Aggregator {self.aggregator} not implemented")

        if self.drop_cols:
            result = result.drop(self.drop_cols)
        return result


class NormalizerToColumnTransformer(TransformerMixin):

    def __init__(
        self,
        column: str,
        granularity: list[str],
        normalize_to_column: str,
        drop_cols: list[str] | None = None,
    ):
        self.column = column
        self.granularity = granularity
        self.normalize_to_column = normalize_to_column
        self.drop_cols = drop_cols or []

    @nw.narwhalify
    def fit(self, df: IntoFrameT, column_names=None) -> "NormalizerToColumnTransformer":
        return self

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        df = df.with_columns(nw.col(self.column).sum().over(self.granularity).alias("__sum_value"))
        result = df.with_columns(
            (nw.col(self.column) / nw.col("__sum_value") * nw.col(self.normalize_to_column)).alias(
                self.column
            )
        )
        result = result.drop(["__sum_value"])
        if self.drop_cols:
            result = result.drop(self.drop_cols)
        return result
