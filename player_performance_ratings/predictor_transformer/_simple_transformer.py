from abc import abstractmethod, ABC
from typing import Literal, Optional

import pandas as pd
from narwhals.typing import FrameT
import narwhals as nw


class SimpleTransformer(ABC):

    def __init__(self, drop_cols: Optional[list[str]] = None):
        self.drop_cols = drop_cols or []

    @abstractmethod
    def transform(self, df: FrameT) -> pd.DataFrame:
        pass


class AggregatorTransformer(SimpleTransformer):

    def __init__(self, columns: list[str],column_to_alias: Optional[dict[str, str]] = None, granularity: Optional[list] = None,
                 aggregator: Literal['sum', 'mean'] = 'sum', drop_cols: Optional[list[str]] = None):
        super().__init__(drop_cols=drop_cols)
        self.columns = columns
        self.column_to_alias = column_to_alias or {f: f"{f}_{aggregator}" for f in columns}
        self.granularity = granularity
        self.aggregator = aggregator

    @nw.narwhalify
    def transform(self, df: FrameT) -> pd.DataFrame:
        if self.aggregator == 'sum':
            if self.granularity:
                return df.with_columns(
                    nw.col(column)
                    .sum()
                    .over(self.granularity)
                    .alias(self.column_to_alias[column])
                    for column in self.columns
                ).drop(self.drop_cols).to_pandas()
            else:
                return df.with_columns(
                    nw.col(column)
                    .sum()
                    .alias(self.column_to_alias[column])
                    for column in self.columns
                ).drop(self.drop_cols).to_pandas()

        elif self.aggregator == 'mean':
            if self.granularity:
                return df.with_columns(
                    nw.col(column)
                    .mean()
                    .over(self.granularity)
                    .alias(self.column_to_alias[column])
                    for column in self.columns
                ).drop(self.drop_cols).to_pandas()
            else:
                return df.with_columns(
                    nw.col(column)
                    .mean()
                    .alias(self.column_to_alias[column])
                    for column in self.columns
                ).drop(self.drop_cols).to_pandas()

        raise NotImplementedError(f"Aggregator {self.aggregator} not implemented")


class NormalizerToColumnTransformer(SimpleTransformer):

    def __init__(self, column: str, granularity: list[str], normalize_to_column: str, drop_cols: Optional[list[str]] = None):
        super().__init__(drop_cols=drop_cols)
        self.column = column
        self.granularity = granularity
        self.normalize_to_column = normalize_to_column
        self.drop_cols = drop_cols or []

    @nw.narwhalify
    def transform(self, df: FrameT) -> pd.DataFrame:
        input_cols = df.columns
        df = df.with_columns(
            nw.col(self.column)
            .sum()
            .over(self.granularity)
            .alias('__sum_value')
        )
        return df.with_columns(
            (nw.col(self.column) / nw.col('__sum_value') * nw.col(self.normalize_to_column)).alias(self.column)
        ).select(input_cols).drop(self.drop_cols).to_pandas()
