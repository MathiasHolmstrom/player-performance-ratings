from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
from player_performance_ratings import ColumnNames

from player_performance_ratings.transformation.base_transformer import BaseTransformer

@dataclass
class ColumnWeight:
    name: str
    weight: float
    lower_is_better: bool = False

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Weight must be positive")
        if self.weight > 1:
            raise ValueError("Weight must be less than 1")


class PerformancesGenerator():

    def __init__(self,
                 column_weights: Union[list[list[ColumnWeight]], list[ColumnWeight]],
                 column_names: Union[list[ColumnNames], ColumnNames],
                 pre_transformations: Optional[list[BaseTransformer]] = None,
                 ):
        self.column_names = column_names if isinstance(column_names, list) else [column_names]
        self.pre_transformations = pre_transformations
        self.column_weights = column_weights if isinstance(column_weights[0], list) else [column_weights]

    def generate(self, df):
        if self.pre_transformations:
            for pre_transformation in self.pre_transformations:
                df = pre_transformation.fit_transform(df)

        for idx, col_name in enumerate(self.column_names):
            df[col_name.performance] = self._weight_columns(df=df, col_name=col_name, col_weights=self.column_weights[idx])
        return df

    def _weight_columns(self, df: pd.DataFrame, col_name: ColumnNames, col_weights: list[ColumnWeight]) -> pd.DataFrame:
        df = df.copy()
        df[f"__{col_name.performance}"] = 0

        df['sum_cols_weights'] = 0
        for column_weight in col_weights:
            df[f'weight__{column_weight.name}'] = column_weight.weight
            df.loc[df[column_weight.name].isna(), f'weight__{column_weight.name}'] = 0
            df.loc[df[column_weight.name].isna(), column_weight.name] = 0
            df['sum_cols_weights'] = df['sum_cols_weights'] + df[f'weight__{column_weight.name}']

        drop_cols = ['sum_cols_weights', f"__{col_name.performance}"]
        for column_weight in col_weights:
            df[f'weight__{column_weight.name}'] / df['sum_cols_weights']
            drop_cols.append(f'weight__{column_weight.name}')

        for column_weight in col_weights:

            if column_weight.lower_is_better:
                df[f"__{col_name.performance}"] += df[f'weight__{column_weight.name}'] * (
                        1 - df[column_weight.name])
            else:
                df[f"__{col_name.performance}"] += df[f'weight__{column_weight.name}'] * df[column_weight.name]

        return  df[f"__{col_name.performance}"]


    @property
    def features_out(self) -> list[str]:
        return [c.performance for c in self.column_names]

