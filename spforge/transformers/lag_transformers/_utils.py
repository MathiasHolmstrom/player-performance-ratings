import logging
from functools import wraps
from typing import Optional

import polars as pl
import pandas as pd
from narwhals.typing import FrameT
import narwhals as nw

from spforge import ColumnNames


def future_validator(method):
    @wraps(method)
    def wrapper(self, df: FrameT, *args, **kwargs):
        assert self.column_names is not None, (
            "column names must have been passed to transform_historical() before calling transform_future."
            " Otherwise historical data is not stored"
        )
        return method(self, df, *args, **kwargs)

    return wrapper


def future_lag_transformations_wrapper(method):
    @wraps(method)
    def wrapper(self, df: FrameT, *args, **kwargs):
        df = df.drop([f for f in self.features_out if f in df.columns])
        input_cols = df.columns
        if "__row_index" not in df.columns:
            df = df.with_row_index("__row_index")

        if isinstance(nw.to_native(df), pd.DataFrame):
            ori_native = "pd"
            df = nw.from_native(pl.DataFrame(nw.to_native(df)))
        else:
            ori_native = "pl"

        if self.unique_constraint:
            assert len(df.select(self.unique_constraint)) == len(
                df
            ), f"Specified unique constraint {self.unique_constraint} is not unique on the input dataframe"

        result = method(self=self, df=df, *args, **kwargs).sort("__row_index")

        input_cols = [c for c in input_cols]
        if ori_native == "pd":
            return result.select(list(set(input_cols + self.features_out))).to_pandas()
        return result.select(list(set(input_cols + self.features_out)))

    return wrapper


def historical_lag_transformations_wrapper(method):
    @wraps(method)
    def wrapper(
        self, df: FrameT, column_names: Optional[ColumnNames] = None, *args, **kwargs
    ):
        input_cols = df.columns
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")
        self.column_names = column_names or self.column_names
        if not self.__class__.__name__ == "RollingMeanDaysTransformer":
            if self.match_id_column is None and not self.column_names:
                raise ValueError(
                    "Either match_id_column or column_names must be passed"
                )
            self.match_id_column = self.match_id_column or self.column_names.match_id
            if self.group_to_granularity:
                self.group_to_granularity = self.group_to_granularity
            elif self.match_id_column:
                self.group_to_granularity = [self.match_id_column, *self.granularity]
            elif self.column_names:
                self.group_to_granularity = [
                    self.column_names.match_id,
                    *self.granularity,
                ]
            else:
                self.group_to_granularity = None
            assert (
                self.group_to_granularity is not None
            ), "Either group_to_granularity or match_id_column must be passed"

        if self.column_names and not self.unique_constraint:
            self.unique_constraint = (
                [
                    self.column_names.match_id,
                    self.column_names.player_id,
                    self.column_names.team_id,
                ]
                if self.column_names.player_id
                and self.column_names.player_id in df.columns
                else (
                    [self.column_names.match_id, self.column_names.team_id]
                    if self.column_names.team_id
                    and self.column_names.team_id in df.columns
                    else None
                )
            )
            if not self.unique_constraint:
                raise ValueError(
                    "Unique Constraint could not be identified as neither player-id not team-id is present in the dataframe. Please pass unique_constraint explicitly"
                )

        if self.unique_constraint:
            assert len(df.select(self.unique_constraint)) == len(
                df
            ), f"Specified unique constraint {self.unique_constraint} is not unique on the input dataframe"
        if (
            self.scale_by_participation_weight
            and not self.column_names
            or self.scale_by_participation_weight
            and not self.column_names.participation_weight
        ):
            raise ValueError(
                "scale_by_participation_weight requires column_names to be provided"
            )
        if not self.column_names and not self.update_column:
            raise ValueError("column_names or update_column must be provided")
        self.update_column = self.update_column or self.column_names.update_match_id

        self.group_to_granularity = self.group_to_granularity or self.unique_constraint
        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            df = nw.from_native(pl.DataFrame(native))
            ori_native = "pd"
        else:
            ori_native = "pl"

        result = method(self=self, df=df, *args, **kwargs).sort("__row_index")

        input_cols = [c for c in input_cols]
        if ori_native == "pd":
            return result.select(list(set(input_cols + self.features_out))).to_pandas()
        return result.select(list(set(input_cols + self.features_out)))

    return wrapper


def transformation_validator(method):
    @wraps(method)
    def wrapper(self, df: FrameT, *args, **kwargs):
        input_row_count = len(df)
        input_cols = df.columns
        result = method(self, df, *args, **kwargs)
        output_row_count = len(result)
        assert (
            input_row_count == output_row_count
        ), f"Row count mismatch: input had {input_row_count} rows, output had {output_row_count} rows"
        for col in input_cols:
            if col == "__row_index":
                continue
            assert col in result.columns, f"Column {col} not found in output"

        return result

    return wrapper


def required_lag_column_names(method):
    @wraps(method)
    def wrapper(
        self, df: FrameT, column_names: Optional[ColumnNames] = None, *args, **kwargs
    ):
        self.column_names = column_names or self.column_names

        if not self.column_names:

            if "__row_index" not in df.columns:
                df = df.with_row_index(name="__row_index")

            if hasattr(self, "days_between_lags") and self.days_between_lags:
                raise ValueError(
                    "column names must be passed if days_between_lags is set"
                )

            assert (
                self.update_column is not None or self.group_to_granularity is not None
            ), "if column names is not passed. Either update_column or group_to_granularity must be passed"

            if self.add_opponent:
                logging.warning(
                    "add_opponent is set but column names must be passed for opponent feats to be created"
                )
        elif self.column_names.update_match_id != self.column_names.match_id:
            self.update_column = self.column_names.update_match_id
            assert (
                self.update_column in df.columns
            ), f"update_column {self.update_column} not found in input dataframe"
        return method(self, df, self.column_names, *args, **kwargs)

    return wrapper
