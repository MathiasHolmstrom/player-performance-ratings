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
        assert (
            self.column_names is not None
        ), "column names must be passed when calling transform_future"
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

        df = df.with_columns(nw.lit(1).alias("is_future"))
        if self.unique_constraint:
            assert len(df.select(self.unique_constraint)) == len(
                df
            ), f"Specified unique constraint {self.unique_constraint} is not unique on the input dataframe"

        result = method(self=self, df=df, *args, **kwargs).sort("__row_index")

        if "is_future" in result.columns:
            result = result.drop("is_future")
        input_cols = [c for c in input_cols if c not in ("is_future")]
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
        if self.column_names and not self.unique_constraint:
            self.unique_constraint = (
                [
                    self.column_names.match_id,
                    self.column_names.player_id,
                    self.column_names.team_id,
                ]
                if self.column_names.player_id
                else [self.column_names.match_id, self.column_names.team_id]
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
        df = df.with_columns(nw.lit(0).alias("is_future"))
        native = nw.to_native(df)
        if isinstance(native, pd.DataFrame):
            df = nw.from_native(pl.DataFrame(native))
            ori_native = "pd"
        else:
            ori_native = "pl"
        result = method(self=self, df=df, *args, **kwargs).sort("__row_index")

        if "is_future" in result.columns:
            result = result.drop("is_future")
        input_cols = [c for c in input_cols if c not in ("is_future")]
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
                self.match_id_update_column is not None
            ), "if column names is not passed. match_id_update_column must be passed"

            if self.add_opponent:
                logging.warning(
                    "add_opponent is set but column names must be passed for opponent feats to be created"
                )
        else:
            self.match_id_update_column = self.column_names.update_match_id
        return method(self, df, self.column_names, *args, **kwargs)

    return wrapper
