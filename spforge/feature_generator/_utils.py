import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
from narwhals.typing import IntoFrameT

from spforge.data_structures import ColumnNames


def future_validator(method):
    @wraps(method)
    def wrapper(self, df: IntoFrameT, *args, **kwargs):
        assert self.column_names is not None, (
            "column names must have been passed to fit_transform() before calling transform."
            " Otherwise historical data is not stored"
        )
        return method(self, df, *args, **kwargs)

    return wrapper


def _is_duckdb_relation(x: Any) -> bool:
    """
    Detect DuckDB relation without importing duckdb.
    DuckDB relation classes typically live in module 'duckdb'
    and are named like DuckDBPyRelation.
    """
    t = type(x)
    mod = getattr(t, "__module__", "") or ""
    name = getattr(t, "__name__", "") or ""
    return mod.startswith("duckdb") and "Relation" in name


def _to_polars_eager(x: Any) -> pl.DataFrame:
    """
    Convert *anything* Narwhals supports into eager polars DataFrame.
    (No collecting here; if the input is genuinely lazy and Narwhals can't
    materialize without execution, this will fail naturally.)
    """
    if isinstance(x, pl.DataFrame):
        return x
    if isinstance(x, pl.LazyFrame):
        raise TypeError("Expected eager input here; got polars LazyFrame.")
    return cast(pl.DataFrame, nw.from_native(x).to_polars())


def _back_to_original_eager_backend(result_pl: pl.DataFrame, original_native: Any) -> Any:
    """
    Convert polars DataFrame back to the original eager backend (pandas/cudf/modin/pyarrow/etc)
    without executing anything extra.
    """
    if isinstance(original_native, pl.DataFrame):
        return result_pl

    ns = nw.get_native_namespace(original_native)

    # Arrow bridge (fast + general)
    if not hasattr(result_pl, "__arrow_c_stream__"):
        raise TypeError("Polars result cannot be exported to Arrow; cannot convert back.")
    return nw.from_arrow(result_pl, backend=ns).to_native()


def to_polars(method: Callable[..., Any]):
    @wraps(method)
    def wrapper(self, df: IntoFrameT, *args, **kwargs) -> IntoFrameT:
        if _is_duckdb_relation(df):
            out = method(self, df, *args, **kwargs)
            return cast(IntoFrameT, out)

        if isinstance(df, pl.LazyFrame):
            out = method(self, df, *args, **kwargs)
            return cast(IntoFrameT, out)

        original_native = df
        pl_df = _to_polars_eager(df)

        result = method(self, pl_df, *args, **kwargs)

        if not isinstance(result, (pl.DataFrame, pl.LazyFrame)):
            return cast(IntoFrameT, result)

        if isinstance(result, pl.LazyFrame):
            return cast(IntoFrameT, result)

        out = _back_to_original_eager_backend(result, original_native)
        return cast(IntoFrameT, out)

    return wrapper


def future_lag_transformations_wrapper(method):
    @wraps(method)
    def wrapper(self, df: IntoFrameT, *args, **kwargs):
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

        result = method(self, df, *args, **kwargs).sort("__row_index")

        input_cols = [c for c in input_cols]
        if ori_native == "pd":
            return result.select(list(set(input_cols + self.features_out))).to_pandas()
        return result.select(list(set(input_cols + self.features_out)))

    return wrapper


def historical_lag_transformations_wrapper(method):
    @wraps(method)
    def wrapper(self, df: IntoFrameT, column_names: ColumnNames | None = None, *args, **kwargs):
        input_cols = df.columns
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")
        self.column_names = column_names or self.column_names
        if self.__class__.__name__ != "RollingMeanDaysTransformer":
            if self.match_id_column is None and not self.column_names:
                raise ValueError("Either match_id_column or column_names must be passed")
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
                if self.column_names.player_id and self.column_names.player_id in df.columns
                else (
                    [self.column_names.match_id, self.column_names.team_id]
                    if self.column_names.team_id and self.column_names.team_id in df.columns
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
            raise ValueError("scale_by_participation_weight requires column_names to be provided")
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

        result = method(self, df, *args, **kwargs).sort("__row_index")

        input_cols = [c for c in input_cols]
        if ori_native == "pd":
            return result.select(list(set(input_cols + self.features_out))).to_pandas()
        return result.select(list(set(input_cols + self.features_out)))

    return wrapper


def transformation_validator(method):
    @wraps(method)
    def wrapper(self, df: IntoFrameT, *args, **kwargs):
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
    def wrapper(self, df: IntoFrameT, column_names: ColumnNames | None = None, *args, **kwargs):
        self.column_names = column_names or self.column_names

        if not self.column_names:

            if "__row_index" not in df.columns:
                df = df.with_row_index(name="__row_index")

            if hasattr(self, "days_between_lags") and self.days_between_lags:
                raise ValueError("column names must be passed if days_between_lags is set")

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
