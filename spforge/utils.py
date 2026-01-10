import contextlib

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
from narwhals.typing import IntoFrameT

from spforge.data_structures import ColumnNames


def is_lightgbm_estimator(est) -> bool:
    try:
        import lightgbm as lgb

        return isinstance(est, lgb.sklearn.LGBMModel)
    except Exception:
        return est.__class__.__module__.startswith("lightgbm")


def coerce_for_lightgbm(X):
    try:
        import narwhals.stable.v2 as nw

        with contextlib.suppress(Exception):
            X = nw.to_native(X)
    except Exception:
        pass

    if hasattr(X, "to_pandas"):
        return X.to_pandas()
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)


def convert_pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    for column in df.select_dtypes(include=["category"]).columns:
        df[column] = df[column].astype(str)

    return pl.from_pandas(df)


@nw.narwhalify
def validate_sorting(df: IntoFrameT, column_names: ColumnNames) -> None:
    sort_cols = (
        [
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]
        if column_names.player_id in df.columns
        else [
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
        ]
    )
    select_cols = (
        [column_names.match_id, column_names.team_id, column_names.player_id]
        if column_names.player_id in df.columns
        else [column_names.match_id, column_names.team_id]
    )

    df_sorted = df.sort(by=sort_cols)
    if (
        df.select(select_cols).to_numpy().tolist()
        != df_sorted.select(select_cols).to_numpy().tolist()
    ):
        for column in select_cols:
            df = df.with_columns(nw.col(column).cast(nw.String))

        df_sorted = df.sort(by=sort_cols)

        if (
            df.select(select_cols).to_numpy().tolist()
            == df_sorted.select(select_cols).to_numpy().tolist()
        ):
            return
        for column in select_cols:
            with contextlib.suppress(Exception):
                df = df.with_columns(nw.col(column).cast(nw.Int64))

        df_sorted = df.sort(by=sort_cols)

        if (
            df.select(select_cols).to_numpy().tolist()
            == df_sorted.select(select_cols).to_numpy().tolist()
        ):
            return

        raise ValueError(
            "df needs to be sorted by date, game_id, team_id, player_id in ascending order"
        )
