import pandas as pd

from player_performance_ratings import ColumnNames
import polars as pl
import narwhals as nw
from narwhals.typing import FrameT


def convert_pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    for column in df.select_dtypes(include=["category"]).columns:
        df[column] = df[column].astype(str)

    return pl.from_pandas(df)


@nw.narwhalify
def validate_sorting(df: FrameT, column_names: ColumnNames) -> None:
    df_sorted = df.sort(
        by=[
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]
    )
    if (
        not df.select(
            column_names.match_id, column_names.team_id, column_names.player_id
        )
        .to_numpy()
        .tolist()
        == df_sorted.select(
            column_names.match_id, column_names.team_id, column_names.player_id
        )
        .to_numpy()
        .tolist()
    ):
        for column in [
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]:
            df = df.with_columns(nw.col(column).cast(nw.String))

        df_sorted = df.sort(
            by=[
                column_names.start_date,
                column_names.match_id,
                column_names.team_id,
                column_names.player_id,
            ]
        )

        if (
            df.select(
                column_names.match_id, column_names.team_id, column_names.player_id
            )
            .to_numpy()
            .tolist()
            == df_sorted.select(
                column_names.match_id, column_names.team_id, column_names.player_id
            )
            .to_numpy()
            .tolist()
        ):
            return
        for column in [
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]:
            try:
                df = df.with_columns(nw.col(column).cast(nw.Int64))

            except:
                pass

        df_sorted = df.sort(
            by=[
                column_names.start_date,
                column_names.match_id,
                column_names.team_id,
                column_names.player_id,
            ]
        )

        if (
            df.select(
                column_names.match_id, column_names.team_id, column_names.player_id
            )
            .to_numpy()
            .tolist()
            == df_sorted.select(
                column_names.match_id, column_names.team_id, column_names.player_id
            )
            .to_numpy()
            .tolist()
        ):
            return

        raise ValueError(
            "df needs to be sorted by date, game_id, team_id, player_id in ascending order"
        )
