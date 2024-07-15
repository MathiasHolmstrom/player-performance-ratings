import pandas as pd

from player_performance_ratings import ColumnNames
import polars as pl


def convert_pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    for column in df.select_dtypes(include=['category']).columns:
        df[column] = df[column].astype(str)

    return pl.from_pandas(df)


def validate_sorting(df: pd.DataFrame, column_names: ColumnNames) -> None:
    df_sorted = df.sort_values(
        by=[
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]
    )

    if not df.equals(df_sorted):
        for column in [
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]:
            df = df.assign(**{column: df[column].astype("str")})

        df_sorted = df.sort_values(
            by=[
                column_names.start_date,
                column_names.match_id,
                column_names.team_id,
                column_names.player_id,
            ]
        )

        if df.equals(df_sorted):
            return
        for column in [
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]:
            try:
                df = df.assign(**{column: df[column].astype("int")})
            except:
                pass

        df_sorted = df.sort_values(
            by=[
                column_names.start_date,
                column_names.match_id,
                column_names.team_id,
                column_names.player_id,
            ]
        )

        if df.equals(df_sorted):
            return

        raise ValueError(
            "df needs to be sorted by date, game_id, team_id, player_id in ascending order"
        )
