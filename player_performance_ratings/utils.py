import pandas as pd

from player_performance_ratings import ColumnNames


def validate_sorting(df: pd.DataFrame, column_names: ColumnNames):
    df_sorted = df.sort_values(
        by=[column_names.start_date, column_names.match_id,
            column_names.team_id, column_names.player_id])

    if not df.equals(df_sorted):
        raise ValueError("df needs to be sorted by date, game_id, team_id, player_id in ascending order")