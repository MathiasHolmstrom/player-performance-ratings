import polars as pl

from spforge import ColumnNames


def add_team_rating(
    df: pl.DataFrame,
    column_names: ColumnNames,  # ColumnNames
    player_rating_col: str,
    team_rating_out: str,
) -> pl.DataFrame:
    mid = column_names.match_id
    tid = column_names.team_id

    return df.with_columns(
        pl.col(player_rating_col).mean().over([mid, tid]).alias(team_rating_out)
    )


def add_team_rating_projected(
    df: pl.DataFrame,
    column_names: ColumnNames,  # ColumnNames
    player_rating_col: str,
    team_rating_out: str,
) -> pl.DataFrame:
    mid = column_names.match_id
    tid = column_names.team_id
    ppw = column_names.projected_participation_weight

    if ppw:
        return df.with_columns(
            (
                (pl.col(ppw) * pl.col(player_rating_col)).sum().over([mid, tid])
                / pl.col(ppw).sum().over([mid, tid])
            ).alias(team_rating_out)
        )

    return add_team_rating(
        df=df,column_names=column_names,player_rating_col=player_rating_col, team_rating_out=team_rating_out
    )


def add_opp_team_rating(
    df: pl.DataFrame,
    column_names,  # ColumnNames
    team_rating_col: str,
    opp_team_rating_out: str,
) -> pl.DataFrame:
    mid = column_names.match_id
    tid = column_names.team_id

    team_sums = df.select([mid, tid, team_rating_col]).unique()

    opp_map = (
        team_sums.join(team_sums, on=mid, suffix="_opp")
        .filter(pl.col(tid) != pl.col(f"{tid}_opp"))
        .select(
            pl.col(mid),
            pl.col(tid),
            pl.col(f"{team_rating_col}_opp").alias(opp_team_rating_out),
        )
    )

    return df.join(opp_map, on=[mid, tid], how="left")




def add_rating_difference_projected(
    df: pl.DataFrame,
    team_rating_col: str,
    opp_team_rating_col: str,
    rating_diff_out: str,
) -> pl.DataFrame:
    return df.with_columns(
        (pl.col(team_rating_col) - pl.col(opp_team_rating_col)).alias(rating_diff_out)
    )


def add_rating_mean_projected(
    df: pl.DataFrame,
    column_names,  # ColumnNames
    player_rating_col: str,
    rating_mean_out: str,
) -> pl.DataFrame:
    """Mean across the entire match (all players). Weighted if projected_participation_weight exists."""
    mid = column_names.match_id
    ppw = column_names.projected_participation_weight

    if ppw:
        return df.with_columns(
            (
                (pl.col(ppw) * pl.col(player_rating_col)).sum().over(mid)
                / pl.col(ppw).sum().over(mid)
            ).alias(rating_mean_out)
        )

    return df.with_columns(pl.col(player_rating_col).mean().over(mid).alias(rating_mean_out))
