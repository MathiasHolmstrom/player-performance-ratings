import polars as pl

from spforge.data_structures import ColumnNames

# Internal column names for scaled participation weights
_SCALED_PW = "__scaled_participation_weight__"
_SCALED_PPW = "__scaled_projected_participation_weight__"


def add_team_rating(
    df: pl.DataFrame,
    column_names: ColumnNames,  # ColumnNames
    player_rating_col: str,
    team_rating_out: str,
) -> pl.DataFrame:
    mid = column_names.match_id
    tid = column_names.team_id

    return df.with_columns(pl.col(player_rating_col).mean().over([mid, tid]).alias(team_rating_out))


def add_day_number_utc(
    df: pl.DataFrame,
    start_date_col: str,
    out_col: str = "__day_number",
) -> pl.DataFrame:
    dtype = df.schema[start_date_col]
    c = pl.col(start_date_col)

    if dtype == pl.Utf8:
        dt = c.cast(pl.Datetime(time_zone="UTC"), strict=False).dt.replace_time_zone(None)
    elif isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        dt = c.dt.replace_time_zone("UTC").dt.replace_time_zone(None)
    elif isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        dt = c.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    else:
        raise TypeError(f"Unsupported dtype for {start_date_col}: {dtype}")

    start_as_int = dt.cast(pl.Date).cast(pl.Int32)
    return df.with_columns((start_as_int - start_as_int.min() + 1).alias(out_col))


def add_team_rating_projected(
    df: pl.DataFrame,
    column_names: ColumnNames,  # ColumnNames
    player_rating_col: str,
    team_rating_out: str,
) -> pl.DataFrame:
    mid = column_names.match_id
    tid = column_names.team_id
    ppw = column_names.projected_participation_weight

    # Use scaled column if available (clipped to [0, 1]), otherwise raw column
    weight_col = _SCALED_PPW if _SCALED_PPW in df.columns else ppw

    if weight_col and weight_col in df.columns:
        return df.with_columns(
            (
                (pl.col(weight_col) * pl.col(player_rating_col)).sum().over([mid, tid])
                / pl.col(weight_col).sum().over([mid, tid])
            ).alias(team_rating_out)
        )

    return add_team_rating(
        df=df,
        column_names=column_names,
        player_rating_col=player_rating_col,
        team_rating_out=team_rating_out,
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

    # Ensure we don't create duplicate columns - use coalesce to handle existing columns
    result = df.join(opp_map, on=[mid, tid], how="left", coalesce=True)
    return result


def add_opponent_rating_projected(
    df: pl.DataFrame,
    opp_team_rating_col: str,
    opponent_rating_out: str,
) -> pl.DataFrame:
    """Alias for add_opp_team_rating - returns the opponent team rating column with a new name."""
    return df.with_columns(pl.col(opp_team_rating_col).alias(opponent_rating_out))


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

    # Use scaled column if available (clipped to [0, 1]), otherwise raw column
    weight_col = _SCALED_PPW if _SCALED_PPW in df.columns else ppw

    if weight_col and weight_col in df.columns:
        return df.with_columns(
            (
                (pl.col(weight_col) * pl.col(player_rating_col)).sum().over(mid)
                / pl.col(weight_col).sum().over(mid)
            ).alias(rating_mean_out)
        )

    return df.with_columns(pl.col(player_rating_col).mean().over(mid).alias(rating_mean_out))
