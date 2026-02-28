import numpy as np
import polars as pl

from spforge.data_structures import ColumnNames

# Internal column names for scaled participation weights
_SCALED_PW = "__scaled_participation_weight__"
_SCALED_PPW = "__scaled_projected_participation_weight__"

_NUMPY_THRESHOLD = 500


def _group_ids(keys: list) -> tuple[np.ndarray, int]:
    """Map list of hashable keys to dense integer group IDs."""
    mapping: dict = {}
    ids = []
    for k in keys:
        if k not in mapping:
            mapping[k] = len(mapping)
        ids.append(mapping[k])
    return np.array(ids, dtype=np.intp), len(mapping)


def add_team_rating(
    df: pl.DataFrame,
    column_names: ColumnNames,  # ColumnNames
    player_rating_col: str,
    team_rating_out: str,
) -> pl.DataFrame:
    mid = column_names.match_id
    tid = column_names.team_id

    if len(df) < _NUMPY_THRESHOLD:
        mid_vals = df[mid].to_list()
        tid_vals = df[tid].to_list()
        rating_arr = df[player_rating_col].to_numpy(allow_copy=True)
        keys, n_groups = _group_ids(list(zip(mid_vals, tid_vals, strict=False)))
        nan_mask = np.isnan(rating_arr)
        r_safe = np.where(nan_mask, 0.0, rating_arr)
        count = np.bincount(keys, weights=(~nan_mask).astype(float), minlength=n_groups)[keys]
        r_sum = np.bincount(keys, weights=r_safe, minlength=n_groups)[keys]
        result = np.where(count > 0, r_sum / count, np.nan)
        return df.with_columns(pl.Series(name=team_rating_out, values=result))

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
        if len(df) < _NUMPY_THRESHOLD:
            mid_vals = df[mid].to_list()
            tid_vals = df[tid].to_list()
            rating_arr = df[player_rating_col].to_numpy(allow_copy=True)
            weight_arr = df[weight_col].to_numpy(allow_copy=True)
            keys, n_groups = _group_ids(list(zip(mid_vals, tid_vals, strict=False)))
            # Treat NaN weight or rating as 0 contribution
            valid = ~(np.isnan(weight_arr) | np.isnan(rating_arr))
            w_safe = np.where(valid, weight_arr, 0.0)
            wr_safe = np.where(valid, weight_arr * rating_arr, 0.0)
            w_sum = np.bincount(keys, weights=w_safe, minlength=n_groups)[keys]
            wr_sum = np.bincount(keys, weights=wr_safe, minlength=n_groups)[keys]
            result = np.where(w_sum > 0, wr_sum / w_sum, np.nan)
            return df.with_columns(pl.Series(name=team_rating_out, values=result))

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

    if len(df) < _NUMPY_THRESHOLD:
        mid_vals = df[mid].to_list()
        tid_vals = df[tid].to_list()
        rating_vals = df[team_rating_col].to_list()

        # Build (match, team) → rating and match → set of teams mappings
        team_rating_map: dict = {}
        match_teams: dict = {}
        for m, t, r in zip(mid_vals, tid_vals, rating_vals, strict=False):
            team_rating_map[(m, t)] = r
            if m not in match_teams:
                match_teams[m] = []
            if t not in match_teams[m]:
                match_teams[m].append(t)

        opp_ratings = []
        for m, t in zip(mid_vals, tid_vals, strict=False):
            opp_teams = [ot for ot in match_teams.get(m, []) if ot != t]
            opp_ratings.append(team_rating_map.get((m, opp_teams[0])) if opp_teams else None)

        return df.with_columns(
            pl.Series(name=opp_team_rating_out, values=opp_ratings, dtype=pl.Float64)
        )

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


def add_player_opponent_mean_projected(
    df: pl.DataFrame,
    column_names: ColumnNames,
    player_rating_col: str,
    opp_team_rating_col: str,
    out_col: str,
) -> pl.DataFrame:
    """Mean of player rating and opponent team rating."""
    return df.with_columns(
        ((pl.col(player_rating_col) + pl.col(opp_team_rating_col)) / 2).alias(out_col)
    )
