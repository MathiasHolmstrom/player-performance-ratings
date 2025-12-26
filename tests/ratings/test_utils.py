import polars as pl
import math

from spforge import ColumnNames
from spforge.ratings.utils import add_team_rating_projected, add_opp_team_rating, \
    add_opponent_rating_projected, add_rating_difference_projected, add_rating_mean_projected


def _base_df():
    # match 1: 2 players per team
    # weights chosen so we can verify weighted means easily
    return pl.DataFrame(
        {
            "match_id": [1, 1, 1, 1],
            "team_id": ["A", "A", "B", "B"],
            "player_rating": [10.0, 20.0, 30.0, 50.0],
            "ppw": [1.0, 3.0, 2.0, 2.0],
        }
    )


def test_add_team_rating_projected_weighted():
    df = _base_df()
    cn = ColumnNames(
        team_id="team_id",
        match_id="match_id",
        start_date="start_date",  # not used here but required by dataclass
        projected_participation_weight="ppw",
    )

    out = add_team_rating_projected(
        df=df,
        column_names=cn,
        player_rating_col="player_rating",
        team_rating_out="TEAM_RATING_PROJECTED",
    )

    # team A weighted mean: (1*10 + 3*20) / (1+3) = 17.5
    got_a = (
        out.filter((pl.col("match_id") == 1) & (pl.col("team_id") == "A"))
        .select(pl.col("TEAM_RATING_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got_a, 17.5, rel_tol=1e-12)

    # team B weighted mean: (2*30 + 2*50) / (2+2) = 40
    got_b = (
        out.filter((pl.col("match_id") == 1) & (pl.col("team_id") == "B"))
        .select(pl.col("TEAM_RATING_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got_b, 40.0, rel_tol=1e-12)


def test_add_team_rating_projected_unweighted():
    df = _base_df().drop("ppw")
    cn = ColumnNames(
        team_id="team_id",
        match_id="match_id",
        start_date="start_date",
        projected_participation_weight=None,
    )

    out = add_team_rating_projected(
        df=df,
        column_names=cn,
        player_rating_col="player_rating",
        team_rating_out="TEAM_RATING_PROJECTED",
    )

    # team A unweighted mean: (10+20)/2 = 15
    got_a = (
        out.filter((pl.col("match_id") == 1) & (pl.col("team_id") == "A"))
        .select(pl.col("TEAM_RATING_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got_a, 15.0, rel_tol=1e-12)


def test_add_opp_team_rating_projected():
    df = _base_df()
    cn = ColumnNames(
        team_id="team_id",
        match_id="match_id",
        start_date="start_date",
        projected_participation_weight="ppw",
    )

    df = add_team_rating_projected(
        df=df,
        column_names=cn,
        player_rating_col="player_rating",
        team_rating_out="TEAM_RATING_PROJECTED",
    )

    df = add_opp_team_rating(
        df=df,
        column_names=cn,
        team_rating_col="TEAM_RATING_PROJECTED",
        opp_team_rating_out="OPP_TEAM_RATING_PROJECTED",
    )

    # For team A, opponent is B => opponent team rating is 40.0
    got_a = (
        df.filter((pl.col("match_id") == 1) & (pl.col("team_id") == "A"))
        .select(pl.col("OPP_TEAM_RATING_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got_a, 40.0, rel_tol=1e-12)

    # For team B, opponent is A => opponent team rating is 17.5
    got_b = (
        df.filter((pl.col("match_id") == 1) & (pl.col("team_id") == "B"))
        .select(pl.col("OPP_TEAM_RATING_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got_b, 17.5, rel_tol=1e-12)


def test_add_opponent_rating_projected_alias():
    df = _base_df()
    cn = ColumnNames(
        team_id="team_id",
        match_id="match_id",
        start_date="start_date",
        projected_participation_weight="ppw",
    )

    df = add_team_rating_projected(df, cn, "player_rating", "TEAM_RATING_PROJECTED")
    df = add_opp_team_rating(df, cn, "TEAM_RATING_PROJECTED", "OPP_TEAM_RATING_PROJECTED")
    df = add_opponent_rating_projected(df, "OPP_TEAM_RATING_PROJECTED", "OPPONENT_RATING_PROJECTED")

    got = (
        df.filter((pl.col("match_id") == 1) & (pl.col("team_id") == "A"))
        .select(pl.col("OPPONENT_RATING_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got, 40.0, rel_tol=1e-12)


def test_add_rating_difference_projected():
    df = _base_df()
    cn = ColumnNames(
        team_id="team_id",
        match_id="match_id",
        start_date="start_date",
        projected_participation_weight="ppw",
    )

    df = add_team_rating_projected(df, cn, "player_rating", "TEAM_RATING_PROJECTED")
    df = add_opp_team_rating(df, cn, "TEAM_RATING_PROJECTED", "OPP_TEAM_RATING_PROJECTED")
    df = add_rating_difference_projected(
        df,
        team_rating_col="TEAM_RATING_PROJECTED",
        opp_team_rating_col="OPP_TEAM_RATING_PROJECTED",
        rating_diff_out="RATING_DIFFERENCE_PROJECTED",
    )

    # team A diff: 17.5 - 40 = -22.5
    got = (
        df.filter((pl.col("match_id") == 1) & (pl.col("team_id") == "A"))
        .select(pl.col("RATING_DIFFERENCE_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got, -22.5, rel_tol=1e-12)


def test_add_rating_mean_projected_weighted_entire_match():
    df = _base_df()
    cn = ColumnNames(
        team_id="team_id",
        match_id="match_id",
        start_date="start_date",
        projected_participation_weight="ppw",
    )

    out = add_rating_mean_projected(
        df=df,
        column_names=cn,
        player_rating_col="player_rating",
        rating_mean_out="RATING_MEAN_PROJECTED",
    )

    # match 1 weighted mean across ALL players:
    # (1*10 + 3*20 + 2*30 + 2*50) / (1+3+2+2) = 230/8 = 28.75
    got = (
        out.filter(pl.col("match_id") == 1)
        .select(pl.col("RATING_MEAN_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got, 28.75, rel_tol=1e-12)


def test_add_rating_mean_projected_unweighted_entire_match():
    df = _base_df().drop("ppw")
    cn = ColumnNames(
        team_id="team_id",
        match_id="match_id",
        start_date="start_date",
        projected_participation_weight=None,
    )

    out = add_rating_mean_projected(
        df=df,
        column_names=cn,
        player_rating_col="player_rating",
        rating_mean_out="RATING_MEAN_PROJECTED",
    )

    # match 1 unweighted mean across ALL players: (10+20+30+50)/4 = 27.5
    got = (
        out.filter(pl.col("match_id") == 1)
        .select(pl.col("RATING_MEAN_PROJECTED").unique())
        .item()
    )
    assert math.isclose(got, 27.5, rel_tol=1e-12)