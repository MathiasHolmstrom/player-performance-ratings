import pandas as pd
import polars as pl
import pytest

from spforge import ColumnNames
from spforge.ratings import (
    LeagueStartRatingOptimizer,
    PlayerRatingGenerator,
    TeamRatingGenerator,
)


def _player_df():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    rows = []
    for i, date in enumerate(dates):
        mid = f"M{i}"
        for player_idx in range(2):
            rows.append(
                {
                    "pid": f"A{player_idx}",
                    "tid": "TA",
                    "mid": mid,
                    "date": date,
                    "league": "LCK",
                    "perf": 0.4,
                }
            )
        for player_idx in range(2):
            rows.append(
                {
                    "pid": f"B{player_idx}",
                    "tid": "TB",
                    "mid": mid,
                    "date": date,
                    "league": "LEC",
                    "perf": 0.6,
                }
            )
    return pd.DataFrame(rows)


def _team_df():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    rows = []
    for i, date in enumerate(dates):
        mid = f"M{i}"
        rows.extend(
            [
                {
                    "tid": "TA",
                    "mid": mid,
                    "date": date,
                    "league": "LCK",
                    "perf": 0.4,
                },
                {
                    "tid": "TB",
                    "mid": mid,
                    "date": date,
                    "league": "LEC",
                    "perf": 0.6,
                },
            ]
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("use_polars", [False, True])
def test_league_start_rating_optimizer__adjusts_player_leagues(use_polars):
    cn = ColumnNames(
        player_id="pid",
        team_id="tid",
        match_id="mid",
        start_date="date",
        league="league",
    )
    df = _player_df()
    if use_polars:
        df = pl.from_pandas(df)
    generator = PlayerRatingGenerator(performance_column="perf", column_names=cn)
    optimizer = LeagueStartRatingOptimizer(
        rating_generator=generator,
        n_iterations=1,
        learning_rate=0.5,
        min_cross_region_rows=1,
    )

    result = optimizer.optimize(df)

    assert result.league_ratings["LCK"] < 1000
    assert result.league_ratings["LEC"] > 1000


@pytest.mark.parametrize("use_polars", [False, True])
def test_league_start_rating_optimizer__adjusts_team_leagues(use_polars):
    cn = ColumnNames(
        team_id="tid",
        match_id="mid",
        start_date="date",
        league="league",
    )
    df = _team_df()
    if use_polars:
        df = pl.from_pandas(df)
    generator = TeamRatingGenerator(performance_column="perf", column_names=cn)
    optimizer = LeagueStartRatingOptimizer(
        rating_generator=generator,
        n_iterations=1,
        learning_rate=0.5,
        min_cross_region_rows=1,
    )

    result = optimizer.optimize(df)

    assert result.league_ratings["LCK"] < 1000
    assert result.league_ratings["LEC"] > 1000
