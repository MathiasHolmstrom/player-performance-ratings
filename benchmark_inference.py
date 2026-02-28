"""
Benchmark numpy vs polars fast paths in ratings/utils.py
for small DataFrames (inference-time workloads).

Run: .venv/bin/python benchmark_inference.py
"""

import time

import numpy as np
import polars as pl

import spforge.ratings.utils as _utils
from spforge.data_structures import ColumnNames
from spforge.ratings.utils import (
    add_opp_team_rating,
    add_team_rating_projected,
)

# ──────────────────────────────────────────────────────────
# Fake ColumnNames
# ──────────────────────────────────────────────────────────
cn = ColumnNames(
    match_id="match_id",
    team_id="team_id",
    player_id="player_id",
    start_date="start_date",
    projected_participation_weight="ppw",
)


# ──────────────────────────────────────────────────────────
# Build a realistic inference-size DataFrame
# 2 teams × 10 players each = 20 rows (single game scenario)
# ──────────────────────────────────────────────────────────
def make_df(n_games: int = 1, players_per_team: int = 10) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for g in range(n_games):
        for t, tid in enumerate(["T1", "T2"]):
            for p in range(players_per_team):
                rows.append(
                    {
                        "match_id": f"M{g}",
                        "team_id": tid,
                        "player_id": f"P{g}_{t}_{p}",
                        "start_date": "2024-01-01",
                        "ppw": float(rng.uniform(0.5, 1.0)),
                        "player_off_rating": float(rng.uniform(1400, 1600)),
                        "team_def_rating_projected": float(rng.uniform(1450, 1550)),
                    }
                )
    return pl.DataFrame(rows)


def bench(label: str, fn, n: int = 1000) -> float:  # noqa: ANN001
    # warmup
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    ms = elapsed / n * 1000
    print(f"  {label:50s} {ms:.3f} ms/call")
    return ms


# ──────────────────────────────────────────────────────────
# Benchmark: add_team_rating_projected (small = numpy path)
# ──────────────────────────────────────────────────────────
print("\n=== add_team_rating_projected (auto dispatch) ===")
for n_games in [1, 4, 10]:
    _df = make_df(n_games=n_games)
    bench(
        f"n_rows={len(_df)}",
        lambda df=_df: add_team_rating_projected(
            df, cn, "player_off_rating", "team_off_rating_projected"
        ),
    )


# ──────────────────────────────────────────────────────────
# Force comparison: patch threshold to force polars vs numpy
# ──────────────────────────────────────────────────────────
print("\n=== add_team_rating_projected: polars vs numpy ===")
for n_games in [1, 4]:
    _df = make_df(n_games=n_games)
    n_rows = len(_df)

    _utils._NUMPY_THRESHOLD = 0
    t_polars = bench(
        f"n_rows={n_rows} POLARS forced",
        lambda df=_df: add_team_rating_projected(
            df, cn, "player_off_rating", "team_off_rating_projected"
        ),
    )

    _utils._NUMPY_THRESHOLD = 9999
    t_numpy = bench(
        f"n_rows={n_rows} NUMPY  forced",
        lambda df=_df: add_team_rating_projected(
            df, cn, "player_off_rating", "team_off_rating_projected"
        ),
    )

    print(f"    → numpy speedup: {t_polars / t_numpy:.1f}x")


# ──────────────────────────────────────────────────────────
# Benchmark: add_opp_team_rating
# ──────────────────────────────────────────────────────────
print("\n=== add_opp_team_rating: polars vs numpy ===")
for n_games in [1, 4, 10]:
    _df2 = make_df(n_games=n_games).with_columns(
        pl.col("player_off_rating")
        .mean()
        .over(["match_id", "team_id"])
        .alias("team_off_rating_projected")
    )

    _utils._NUMPY_THRESHOLD = 0
    t_polars = bench(
        f"n_rows={len(_df2)} POLARS",
        lambda df=_df2: add_opp_team_rating(df, cn, "team_off_rating_projected", "opp_rating"),
    )

    _utils._NUMPY_THRESHOLD = 9999
    t_numpy = bench(
        f"n_rows={len(_df2)} NUMPY ",
        lambda df=_df2: add_opp_team_rating(df, cn, "team_off_rating_projected", "opp_rating"),
    )

    print(f"    → numpy speedup: {t_polars / t_numpy:.1f}x")


# restore
_utils._NUMPY_THRESHOLD = 500
print("\nDone.")
