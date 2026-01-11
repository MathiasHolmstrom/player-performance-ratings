import pandas as pd
import polars as pl

df = pd.read_parquet(r"C:\Users\Admin\Downloads\stats_team_week_2024.parquet")
df["score"] = (
    df["passing_tds"] * 6
    + df["rushing_tds"] * 6
    + df["pat_made"]
    + df["fg_made"] * 3
    + df["rushing_2pt_conversions"] * 2
    + df["passing_2pt_conversions"] * 2
)
df["game_id"] = df["week"].astype(str) + df["team"] + df["opponent_team"]
opp_df = df[["game_id", "team", "opponent_team", "week", "score"]]
opp_df = opp_df.rename(columns={"score": "opponent_score"})
opp_df["opp_game_id"] = opp_df["week"].astype(str) + opp_df["opponent_team"] + opp_df["team"]
df = df.merge(opp_df[["opp_game_id", "opponent_score"]], left_on="game_id", right_on="opp_game_id")
gt = df.unique(["gameid", "teamname"]).select(["gameid", "teamname"])

df_with_opponent = gt.join(gt, on="gameid", how="left", suffix="_opp")
df_with_opponent = df_with_opponent.filter(pl.col("teamname") != pl.col("teamname_opp"))
df_with_opponent = df_with_opponent.with_columns(pl.col("teamname_opp").alias("opponent")).drop(
    "teamname_opp"
)
df = df.join(df_with_opponent, on=["gameid", "teamname"], how="left")

player_df = pd.read_csv(r"C:\Users\Admin\Downloads\stats_player_week_2024.csv")
u = 3
