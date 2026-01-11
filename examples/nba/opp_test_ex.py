import pandas as pd

from spforge.transformers import RollingAgainstOpponentTransformer

df = pd.read_parquet("data/game_player_subsample.parquet")
df["points_per_minute"] = df["points"] / df["minutes"]
df["is_starting"] = (df["start_position"] != "").astype(int)

opp_transformer = RollingAgainstOpponentTransformer(
    granularity=["start_position"],
    features=["points_per_minute"],
    window=2,
    update_column="game_id",
    team_column="team_id",
)
df = opp_transformer.transform_historical(df)
v = df[(df["team_id_opponent"] == 1610612738) & (df["start_position"] == "C")][
    ["start_date", *opp_transformer.features_out, "points_per_minute"]
]
u = 2
