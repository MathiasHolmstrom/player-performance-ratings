import pandas as pd

df = pd.read_pickle("game_player.pickle")
df = df.drop_duplicates(subset=["game_id", "player_id"])
gt = pd.read_pickle("game_team.pickle")
game = pd.read_pickle("game.pickle")
possessions = pd.read_pickle("possessions.pickle")
lineups = pd.read_pickle("lineups.pickle")

possessions = possessions.merge(
    lineups.rename(columns={"lineup": "lineup_offense"}),
    left_on=["lineup_id_offense"],
    right_on=["lineup_id"],
    how="left",
)
possessions = possessions.merge(
    lineups.rename(columns={"lineup": "lineup_defense"}),
    left_on=["lineup_id_defense"],
    right_on=["lineup_id"],
    how="left",
)

exploded = possessions.explode(["lineup_offense"]).rename(
    columns={"lineup_offense": "player_id"}
)
exploded["net_seconds"] = (
    exploded["seconds_played_end"] - exploded["seconds_played_start"]
)
grp = (
    exploded.groupby(["game_id", "player_id"])["points", "net_seconds"]
    .sum()
    .reset_index()
)


q2 = possessions[possessions["game_id"] == "0011900062"]

game = game.rename(columns={"minutes": "game_minutes"})
gt = gt.rename(columns={"pace": "team_pace", "possessions": "team_possessions"})
df = df.merge(game[["game_id", "game_minutes", "start_date"]], on="game_id", how="left")
df = df.merge(
    gt[
        [
            "game_id",
            "team_id",
            "team_id_opponent",
            "score",
            "score_opponent",
            "won",
            "location",
            "team_possessions",
            "team_pace",
        ]
    ],
    on=["game_id", "team_id"],
    how="inner",
)
df.loc[df["won"] == True, "won"] = 1
df.loc[df["won"] == False, "won"] = 0

# randomly select 10% of game_ids

df = df[
    [
        "team_id",
        "start_date",
        "game_id",
        "player_id",
        "player_name",
        "start_position",
        "team_id_opponent",
        "points",
        "game_minutes",
        "pace",
        "possessions",
        "minutes",
        "won",
        "plus_minus",
        "location",
        "score",
        "score_opponent",
        "team_possessions",
        "team_pace",
    ]
]
# df = df[df['start_date'].between('2022-10-17', '2023-02-01')]

df.to_pickle("game_player_full.pickle")
