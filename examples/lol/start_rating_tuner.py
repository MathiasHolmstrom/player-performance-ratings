import pandas as pd

from src.auto_predictor.auto_predictor import StartRatingTuner
from src.ratings.data_structures import ColumnNames

df = pd.read_csv("data/2023_LoL.csv")
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance="result",
    league='league'
)

start_rating_tuner = StartRatingTuner(column_names=column_names)
start_rating_tuner.tune(df=df)
