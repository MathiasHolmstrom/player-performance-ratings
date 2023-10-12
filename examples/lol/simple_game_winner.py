import pandas as pd

from src.predictor.match_predictor import MatchPredictor
from src.ratings.data_structures import ColumnNames
from sklearn.metrics import log_loss

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
)
match_predictor = MatchPredictor(column_names=column_names)
df = match_predictor.generate(df)

print(log_loss(df['result'], df[match_predictor.predictor.prob_column_name]))
