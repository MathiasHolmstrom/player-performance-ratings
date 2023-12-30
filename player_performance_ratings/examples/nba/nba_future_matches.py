import pandas as pd

from player_performance_ratings import ColumnNames, PredictColumnNames
from player_performance_ratings.predictor import MatchPredictor

df = pd.read_pickle(r"data/game_player_subsample.pickle")

df = (
    df.assign(team_count=df.groupby("game_id")["team_id"].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

df = df.sort_values(["start_date","game_id", "team_id", "player_id"])

historical_df = df[df['start_date'] < '2023-02-01']
future_df = df[df['start_date'] >= '2023-02-01'][['game_id', 'team_id', 'player_id', 'start_date']]
column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_id",
    performance="won",
)
match_predictor = MatchPredictor(
    column_names=column_names,
)
historical_df[PredictColumnNames.TARGET] = historical_df['won']
match_predictor.generate_historical(historical_df)

predicted_future_df = match_predictor.predict(future_df)
