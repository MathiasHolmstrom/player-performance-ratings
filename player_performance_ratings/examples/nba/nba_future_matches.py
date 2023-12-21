import joblib

from player_performance_ratings.examples.utils import load_nba_game_player_data
from player_performance_ratings import MatchPredictor
from player_performance_ratings.consts import PredictColumnNames

match_predictor: MatchPredictor = joblib.load('match_predictor.pkl')
column_names = match_predictor.column_names
df = load_nba_game_player_data()
game_ids = df[match_predictor.column_names.match_id].unique().tolist()
df = df[df[match_predictor.column_names.match_id].isin(game_ids[len(game_ids)- 10:])]
df[PredictColumnNames.TARGET] = df['won']
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])

df = (
    df.assign(team_count=df.groupby(column_names.match_id)[column_names.team_id].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)


match_predictor.predict(df)