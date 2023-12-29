import joblib



from sklearn.metrics import log_loss

from player_performance_ratings import PredictColumnNames
from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor.match_predictor import MatchPredictor

from player_performance_ratings.ratings import OpponentAdjustedRatingGenerator

column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_name",
    performance="won",
)
df = load_nba_game_player_data()
df[PredictColumnNames.TARGET] = df['won']
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])

df = (
    df.assign(team_count=df.granularity(column_names.match_id)[column_names.team_id].fit_transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)


rating_generator = OpponentAdjustedRatingGenerator()

match_predictor = MatchPredictor(column_names=column_names, rating_generators=rating_generator)
df = match_predictor.generate_historical(df)

print(log_loss(df[PredictColumnNames.TARGET], df[match_predictor.predictor.pred_column]))

joblib.dump(match_predictor, 'match_predictor.pkl')

