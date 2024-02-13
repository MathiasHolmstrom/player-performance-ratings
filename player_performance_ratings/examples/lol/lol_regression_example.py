import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from player_performance_ratings import ColumnNames

from player_performance_ratings.predictor import MatchPredictor
from player_performance_ratings.ratings import BayesianTimeWeightedRating, ColumnWeight

df = pd.read_parquet("data/subsample_lol_data")
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])
df['__target'] = df['kills']
df["__target"] = df["__target"].clip(0, 8)

df['kills_per_minute'] = df['kills'] / df['gamelength'] * 60

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)
column_names_kpm = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='kills_per_minute',
    league='league',
    position='position',
)

column_names_kills = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='kills',
    league='league',
    position='position'
)
time_weighed_rating_kills_per_minute = BayesianTimeWeightedRating(column_names=column_names_kpm)
time_weighed_rating_kills = BayesianTimeWeightedRating(column_names=column_names_kills)



match_predictor = MatchPredictor(
    rating_generators=[time_weighed_rating_kills_per_minute, time_weighed_rating_kills],
    use_auto_create_performance_calculator=True,
    estimator=LGBMRegressor(),
    column_weights=[[ColumnWeight(name='kills_per_minute', weight=1)], [ColumnWeight(name='kills', weight=1)]],
    other_categorical_features=["position"]

)

df_predictions = match_predictor.train(df=df)

print(mean_absolute_error(df_predictions['__target'], df_predictions[match_predictor._predictor.pred_column]))
