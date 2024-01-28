import pandas as pd
from lightgbm import LGBMClassifier


from player_performance_ratings.ratings import ColumnWeight

from player_performance_ratings import ColumnNames, PredictColumnNames, PipelineFactory
from player_performance_ratings.ratings.rating_calculators import OpponentAdjustedRatingGenerator

df = pd.read_pickle(r"data/game_player_subsample.pickle")

df = (
    df.assign(team_count=df.groupby("game_id")["team_id"].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

df = df.sort_values(["start_date", "game_id", "team_id", "player_id"])
df["max_minutes"] = df.groupby(["game_id", "team_id"])["game_minutes"].transform('max')
df['participation_weight'] = df['minutes'] / df['max_minutes']
df.drop(columns=['max_minutes'], inplace=True)

historical_df = df[df['start_date'] < '2023-02-01']
future_df = df[df['start_date'] >= '2023-02-01'][['game_id', 'team_id', 'player_id', 'start_date', 'participation_weight', "location"]]

column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_id",
    performance="plus_minus",
    participation_weight="participation_weight",
)

match_predictor = PipelineFactory(
    rating_generators=[OpponentAdjustedRatingGenerator(column_names=column_names)],
    use_auto_create_performance_calculator=True,
    column_weights=[ColumnWeight(name='plus_minus', weight=1)],
    estimator=LGBMClassifier(),
    group_predictor_by_game_team = True,
    other_categorical_features=["location"]
)

historical_df[PredictColumnNames.TARGET] = historical_df['won']
match_predictor.generate_historical(historical_df)

predicted_future_df = match_predictor.predict(future_df)
