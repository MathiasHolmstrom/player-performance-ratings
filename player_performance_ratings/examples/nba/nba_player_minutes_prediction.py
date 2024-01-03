import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from player_performance_ratings.transformation import LagTransformer

from player_performance_ratings import ColumnNames, PredictColumnNames
from player_performance_ratings.predictor import MatchPredictor
from player_performance_ratings.transformation.post_transformers import NormalizerTransformer, \
    GameTeamMembersColumnsTransformer, SklearnPredictorTransformer

df = pd.read_pickle(r"data/game_player_subsample.pickle")
df = df.drop_duplicates(subset=['game_id', 'player_id'])
df = (
    df.assign(team_count=df.groupby("game_id")["team_id"].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)

df.loc[df['start_position'] != '', 'starting'] = 1
df.loc[df['start_position'] == '', 'starting'] = 0

df = df.sort_values(["start_date", "game_id", "team_id", "player_id"])
column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_id",
    performance=None,
)

lag_transformer = LagTransformer(
    features=["minutes"],
    lag_length=20,
    granularity=[column_names.player_id, "starting"],
    column_names=column_names,
)

post_rating_transformers = [
    LagTransformer(
        features=["minutes"],
        lag_length=20,
        granularity=[column_names.player_id, "starting"],
        column_names=column_names,
    ),
    SklearnPredictorTransformer(estimator=LGBMRegressor(), target=f"__target",
                                features=lag_transformer.features_out + ["starting"], feature_out=f"{PredictColumnNames.TARGET}_transform_prediction"),
    GameTeamMembersColumnsTransformer(column_names=column_names,
                                      features=[f"{PredictColumnNames.TARGET}_transform_prediction"],
                                      sort_by=["starting"],
                                      players_per_match_per_team_count=8)
]

post_prediction_transformers = [
    NormalizerTransformer(features=[f"{PredictColumnNames.TARGET}_prediction"],
                          granularity=[column_names.team_id, column_names.match_id],
                          create_target_as_mean=True)]

match_predictor = MatchPredictor(
    post_rating_transformers=post_rating_transformers,
    post_prediction_transformers=post_prediction_transformers,
    estimator=LGBMRegressor(),
    date_column_name=column_names.start_date,
    other_categorical_features=["starting"],
)

df[PredictColumnNames.TARGET] = df['minutes']
df = match_predictor.generate_historical(df)
test_df = df[df[column_names.start_date] > match_predictor.train_split_date]

print("MAE", abs(test_df['minutes'] - test_df[match_predictor.predictor.pred_column]).mean())
