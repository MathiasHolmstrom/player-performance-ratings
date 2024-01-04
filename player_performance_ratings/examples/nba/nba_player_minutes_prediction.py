import pickle

import pandas as pd
from lightgbm import LGBMRegressor
from player_performance_ratings.scorer.score import SklearnScorer
from sklearn.metrics import mean_absolute_error
from player_performance_ratings.tuner import MatchPredictorTuner

from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory

from player_performance_ratings.transformation import LagTransformer

from player_performance_ratings import ColumnNames, PredictColumnNames

from player_performance_ratings.transformation.post_transformers import NormalizerTransformer
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner
from player_performance_ratings.tuner.utils import get_default_lgbm_regressor_search_range_by_learning_rate

df = pd.read_pickle(r"data/game_player_full.pickle")
df = df.drop_duplicates(subset=['game_id', 'player_id'])
df = (
    df.assign(team_count=df.groupby("game_id")["team_id"].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)

df.loc[df['start_position'] != '', 'starting'] = 1
df.loc[df['start_position'] == '', 'starting'] = 0

df['is_playoff'] = 0
df.loc[
    (df['start_date'].between('2018-04-13', '2018-06-20')) |
    (df['start_date'].between('2019-04-13', '2019-06-27')) |
    (df['start_date'].between('2020-08-13', '2020-09-20')) |
    (df['start_date'].between('2021-05-20', '2021-07-22')) |
    (df['start_date'].between('2022-04-13', '2022-06-27')) |
    (df['start_date'].between('2023-04-13', '2023-06-27')), 'is_playoff'
] = 1

df[PredictColumnNames.TARGET] = df['minutes']

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

train_split_date = df.iloc[int(len(df) / 1.3)][column_names.start_date]

post_rating_transformers = [
    LagTransformer(
        features=["minutes"],
        lag_length=25,
        granularity=[column_names.player_id, "starting"],
        column_names=column_names,
        days_between_lags=[1, 2, 3, 10, 25]
    ),
    LagTransformer(
        features=["minutes"],
        lag_length=10,
        granularity=[column_names.player_id, "starting", "is_playoff"],
        column_names=column_names,
        days_between_lags=[1, 2, 4],
        prefix="is_playoff_lag_"
    ),
    LagTransformer(
        features=["minutes"],
        lag_length=1,
        granularity=[column_names.player_id, "starting", 'location'],
        column_names=column_names,
        prefix="starting_location_lag_"
    ),
    LagTransformer(
        features=[],
        lag_length=1,
        granularity=[column_names.player_id],
        column_names=column_names,
        prefix="future_lag",
        future_lag=True,
        days_between_lags=[2]
    ),
]

post_prediction_transformers = [
    NormalizerTransformer(features=[f"{PredictColumnNames.TARGET}_prediction"],
                          granularity=[column_names.team_id, column_names.match_id],
                          create_target_as_mean=True)]

match_predictor_factory = MatchPredictorFactory(
    post_rating_transformers=post_rating_transformers,
    post_prediction_transformers=post_prediction_transformers,
    estimator=LGBMRegressor(reg_alpha=1, learning_rate=0.02, verbose=-100),
    date_column_name=column_names.start_date,
    other_categorical_features=["starting", "is_playoff"],
)

predictor_tuner = PredictorTuner(
    search_ranges=get_default_lgbm_regressor_search_range_by_learning_rate(
        learning_rate=match_predictor_factory.estimator.learning_rate),
    n_trials=65,
    date_column_name=column_names.start_date,
)

match_tuner = MatchPredictorTuner(
    scorer=SklearnScorer(target=PredictColumnNames.TARGET,
                         scorer_function=mean_absolute_error,
                         pred_column=match_predictor_factory.predictor.pred_column),
    match_predictor_factory=match_predictor_factory,
    predictor_tuner=predictor_tuner
)

best_model = match_tuner.tune(df=df)

df_with_minutes_prediction = best_model.generate_historical(df=df)

pickle.dump(best_model, open("models/nba_minute_prediction", 'wb'))
df_with_minutes_prediction.to_pickle("data/game_player_full_with_minutes_prediction.pickle")
