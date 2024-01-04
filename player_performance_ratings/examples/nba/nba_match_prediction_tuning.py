import pickle

import pandas as pd
from lightgbm import LGBMClassifier

from player_performance_ratings.transformation import LagTransformer

from player_performance_ratings.ratings import ColumnWeight
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner
from player_performance_ratings.tuner.rating_generator_tuner import OpponentAdjustedRatingGeneratorTuner

from player_performance_ratings.scorer import LogLossScorer

from player_performance_ratings import ColumnNames

from player_performance_ratings.tuner.utils import ParameterSearchRange, get_default_team_rating_search_range, \
    get_default_lgbm_classifier_search_range_by_learning_rate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from venn_abers import VennAbersCalibrator

from player_performance_ratings.consts import PredictColumnNames

from player_performance_ratings.predictor.estimators.sklearn_models import SkLearnWrapper
from player_performance_ratings.ratings.enums import RatingColumnNames

from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import OpponentAdjustedRatingGenerator
from player_performance_ratings.tuner import MatchPredictorTuner
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory

column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_id",
    performance="plus_minus_per_minute",
    participation_weight="participation_weight",
    projected_participation_weight="projected_participation_weight",

)

df = pd.read_pickle("data/game_player_full_with_minutes_prediction.pickle")

# df = df[df['game_id'].isin(gm['game_id'].unique().tolist())]


df[PredictColumnNames.TARGET] = df['won']
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])
df = df[df['game_minutes'] > 46]
df = df.rename(columns={'__target_prediction': 'minutes_prediction'})
df['plus_minus_per_minute'] = df['plus_minus'] / df['game_minutes']
df['participation_weight'] = df['minutes'] / df['game_minutes']
df['projected_participation_weight'] = df['minutes_prediction'] / df['game_minutes'].mean()
df['score_difference'] = df['score'] - df['score_opponent']

df = (
    df.assign(team_count=df.groupby('game_id')['team_id'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)
df = df.drop_duplicates(subset=['game_id', 'player_id'])

start_rating_search_range = [
    ParameterSearchRange(
        name='league_quantile',
        type='uniform',
        low=0.04,
        high=.4,
    ),
    ParameterSearchRange(
        name='min_count_for_percentiles',
        type='int',
        low=50,
        high=400,
    ),
]
df.loc[df['minutes'] > 0, 'plus_minus_per_minute'] = df['plus_minus'] / df['minutes']
df.loc[df['minutes'] == 0, 'plus_minus_per_minute'] = 0

estimator = SkLearnWrapper(
    VennAbersCalibrator(
        estimator=LGBMClassifier(max_depth=2, learning_rate=0.1, n_estimators=300, verbose=-100, reg_alpha=1.2),
        inductive=True, cal_size=0.2, random_state=101))

rating_generator = OpponentAdjustedRatingGenerator(column_names=column_names)

column_weights = [ColumnWeight(name='plus_minus', weight=1)]

post_rating_transformers = [
    LagTransformer(
        features=["score_difference", RatingColumnNames.RATING_DIFFERENCE_PROJECTED],
        lag_length=10,
        granularity=[column_names.player_id],
        column_names=column_names,
        days_between_lags=[1, 2, 3, 4, 5]
    ),
    LagTransformer(
        features=[],
        lag_length=4,
        granularity=[column_names.player_id],
        column_names=column_names,
        prefix="future_lag",
        future_lag=True,
        days_between_lags=[1, 2, 3, 4]
    ),
    LagTransformer(
        features=["score_difference", RatingColumnNames.RATING_DIFFERENCE_PROJECTED],
        lag_length=8,
        granularity=[column_names.player_id, 'location'],
        column_names=column_names,
        days_between_lags=[1, 2, 3, 4, 5],
        prefix="location_lag"
    ),
]

match_predictor_factory = MatchPredictorFactory(
    post_rating_transformers=post_rating_transformers,
    use_auto_create_performance_calculator=True,
    rating_generators=rating_generator,
    other_categorical_features=["location"],
    estimator=estimator,
    date_column_name=column_names.start_date,
    column_weights=column_weights,
    group_predictor_by_game_team=True,
    team_id_column_name=column_names.team_id,
    match_id_column_name=column_names.match_id,
)

rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
    team_rating_search_ranges=get_default_team_rating_search_range(),
    start_rating_search_ranges=start_rating_search_range,
    team_rating_n_trials=30,
    start_rating_n_trials=8,
)
predictor_tuner = PredictorTuner(
    default_params={'learning_rate': 0.04},
    search_ranges=get_default_lgbm_classifier_search_range_by_learning_rate(learning_rate=0.04),
    n_trials=30,
    date_column_name=column_names.start_date,
)


tuner = MatchPredictorTuner(
    match_predictor_factory=match_predictor_factory,
    fit_best=True,
    scorer=LogLossScorer(pred_column=match_predictor_factory.predictor.pred_column),
    rating_generator_tuners=rating_generator_tuner,
)
best_match_predictor = tuner.tune(df=df)
df_with_minutes_prediction = best_match_predictor.generate_historical(df=df)

pickle.dump(best_match_predictor, open("models/nba_game_winner", 'wb'))
df_with_minutes_prediction.to_pickle("data/game_player_predictions.pickle")