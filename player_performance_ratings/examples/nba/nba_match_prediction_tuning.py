import pandas as pd

from player_performance_ratings.ratings import ColumnWeight
from player_performance_ratings.tuner.rating_generator_tuner import OpponentAdjustedRatingGeneratorTuner

from player_performance_ratings.scorer import LogLossScorer

from player_performance_ratings import ColumnNames

from player_performance_ratings.tuner.utils import ParameterSearchRange, get_default_team_rating_search_range
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

df = (
    df.assign(team_count=df.groupby('game_id')['team_id'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)
df = df.drop_duplicates(subset=['game_id', 'player_id'])
features = ["plus_minus_per_minute"]

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

estimator = SkLearnWrapper(
    VennAbersCalibrator(estimator=LogisticRegression(), inductive=True, cal_size=0.2, random_state=101))

rating_generator = OpponentAdjustedRatingGenerator(column_names=column_names)

rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
    team_rating_search_ranges=get_default_team_rating_search_range(),
    start_rating_search_ranges=start_rating_search_range,
)

column_weights = [ColumnWeight(name='plus_minus', weight=1)]


match_predictor_factory = MatchPredictorFactory(
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


tuner = MatchPredictorTuner(
    match_predictor_factory=match_predictor_factory,
    fit_best=True,
    scorer=LogLossScorer(pred_column=match_predictor_factory.predictor.pred_column),
    rating_generator_tuners=rating_generator_tuner,

)
best_match_predictor = tuner.tune(df=df)

