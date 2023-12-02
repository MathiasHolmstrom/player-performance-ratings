import numpy as np
from sklearn.preprocessing import StandardScaler

from examples.utils import load_nba_game_matchup_data

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings import RatingGenerator, SkLearnTransformerWrapper, \
    MinMaxTransformer, TeamRatingTuner, StartRatingTuner, MatchPredictorTuner, ParameterSearchRange
from player_performance_ratings.predictor.estimators.classifier import SkLearnGameTeamPredictor
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.match_rating import TeamRatingGenerator
from player_performance_ratings.ratings.match_rating.performance_predictor import RatingDifferencePerformancePredictor
from player_performance_ratings.ratings.match_rating.start_rating.start_rating_generator import StartRatingGenerator

column_names = ColumnNames(
    team_id='lineup_id',
    match_id='matchup_game_id',
    start_date="start_date",
    player_id="player_id",
    performance="plus_minus_per_minute",
    participation_weight="participation_weight",
    rating_update_id="game_id"

)
df = load_nba_game_matchup_data()
df.loc[df['points'] > df['points_opponent'], column_names.performance] = 1
df.loc[df['points'] < df['points_opponent'], column_names.performance] = 0
df.loc[df['points'] == df['points_opponent'], column_names.performance] = 0.5
df[PredictColumnNames.TARGET] = df['won']

min_lineup = np.minimum(df['lineup_id'], df['lineup_id_opponent'])
max_lineup = np.maximum(df['lineup_id'], df['lineup_id_opponent'])

# Combine the min and max values into a tuple
df['sorted_lineup'] = list(zip(min_lineup, max_lineup))
df = df[df['game_minutes'] > 46]

df[column_names.match_id] = df['game_id'].astype(str) + '_' + df['sorted_lineup'].astype(str)
df = df.drop(columns=['sorted_lineup'])

df = (
    df.assign(team_count=df.groupby(column_names.match_id)[column_names.team_id].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])
df['plus_minus'] = df['points'] - df['points_opponent']
df['plus_minus_per_minute'] = df['plus_minus'] / df['minutes_lineup_matchup']
df.loc[df['minutes_lineup_matchup'] == 0, 'plus_minus_per_minute'] = 0
print(len(df['game_id'].unique()))
mean_participation_weight = df['participation_weight'].mean()
features = ['plus_minus_per_minute']

standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)
pre_transformer_search_ranges = [
    (standard_scaler, []),
    (MinMaxTransformer(features=features), [])
]

team_rating_search_ranges = [
    ParameterSearchRange(
        name='certain_weight',
        type='uniform',
        low=0.7,
        high=0.95
    ),
    ParameterSearchRange(
        name='certain_days_ago_multiplier',
        type='uniform',
        low=0.02,
        high=.12,
    ),
    ParameterSearchRange(
        name='max_days_ago',
        type='uniform',
        low=40,
        high=200,
    ),
    ParameterSearchRange(
        name='max_certain_sum',
        type='uniform',
        low=20,
        high=70,
    ),
    ParameterSearchRange(
        name='certain_value_denom',
        type='uniform',
        low=10,
        high=50
    ),
    ParameterSearchRange(
        name='reference_certain_sum_value',
        type='uniform',
        low=0.4,
        high=5
    ),
    ParameterSearchRange(
        name='rating_change_multiplier',
        type='uniform',
        low=35,
        high=140
    ),
    ParameterSearchRange(
        name='participation_weight_coef',
        type='uniform',
        low=0,
        high=10
    ),
    ParameterSearchRange(
        name='team_rating_diff_coef',
        type='uniform',
        low=0,
        high=0.002
    ),
]

performance_predictor = RatingDifferencePerformancePredictor(
    team_rating_diff_coef=0,
    rating_diff_coef=0.005757,
    participation_weight_coef=1,
    mean_participation_weight=mean_participation_weight

)

rating_generator = RatingGenerator(
    store_game_ratings=True,
    column_names=column_names,
    team_rating_generator=TeamRatingGenerator(
        performance_predictor=performance_predictor,
        start_rating_generator=StartRatingGenerator(
            team_weight=0,
        )
    )
)

pre_transformers = [
    standard_scaler,
    MinMaxTransformer(features=features, quantile=0.99),
]

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

predictor = SkLearnGameTeamPredictor(features=[RatingColumnNames.RATING_DIFFERENCE],
                                     weight_column='participation_weight',
                                     team_id_column='team_id', game_id_colum=column_names.rating_update_id,
                                     target='won')

match_predictor = MatchPredictor(column_names=column_names, rating_generator=rating_generator, predictor=predictor,
                                 pre_rating_transformers=pre_transformers, train_split_date="2022-05-01")

team_rating_tuner = TeamRatingTuner(match_predictor=match_predictor,
                                    n_trials=70,
                                    search_ranges=team_rating_search_ranges
                                    )

start_rating_tuner = StartRatingTuner(column_names=column_names,
                                      match_predictor=match_predictor,
                                      search_ranges=start_rating_search_range,
                                      n_trials=9,
                                      )

tuner = MatchPredictorTuner(
    team_rating_tuner=team_rating_tuner,
    start_rating_tuner=start_rating_tuner,
    fit_best=True,

)
best_match_predictor = tuner.tune(df=df)
tuner.tune(df=df)
