import numpy as np
from sklearn.preprocessing import StandardScaler

from player_performance_ratings.examples.utils import load_nba_game_matchup_data

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings import RatingGenerator, SkLearnTransformerWrapper, \
    MinMaxTransformer, TeamRatingTuner, StartRatingTuner, MatchPredictorTuner, ParameterSearchRange
from player_performance_ratings.predictor.estimators.classifier import SkLearnGameTeamPredictor
from player_performance_ratings.predictor.estimators.ordinal_classifier import OrdinalClassifier
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.match_rating import TeamRatingGenerator
from player_performance_ratings.ratings.opponent_adjusted_rating.performance_predictor import RatingMeanPerformancePredictor
from player_performance_ratings.ratings.opponent_adjusted_rating.start_rating_generator import StartRatingGenerator
from player_performance_ratings.scorer.score import OrdinalLossScorer

column_names = ColumnNames(
    team_id='lineup_id',
    match_id='matchup_game_id',
    start_date="start_date",
    player_id="player_id",
    performance="total_points_lineup_matchup_per_minute",
    participation_weight="participation_weight",
    rating_update_id="game_id"

)

df = load_nba_game_matchup_data()
df['total_score'] = df['score'] + df['score_opponent']

df = df[df['game_minutes'] > 46]

df.loc[df['total_score'] > 248, 'total_score'] = 248
df.loc[df['total_score'] < 207, 'total_score'] = 207

df[PredictColumnNames.TARGET] = df['total_score']

min_lineup = np.minimum(df['lineup_id'], df['lineup_id_opponent'])
max_lineup = np.maximum(df['lineup_id'], df['lineup_id_opponent'])

# Combine the min and max values into a tuple
df['sorted_lineup'] = list(zip(min_lineup, max_lineup))

df[column_names.match_id] = df['game_id'].astype(str) + '_' + df['sorted_lineup'].astype(str)
df = df.drop(columns=['sorted_lineup'])

df = (
    df.assign(team_count=df.granularity(column_names.match_id)[column_names.team_id].fit_transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])
df['total_points_lineup_matchup'] = df['points'] + df['points_opponent']
df['total_points_lineup_matchup_per_minute'] = df['total_points_lineup_matchup'] / df['minutes_lineup_matchup']
df.loc[df['minutes_lineup_matchup'] == 0, 'total_points_lineup_matchup_per_minute'] = 0

mean_participation_weight = df['participation_weight'].mean()
features = ['total_points_lineup_matchup_per_minute']

standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)
pre_transformer_search_ranges = [
    (standard_scaler, []),
    (MinMaxTransformer(features=features), [])
]

performance_predictor = RatingMeanPerformancePredictor(
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
        high=.5,
    ),
    ParameterSearchRange(
        name='min_count_for_percentiles',
        type='int',
        low=50,
        high=400,
    ),
]

predictor = SkLearnGameTeamPredictor(features=[RatingColumnNames.RATING_MEAN],
                                     weight_column='participation_weight',
                                     model=OrdinalClassifier(),
                                     team_id_column='team_id', game_id_colum=column_names.rating_update_id,
                                     target='total_score', multiclassifier=True)

scorer = OrdinalLossScorer(
    pred_column=predictor.pred_column,
    granularity=['game_id', 'team_id']
)

match_predictor = MatchPredictor(column_names=column_names, rating_generator=rating_generator, predictor=predictor,
                                 pre_rating_transformers=pre_transformers, train_split_date="2022-05-01")

team_rating_tuner = TeamRatingTuner(match_predictor=match_predictor,
                                    n_trials=35,
                                    scorer=scorer,
                                    )

start_rating_tuner = StartRatingTuner(column_names=column_names,
                                      match_predictor=match_predictor,
                                      search_ranges=start_rating_search_range,
                                      n_trials=9,
                                      scorer=scorer,
                                      )

tuner = MatchPredictorTuner(
    team_rating_tuner=team_rating_tuner,
    start_rating_tuner=start_rating_tuner,
    fit_best=True,

)
best_match_predictor = tuner.tune(df=df)
tuner.tune(df=df)
