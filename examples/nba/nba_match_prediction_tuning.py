from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from venn_abers import VennAbersCalibrator

from examples.utils import load_nba_game_player_data, load_nba_game_matchup_data
from player_performance_ratings import ColumnNames, MatchPredictorTuner, StartRatingTuner, \
    TeamRatingTuner
from player_performance_ratings import MatchPredictor
from player_performance_ratings import RatingGenerator
from player_performance_ratings import ParameterSearchRange

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor.estimators.classifier import SkLearnGameTeamPredictor
from player_performance_ratings.predictor.estimators.sklearn_models import SkLearnWrapper
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.match_rating import TeamRatingGenerator
from player_performance_ratings.ratings.match_rating.performance_predictor import PerformancePredictor, \
    RatingDifferencePerformancePredictor
from player_performance_ratings.ratings.match_rating.start_rating.start_rating_generator import StartRatingGenerator

from player_performance_ratings.preprocessing.common import SkLearnTransformerWrapper, MinMaxTransformer


column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_id",
    performance="plus_minus_per_minute",
    participation_weight="participation_weight",

)
gm = load_nba_game_matchup_data()
df = load_nba_game_player_data()

df = df[df['game_id'].isin(gm['game_id'].unique().tolist())]


print(len(df['game_id'].unique()))
df[PredictColumnNames.TARGET] = df['won']
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])
df = df[df['game_minutes']>46]
df['plus_minus_per_minute'] = df['plus_minus'] / df['game_minutes']
df['participation_weight'] = df['minutes'] / df['game_minutes']

df = (
    df.assign(team_count=df.groupby('game_id')['team_id'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)


features = ["plus_minus_per_minute"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)
pre_transformers = [
    standard_scaler,
    MinMaxTransformer(features=features),
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


rating_generator = RatingGenerator(
    store_game_ratings=True,
    column_names=column_names,
    team_rating_generator=TeamRatingGenerator(
        performance_predictor=RatingDifferencePerformancePredictor(
            team_rating_diff_coef=0,
        ),
        start_rating_generator=StartRatingGenerator(
            team_weight=0,
        )
    )
)

estimator = SkLearnWrapper(VennAbersCalibrator(estimator=LogisticRegression(), inductive=True, cal_size=0.2, random_state=101))

predictor = SkLearnGameTeamPredictor(model=estimator,features=[RatingColumnNames.RATING_DIFFERENCE], game_id_colum='game_id',
                                     team_id_column='team_id', weight_column='participation_weight')
match_predictor = MatchPredictor(
    rating_generator=rating_generator,
    column_names=column_names,
    predictor=predictor,
    pre_rating_transformers=pre_transformers,
    train_split_date="2022-05-01"
)


player_rating_tuner = TeamRatingTuner(match_predictor=match_predictor,
                                        n_trials=25
                                        )

start_rating_tuner = StartRatingTuner(column_names=column_names,
                                      match_predictor=match_predictor,
                                      n_trials=8,
                                      search_ranges=start_rating_search_range,
                                      )

tuner = MatchPredictorTuner(
    # pre_transformer_tuner=pre_transformer_tuner,
    team_rating_tuner=player_rating_tuner,
    start_rating_tuner=start_rating_tuner,
    fit_best=True,
)
best_match_predictor = tuner.tune(df=df)
tuner.tune(df=df)
