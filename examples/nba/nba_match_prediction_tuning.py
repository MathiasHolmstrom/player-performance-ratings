from sklearn.preprocessing import StandardScaler

from examples.utils import load_nba_game_player_data, load_nba_game_matchup_data
from player_performance_ratings import ColumnNames, MatchPredictorTuner, PreTransformerTuner, StartRatingTuner
from player_performance_ratings import MatchPredictor
from player_performance_ratings import SKLearnClassifierWrapper
from player_performance_ratings import RatingGenerator
from player_performance_ratings import ParameterSearchRange
from player_performance_ratings import PlayerRatingTuner
from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.predictor.ml_wrappers.classifier import SkLearnGamePredictor
from player_performance_ratings.ratings.enums import RatingColumnNames

from player_performance_ratings.transformers.common import SkLearnTransformerWrapper, MinMaxTransformer, \
    ColumnsWeighter, ColumnWeight

column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_id",
    performance="performance",
    participation_weight="participation_weight",

)
df = load_nba_game_player_data()
df[PredictColumnNames.TARGET] = df['won']
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])
df['plus_minus_per_minute'] = df['plus_minus'] / df['game_minutes']

df = (
    df.assign(team_count=df.groupby('game_id')['team_id'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)

start_rating_search_range = [
    ParameterSearchRange(
        name='league_quantile',
        type='uniform',
        low=0.12,
        high=.4,
    ),
    ParameterSearchRange(
        name='min_count_for_percentiles',
        type='discrete_uniform',
        low=50,
        high=200,
    ),
    ParameterSearchRange(
        name='team_rating_subtract',
        type='int',
        low=20,
        high=300
    ),
    ParameterSearchRange(
        name='team_weight',
        type='uniform',
        low=0,
        high=0.7
    )
]

player_search_ranges = [
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
        high=150,
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
        low=15,
        high=50
    ),
    ParameterSearchRange(
        name='reference_certain_sum_value',
        type='uniform',
        low=0.5,
        high=5
    ),
    ParameterSearchRange(
        name='rating_change_multiplier',
        type='uniform',
        low=40,
        high=240
    ),
]

features = ["plus_minus_per_minute"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)
pre_transformers = [
    standard_scaler,
    MinMaxTransformer(features=features),
    ColumnsWeighter(weighted_column_name=column_names.performance, column_weights=[
        ColumnWeight(name='plus_minus_per_minute', weight=1)
    ])
]


rating_generator = RatingGenerator()
predictor = SkLearnGamePredictor(features=[RatingColumnNames.RATING_DIFFERENCE],game_id_colum='game_id', team_id_column='team_id')
match_predictor = MatchPredictor(
    rating_generator=rating_generator,
    column_names=column_names,
    predictor=predictor,
    pre_rating_transformers=pre_transformers,
)


player_rating_tuner = PlayerRatingTuner(match_predictor=match_predictor,
                                        search_ranges=player_search_ranges,
                                        n_trials=25
                                        )

start_rating_tuner = StartRatingTuner(column_names=column_names,
                                      match_predictor=match_predictor,
                                      n_trials=25,
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
