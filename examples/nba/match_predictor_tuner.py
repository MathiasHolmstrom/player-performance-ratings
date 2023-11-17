from sklearn.preprocessing import StandardScaler

from examples.utils import load_nba_game_player_data
from player_performance_ratings import ColumnNames, MatchPredictorTuner, PreTransformerTuner
from player_performance_ratings import MatchPredictor
from player_performance_ratings import SKLearnClassifierWrapper
from player_performance_ratings import PlayerRatingGenerator
from player_performance_ratings import TeamRatingGenerator
from player_performance_ratings import RatingGenerator
from player_performance_ratings import ParameterSearchRange
from player_performance_ratings import PlayerRatingTuner
from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.ratings.enums import RatingColumnNames

from player_performance_ratings.transformers.common import SkLearnTransformerWrapper, MinMaxTransformer, \
    ColumnsWeighter

column_names = ColumnNames(
    team_id='TEAM_ID',
    match_id='GAME_ID',
    start_date="START_DATE",
    player_id="PLAYER_NAME",
    performance="performance",

)
df = load_nba_game_player_data()
df[PredictColumnNames.TARGET] = df['WON']
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])

df = (
    df.assign(team_count=df.groupby(column_names.match_id)[column_names.team_id].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

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

features = ["SCORE", "SCORE_OPPONENT"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)
pre_transformers = [
    standard_scaler,
]

duration_performance_search_range = []
column_weigher_search_range = [
    ParameterSearchRange(
        name='SCORE',
        type='uniform',
        low=0.4,
        high=0.6
    ),
    ParameterSearchRange(
        name='SCORE_OPPONENT',
        type='uniform',
        low=0.4,
        high=0.6,
        custom_params={'lower_is_better': True}
    ),

]

features = ["SCORE", "SCORE_OPPONENT"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

pre_transformer_search_ranges = [
    (standard_scaler, []),
    (MinMaxTransformer(features=features), []),
    (
        ColumnsWeighter(weighted_column_name=column_names.performance, column_weights=[]),
        column_weigher_search_range),
]

team_rating_generator = TeamRatingGenerator(
    player_rating_generator=PlayerRatingGenerator())
rating_generator = RatingGenerator()
predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.RATING_DIFFERENCE],
                                     granularity=[column_names.match_id, column_names.team_id])
match_predictor = MatchPredictor(
    rating_generator=rating_generator,
    column_names=column_names,
    predictor=predictor,
    pre_rating_transformers=pre_transformers,
)

pre_transformer_tuner = PreTransformerTuner(match_predictor=match_predictor,
                                            pre_transformer_search_ranges=pre_transformer_search_ranges,
                                            n_trials=15
                                            )

player_rating_tuner = PlayerRatingTuner(match_predictor=match_predictor,
                                        search_ranges=player_search_ranges,
                                        n_trials=20
                                        )

tuner = MatchPredictorTuner(
    pre_transformer_tuner=pre_transformer_tuner,
    player_rating_tuner=player_rating_tuner,
    fit_best=True,
)
best_match_predictor = tuner.tune(df=df)
tuner.tune(df=df)
