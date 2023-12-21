import pickle

from sklearn.preprocessing import StandardScaler

from player_performance_ratings.examples.utils import load_lol_data
from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings import MatchPredictor, OpponentAdjustedRatingGenerator
from player_performance_ratings import SKLearnClassifierWrapper

from player_performance_ratings import TeamRatingGenerator

from player_performance_ratings import SkLearnTransformerWrapper, MinMaxTransformer
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.transformations.pre_transformers import ColumnsWeighter
from player_performance_ratings.tuner import TransformerTuner
from player_performance_ratings.tuner.utils import ParameterSearchRange

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='performance',
    league='league'
)
df = load_lol_data()
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

team_rating_generator = TeamRatingGenerator()
rating_generator = OpponentAdjustedRatingGenerator()
predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.RATING_DIFFERENCE], target='result')

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
        low=30,
        high=100
    )
]

features = ["result"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

pre_transformers = [
    standard_scaler,
]

duration_performance_search_range = []
column_weigher_search_range = [
    ParameterSearchRange(
        name='damagetochampions',
        type='uniform',
        low=0,
        high=0.45
    ),
    ParameterSearchRange(
        name='deaths',
        type='uniform',
        low=0,
        high=.3,
    ),
    ParameterSearchRange(
        name='kills',
        type='uniform',
        low=0,
        high=0.3
    ),
    ParameterSearchRange(
        name='result',
        type='uniform',
        low=0.25,
        high=0.85
    ),
]

features = ["damagetochampions", "result",
            "kills", "deaths"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

pre_transformer_search_ranges = [
    (standard_scaler, []),
    (MinMaxTransformer(features=features), []),
    (
        ColumnsWeighter(weighted_column_name=column_names.performance, column_weights=[]),
        column_weigher_search_range),
]

match_predictor = MatchPredictor(
    column_names=column_names,
    predictor=predictor,
    pre_rating_transformers=pre_transformers,
)

start_rating_search_range = [
    ParameterSearchRange(
        name='team_weight',
        type='uniform',
        low=0.12,
        high=.4,
    ),
    ParameterSearchRange(
        name='league_quantile',
        type='uniform',
        low=0.12,
        high=.4,
    ),
    ParameterSearchRange(
        name='min_count_for_percentiles',
        type='uniform',
        low=20,
        high=100,
    )
]

pre_transformer_tuner = TransformerTuner(match_predictor=match_predictor,
                                            pre_transformer_search_ranges=pre_transformer_search_ranges,
                                            n_trials=15
                                            )

player_rating_tuner = PlayerRatingTuner(match_predictor=match_predictor,
                                        search_ranges=player_search_ranges,
                                        n_trials=20
                                        )

start_rating_tuner = StartRatingTuner(column_names=column_names,
                                      match_predictor=match_predictor,
                                      n_trials=15,
                                      search_ranges=start_rating_search_range)

tuner = MatchPredictorTuner(
    pre_transformer_tuner=pre_transformer_tuner,
    team_rating_tuner=player_rating_tuner,
    start_rating_tuner=start_rating_tuner,
    fit_best=True,
)
best_match_predictor = tuner.tune(df=df)
pickle.dump(best_match_predictor, open("models/lol_match_predictor", 'wb'))
