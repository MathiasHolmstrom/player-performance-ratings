from sklearn.preprocessing import StandardScaler

from player_performance_ratings.predictor.estimators.ordinal_classifier import OrdinalClassifier
from examples.utils import load_data
from player_performance_ratings import TeamRatingGenerator, MatchPredictor, \
    SKLearnClassifierWrapper, SkLearnTransformerWrapper, RatingColumnNames, PreTransformerTuner, PlayerRatingTuner, \
    ParameterSearchRange, MatchPredictorTuner
from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.ratings.data_prepararer import MatchGenerator
from player_performance_ratings.ratings.match_rating.performance_predictor import RatingMeanPerformancePredictor
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.scorer.score import OrdinalLossScorer
from player_performance_ratings.transformers.common import DiminishingValueTransformer, MinMaxTransformer, \
    ColumnsWeighter

df = load_data()
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df['minutes_played'] = df['gamelength'] / 60
df = df[df['position'] != 'team']
df['total_kills'] = df.groupby(['gameid', 'teamname'])['kills'].transform('sum') + df.groupby(['gameid', 'teamname'])[
    'deaths'].transform('sum')
df['total_kills_per_minute'] = df['total_kills'] / df['minutes_played']
df['team_count'] = df.groupby(['gameid'])['teamname'].transform('nunique')

df = df[df['team_count'] == 2]

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance="total_kills_per_minute",
)

match_generator = MatchGenerator(column_names=column_names)
matches = match_generator.convert_df_to_matches(df=df)

rating_generator = RatingGenerator(team_rating_generator=TeamRatingGenerator(
    player_rating_generator=TeamRatingGenerator(performance_predictor=RatingMeanPerformancePredictor())))
match_ratings = rating_generator.generate(matches=matches)

predictor = SKLearnClassifierWrapper(
    model=OrdinalClassifier(),
    features=[RatingColumnNames.RATING_MEAN],
    target='total_kills',
    multiclassifier=True,
    granularity=[column_names.team_id, column_names.match_id]
)

column_weigher_search_range = [
    ParameterSearchRange(
        name='total_kills_per_minute',
        type='uniform',
        low=0.6,
        high=1
    ),
    ParameterSearchRange(
        name='total_kills',
        type='uniform',
        low=0,
        high=.4,
    ),

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
        low=30,
        high=100
    )
]

features = ["total_kills_per_minute", "total_kills"]

standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

pre_transformer_search_ranges = [
    (standard_scaler, []),
    (DiminishingValueTransformer(features=features), []),
    (MinMaxTransformer(features=features), []),
    (
        ColumnsWeighter(weighted_column_name=column_names.performance, column_weights=[]),
        column_weigher_search_range),
]
match_predictor = MatchPredictor(pre_rating_transformers=[], column_names=column_names,
                                 rating_generator=rating_generator, predictor=predictor)
scorer = OrdinalLossScorer(
    pred_column=match_predictor.predictor.pred_column,
    granularity=[column_names.team_id, column_names.match_id]
)

pre_transformer_tuner = PreTransformerTuner(match_predictor=match_predictor,
                                            pre_transformer_search_ranges=pre_transformer_search_ranges,
                                            n_trials=1,
                                            scorer=scorer,
                                            )

player_rating_tuner = PlayerRatingTuner(match_predictor=match_predictor,
                                        search_ranges=player_search_ranges,
                                        scorer=scorer,
                                        n_trials=3
                                        )

tuner = MatchPredictorTuner(
    pre_transformer_tuner=pre_transformer_tuner,
    team_rating_tuner=player_rating_tuner,
    fit_best=True,
    target='total_kills'
)
best_match_predictor = tuner.tune(df=df)
