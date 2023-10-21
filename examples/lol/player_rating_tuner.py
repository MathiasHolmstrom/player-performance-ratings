from sklearn.preprocessing import StandardScaler

from examples.lol.custom_performance import DurationPerformanceGenerator, LolPlayerPerformanceGenerator, \
    FinalLolTransformer
from examples.utils import load_data
from src.auto_predictor.tuner.base_tuner import ParameterSearchRange
from src.auto_predictor.tuner.player_rating_tuner import PlayerRatingTuner
from src.predictor.match_predictor import MatchPredictor
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.transformers.common import SkLearnTransformerWrapper, MinMaxTransformer, ColumnsWeighter, ColumnWeight

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='performance',
    league='league'
)
df = load_data()
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

search_ranges = [
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

features = ["net_damage_percentage", "net_deaths_percentage",
            "net_kills_assists_percentage", "team_duration_performance"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

pre_transformers = [
    DurationPerformanceGenerator(),
    LolPlayerPerformanceGenerator(),
    standard_scaler,
    MinMaxTransformer(features=features),
    ColumnsWeighter(
        weighted_column_name=column_names.performance, column_weights=[
            ColumnWeight(
                name='net_damage_percentage',
                weight=0.25,
            ),
            ColumnWeight(
                name='net_deaths_percentage',
                weight=0.1,
            ),
            ColumnWeight(
                name='net_kills_assists_percentage',
                weight=0.1,
            ),
            ColumnWeight(
                name='team_duration_performance',
                weight=0.55,
            ),
        ]
    ),
    FinalLolTransformer(column_names)
]

team_rating_generator = TeamRatingGenerator(
    player_rating_generator=PlayerRatingGenerator())
rating_generator = RatingGenerator()
predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.rating_difference], target='result')

match_predictor = MatchPredictor(
    rating_generator=rating_generator,
    column_names=column_names,
    predictor=predictor,
    pre_rating_transformers=pre_transformers,
)

tuner = PlayerRatingTuner(match_predictor=match_predictor,
                          search_ranges=search_ranges,
                          n_trials=100
                          )
tuner.tune(df=df)
