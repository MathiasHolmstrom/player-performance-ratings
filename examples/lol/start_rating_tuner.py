import pandas as pd
from sklearn.preprocessing import StandardScaler

from examples.lol.custom_performance import DurationPerformanceGenerator, LolPlayerPerformanceGenerator, \
    FinalLolTransformer
from src.auto_predictor.optimizer.start_rating_optimizer import StartLeagueRatingOptimizer
from src.auto_predictor.tuner import StartRatingTuner
from src.auto_predictor.tuner.base_tuner import ParameterSearchRange
from src.predictor.match_predictor import MatchPredictor
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.transformers import MinMaxTransformer, ColumnsWeighter
from src.transformers.common import SkLearnTransformerWrapper, ColumnWeight

df = pd.read_csv("data/2023_LoL.csv")
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance="performance",
    league='league'
)
team_rating_generator = TeamRatingGenerator(
    player_rating_generator=PlayerRatingGenerator())
rating_generator = RatingGenerator()
predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.rating_difference], target='result',
                                     granularity=[column_names.match_id, column_names.team_id])

features = ["net_damage_percentage", "net_deaths_percentage",
            "net_kills_assists_percentage", "team_duration_performance"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

column_weights = [
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
    )
]
pre_rating_transformers = [
    DurationPerformanceGenerator(),
    LolPlayerPerformanceGenerator(),
    standard_scaler,
    MinMaxTransformer(features=features),
    ColumnsWeighter(weighted_column_name=column_names.performance, column_weights=column_weights),
    FinalLolTransformer(column_names),
]

for pre_rating_transformer in pre_rating_transformers:
    df = pre_rating_transformer.transform(df)

match_predictor = MatchPredictor(
    pre_rating_transformers=pre_rating_transformers,
    rating_generator=rating_generator,
    column_names=column_names,
    predictor=predictor,
)
search_range = [
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

start_rating_parameters = {
    'league_ratings': {
        'LFL2': 995.3690978570788, 'DDH': 1001.6972563977271, 'EL': 1001.9623801765927,
        'LPL': 1193.0036543233155, 'GL': 972.0860804918582, 'LCKC': 1047.1402570481089, 'NEXO': 891.655203212699,
        'UL': 940.8878432041025, 'LVP SL': 1081.4615017055773, 'LCK': 1154.8937930743673, 'LFL': 1069.6820198639168,
        'PRM': 1083.1975832498783, 'LMF': 943.49789088995, 'SL (LATAM)': 814.6748295855517, 'VL': 841.8126766232117,
        'CBLOL': 1005.8671690942587, 'LEC': 1074.558028243576, 'NACL': 1019.0726786568753, 'LCO': 822.4594339879523,
        'CBLOLA': 898.7513037925586, 'LHE': 904.4400617161584, 'NLC': 956.2065299896113, 'GLL': 929.8384912191916,
        'ESLOL': 884.3034129874928, 'LLA': 1097.8867133917997, 'EBL': 923.7731732694314, 'TCL': 1036.971705880626,
        'PGN': 990.9156256948643, 'LPLOL': 895.1271163310976, 'LCS': 1076.694045068959, 'HM': 921.2211142189522,
        'LJL': 939.969333778972, 'HC': 871.5986289844975, 'AL': 957.638651307455, 'PCS': 1044.872177953617,
        'LDL': 1043.1029452803796, 'VCS': 964.1923669197374, 'EM': 838.7029865377992, 'LAS': 1002.5220624803582,
        'LRN': 1021.1969680354867, 'LRS': 1088.9773334605372, 'LJLA': 967.8531921903349, 'CT': 971.3692074260497,
        'EPL': 1493.4401321751918, 'CDF': 1053.5173594266287
        , 'IC': 873.8968953062821, 'WLDs': 996.9654114054242}

}

start_rating_optimizer = StartLeagueRatingOptimizer(column_names=column_names, match_predictor=match_predictor)

start_rating_tuner = StartRatingTuner(column_names=column_names, match_predictor=match_predictor, n_trials=4,
                                      search_ranges=search_range, start_rating_parameters=start_rating_parameters,
                                      start_rating_optimizer=start_rating_optimizer)
start_rating_tuner.tune(df=df)
