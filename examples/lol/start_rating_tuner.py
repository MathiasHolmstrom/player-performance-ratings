import os

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

file_names = [
    # "2018_LoL.csv",
    "2019_LoL.csv",
    "2020_LoL.csv",
    "2021_LoL.csv",
    "2022_LoL.csv",
    "2023_LoL.csv"
]
dfs = []
for index, file_name in enumerate(file_names):
    full_path = os.path.join("data", file_name)
    df = pd.read_csv(full_path)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df[df['league'] != 'UPL'][['gameid', 'league', 'date', 'teamname', 'playername', 'result',
                                'gamelength', 'totalgold', 'teamkills', 'teamdeaths', 'position',
                                'damagetochampions',
                                'champion', 'kills', 'assists', 'deaths']]
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

start_rating_optimizer = StartLeagueRatingOptimizer(column_names=column_names, match_predictor=match_predictor)

start_rating_tuner = StartRatingTuner(column_names=column_names,
                                      match_predictor=match_predictor,
                                      n_trials=20,
                                      search_ranges=search_range,
                                      start_rating_optimizer=start_rating_optimizer
                                      )
start_rating_tuner.tune(df=df)
