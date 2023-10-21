import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

from examples.lol.custom_performance import DurationPerformanceGenerator, LolPlayerPerformanceGenerator, \
    FinalLolTransformer
from src.auto_predictor.tuner import PreTransformerTuner, StartRatingTuner
from src.auto_predictor.tuner.base_tuner import ParameterSearchRange
from src.auto_predictor.tuner.match_predicter_tuner import MatchPredictorTuner
from src.auto_predictor.tuner.player_rating_tuner import PlayerRatingTuner
from src.predictor.match_predictor import MatchPredictor
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.transformers.common import SkLearnTransformerWrapper, MinMaxTransformer, ColumnsWeighter

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='performance',
    league='league'
)
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

team_rating_generator = TeamRatingGenerator(
    player_rating_generator=PlayerRatingGenerator())
rating_generator = RatingGenerator()
predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.rating_difference], target='result')

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
        name='net_damage_percentage',
        type='uniform',
        low=0,
        high=0.45
    ),
    ParameterSearchRange(
        name='net_deaths_percentage',
        type='uniform',
        low=0,
        high=.3,
    ),
    ParameterSearchRange(
        name='net_kills_assists_percentage',
        type='uniform',
        low=0,
        high=0.3
    ),
    ParameterSearchRange(
        name='team_duration_performance',
        type='uniform',
        low=0.25,
        high=0.85
    ),
]

features = ["net_damage_percentage", "net_deaths_percentage",
            "net_kills_assists_percentage", "team_duration_performance"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

pre_transformer_search_ranges = [
    (DurationPerformanceGenerator(), []),
    (LolPlayerPerformanceGenerator(), []),
    (standard_scaler, []),
    (MinMaxTransformer(features=features), []),
    (
        ColumnsWeighter(weighted_column_name=column_names.performance, column_weights=[]),
        column_weigher_search_range),
    (FinalLolTransformer(column_names), []),
]

match_predictor = MatchPredictor(
    rating_generator=rating_generator,
    column_names=column_names,
    predictor=predictor,
    pre_rating_transformers=pre_transformers,
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

pre_transformer_tuner = PreTransformerTuner(match_predictor=match_predictor,
                                            pre_transformer_search_ranges=pre_transformer_search_ranges,
                                            n_trials=15
                                            )

player_rating_tuner = PlayerRatingTuner(match_predictor=match_predictor,
                                        search_ranges=player_search_ranges,
                                        n_trials=35
                                        )

start_rating_tuner = StartRatingTuner(column_names=column_names,
                                      match_predictor=match_predictor,
                                      n_trials=8,
                                      search_ranges=search_range)

tuner = MatchPredictorTuner(
    pre_transformer_tuner=pre_transformer_tuner,
    player_rating_tuner=player_rating_tuner,
    start_rating_tuner=start_rating_tuner,
    fit_best=True,
)
best_match_predictor = tuner.tune(df=df)
pickle.dump(best_match_predictor, open("models/lol_match_predictor", 'wb'))
