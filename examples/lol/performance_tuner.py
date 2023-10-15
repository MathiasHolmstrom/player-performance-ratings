import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from examples.lol.custom_performance import DurationPerformanceGenerator, \
    LolPlayerPerformanceGenerator, FinalLolTransformer
from src.auto_predictor.tuner import PreTransformerTuner
from src.auto_predictor.tuner.pre_transformer_tuner import ParameterSearchRange
from src.predictor.match_predictor import MatchPredictor
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.rating_generator import RatingGenerator
from src.transformers import BaseTransformer, ColumnsWeighter

from src.transformers.common import MinMaxTransformer, ColumnWeight

if __name__ == '__main__':
    column_names = ColumnNames(
        team_id='teamname',
        match_id='gameid',
        start_date="date",
        player_id="playername",
        performance='performance',
        league='league'
    )
    df = pd.read_csv("data/2023_LoL.csv")
    df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

    df = (
        df.loc[lambda x: x.position != 'team']
        .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
        .loc[lambda x: x.team_count == 2]
    )

    match_predictor = MatchPredictor(
        column_names=column_names,
        target='result'
    )

    duration_performance_search_range = []
    column_weigher_search_range = [
        ParameterSearchRange(
            name='net_damage_percentage',
            type='uniform',
            low=0,
            high=0.4
        ),
        ParameterSearchRange(
            name='net_deaths_percentage',
            type='uniform',
            low=0,
            high=0.2
        ),
        ParameterSearchRange(
            name='net_kills_assists_percentage',
            type='uniform',
            low=0,
            high=0.4
        ),
        ParameterSearchRange(
            name='team_duration_performance',
            type='uniform',
            low=0.3,
            high=0.8
        ),
    ]

    pre_transformer_search_ranges = [
        (DurationPerformanceGenerator(), []),
        (LolPlayerPerformanceGenerator(), []),
        (MinMaxTransformer(features=["net_damage_percentage", "net_deaths_percentage",
                                     "net_kills_assists_percentage", "team_duration_performance"]), []),
        (
            ColumnsWeighter(weighted_column_name=column_names.performance, column_weights=[]),
            column_weigher_search_range),
        (FinalLolTransformer(column_names), []),
    ]

    rating_generator = RatingGenerator()
    predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.rating_difference], target='result')
    pre_transformer_tuner = PreTransformerTuner(column_names=column_names,
                                                pre_transformer_search_ranges=pre_transformer_search_ranges,
                                                rating_generator=rating_generator,
                                                predictor=predictor
                                                )
    pre_transformer_tuner.tune(df=df)
