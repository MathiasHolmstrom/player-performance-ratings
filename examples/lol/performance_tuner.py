from sklearn.preprocessing import StandardScaler

from examples.lol.custom_performance import DurationPerformanceGenerator, \
    LolPlayerPerformanceGenerator, FinalLolTransformer
from examples.utils import load_data
from src.auto_predictor.tuner import PreTransformerTuner
from src.auto_predictor.tuner.pre_transformer_tuner import ParameterSearchRange
from src.predictor.match_predictor import MatchPredictor
from src.predictor.ml_wrappers.classifier import SKLearnClassifierWrapper
from src.ratings.data_structures import ColumnNames
from src.ratings.enums import RatingColumnNames
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src.ratings.rating_generator import RatingGenerator
from src.transformers import BaseTransformer, ColumnsWeighter

from src.transformers.common import MinMaxTransformer, ColumnWeight, SkLearnTransformerWrapper

if __name__ == '__main__':
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
            low=0.15,
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

    team_rating_generator = TeamRatingGenerator(
        player_rating_generator=PlayerRatingGenerator(rating_change_multiplier=80))
    rating_generator = RatingGenerator()
    predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.rating_difference], target='result')

    match_predictor = MatchPredictor(
        rating_generator=rating_generator,
        column_names=column_names,
        predictor=predictor
    )

    pre_transformer_tuner = PreTransformerTuner(match_predictor=match_predictor,
                                                pre_transformer_search_ranges=pre_transformer_search_ranges,
                                                n_trials=1
                                                )
    pre_transformer_tuner.tune(df=df)
