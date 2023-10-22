

from player_performance_ratings import MatchPredictor
from player_performance_ratings import SKLearnClassifierWrapper
from player_performance_ratings import PlayerRatingGenerator
from player_performance_ratings import TeamRatingGenerator

from sklearn.preprocessing import StandardScaler


from examples.utils import load_data

from player_performance_ratings import ColumnNames
from player_performance_ratings import RatingColumnNames, RatingGenerator

from player_performance_ratings import MinMaxTransformer, SkLearnTransformerWrapper, ColumnsWeighter
from player_performance_ratings import PreTransformerTuner
from player_performance_ratings import ParameterSearchRange

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
            name='result',
            type='uniform',
            low=0,
            high=0.45
        ),
        ParameterSearchRange(
            name='kills',
            type='uniform',
            low=0,
            high=.3,
        ),
        ParameterSearchRange(
            name='deaths',
            type='uniform',
            low=0,
            high=0.3
        ),
        ParameterSearchRange(
            name='assists',
            type='uniform',
            low=0,
            high=0.2
        ),
        ParameterSearchRange(
            name='damagetochampions',
            type='uniform',
            low=0,
            high=0.4,
        ),
    ]

    features = ["result", "kills",
                "deaths", "assists", "damagetochampions"]
    standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

    pre_transformer_search_ranges = [
        (standard_scaler, []),
        (MinMaxTransformer(features=features), []),
        (
            ColumnsWeighter(weighted_column_name=column_names.performance, column_weights=[]),
            column_weigher_search_range),
    ]

    team_rating_generator = TeamRatingGenerator(
        player_rating_generator=PlayerRatingGenerator(rating_change_multiplier=80))
    rating_generator = RatingGenerator()
    predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.RATING_DIFFERENCE], target='result')

    match_predictor = MatchPredictor(
        rating_generator=rating_generator,
        column_names=column_names,
        predictor=predictor
    )

    pre_transformer_tuner = PreTransformerTuner(match_predictor=match_predictor,
                                                pre_transformer_search_ranges=pre_transformer_search_ranges,
                                                n_trials=10
                                                )
    pre_transformer_tuner.tune(df=df)
