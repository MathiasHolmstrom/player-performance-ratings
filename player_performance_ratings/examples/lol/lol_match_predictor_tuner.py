import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression



from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.predictor import GameTeamPredictor

from player_performance_ratings.tuner.rating_generator_tuner import UpdateRatingGeneratorTuner
from player_performance_ratings.ratings import UpdateRatingGenerator

from player_performance_ratings.data_structures import ColumnNames

from player_performance_ratings.tuner import PipelineTuner, PerformancesGeneratorTuner
from player_performance_ratings.tuner.utils import ParameterSearchRange, get_default_team_rating_search_range

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    league='league',
    position='position',
)
df = pd.read_parquet("data/subsample_lol_data")
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])
df['champion_position'] = df['champion'] + df['position']
df['__target'] = df['result']

df = df.drop_duplicates(subset=['gameid', 'teamname', 'playername'])

df = (
    df.assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)

rating_generator = UpdateRatingGenerator(performance_column='performance')

weighter_search_range = {
    'performance':
        [
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
                lower_is_better=True
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
}

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

performance_generator_tuner = PerformancesGeneratorTuner(performances_search_range=weighter_search_range,
                                                         n_trials=1
                                                         )

rating_generator_tuner = UpdateRatingGeneratorTuner(
    team_rating_search_ranges=get_default_team_rating_search_range(),
    start_rating_search_ranges=start_rating_search_range,
    optimize_league_ratings=True,
    team_rating_n_trials=1
)

estimator = LogisticRegression()
predictor = GameTeamPredictor(
    estimator=estimator,
    game_id_colum="gameid",
    team_id_column="teamname",
)

pipeline = Pipeline(
    rating_generators=rating_generator,
    predictor=predictor,
    column_names=column_names
)


tuner = PipelineTuner(
    performances_generator_tuners=performance_generator_tuner,
    rating_generator_tuners=rating_generator_tuner,
  #  predictor_tuner=predictor_tuner,
    fit_best=True,
    pipeline=pipeline,
)
best_match_predictor = tuner.tune(df=df)
pickle.dump(best_match_predictor, open("models/lol_match_predictor", 'wb'))
