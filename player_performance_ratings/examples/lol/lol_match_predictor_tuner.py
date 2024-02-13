import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


from player_performance_ratings.scorer.score import SklearnScorer
from sklearn.metrics import log_loss

from player_performance_ratings.cross_validator.cross_validator import MatchKFoldCrossValidator
from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.predictor import GameTeamPredictor

from player_performance_ratings.tuner.rating_generator_tuner import UpdateRatingGeneratorTuner
from player_performance_ratings.ratings import UpdateRatingGenerator, ColumnWeight

from player_performance_ratings.data_structures import ColumnNames

from player_performance_ratings.tuner import PipelineTuner, PerformancesGeneratorTuner
from player_performance_ratings.tuner.utils import ParameterSearchRange, get_default_team_rating_search_range

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='performance',
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

rating_generator = UpdateRatingGenerator(column_names=column_names)

weighter_search_range = {
    column_names.performance:
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

performance_generator_tuner = PerformancesGeneratorTuner(performances_weight_search_ranges=weighter_search_range,
                                                         lower_is_better_features=["deaths"],
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
column_weights = [ColumnWeight(name='kills', weight=1),
                  ColumnWeight(name='deaths', weight=1),
                  ColumnWeight(name='damagetochampions', weight=1),
                  ColumnWeight(name='result', weight=1)]

pipeline = Pipeline(
    rating_generators=rating_generator,
    column_weights=column_weights,
    predictor=predictor,
)

cross_validator = MatchKFoldCrossValidator(
    scorer=SklearnScorer(pred_column=pipeline.predictor.pred_column, scorer_function=log_loss),
    match_id_column_name=column_names.match_id,
    n_splits=1,
    date_column_name=column_names.start_date
)

tuner = PipelineTuner(
    performances_generator_tuners=performance_generator_tuner,
    rating_generator_tuners=rating_generator_tuner,
  #  predictor_tuner=predictor_tuner,
    fit_best=True,
    pipeline=pipeline,
    cross_validator=cross_validator,
)
best_match_predictor = tuner.tune(df=df)
pickle.dump(best_match_predictor, open("models/lol_match_predictor", 'wb'))
