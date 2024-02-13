import pandas as pd
from lightgbm import LGBMClassifier

from player_performance_ratings.tuner.predictor_tuner import PredictorTuner
from player_performance_ratings.tuner.utils import get_default_team_rating_search_range, ParameterSearchRange, \
    get_default_lgbm_classifier_search_range

from player_performance_ratings.predictor.transformer import ConvertDataFrameToCategoricalTransformer
from player_performance_ratings.ratings.rating_calculators.performance_predictor import \
    RatingNonOpponentPerformancePredictor
from player_performance_ratings.scorer import OrdinalLossScorer

from player_performance_ratings import ColumnNames, Pipeline
from player_performance_ratings.cross_validator.cross_validator import MatchKFoldCrossValidator
from player_performance_ratings.predictor import Predictor, SkLearnWrapper
from player_performance_ratings.ratings import ColumnWeight, UpdateRatingGenerator, PerformancesGenerator, \
    RatingEstimatorFeatures
from player_performance_ratings.ratings.rating_calculators import MatchRatingGenerator, RatingMeanPerformancePredictor, \
    StartRatingGenerator
from player_performance_ratings.tuner import PipelineTuner, PerformancesGeneratorTuner
from player_performance_ratings.tuner.rating_generator_tuner import UpdateRatingGeneratorTuner

df = pd.read_pickle(
    r"C:\Users\Admin\PycharmProjects\rating-model-transformer\examples\lol\data\lol_game_winner_total_kills_df")

df = df[df['date'] > '2021-01-01']
df['league_position'] = df['league'] + '__' + df['position']
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])



df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)


column_names_weighted = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='performance',
    league='league_position',
    position='position',
)

column_names_kpm = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='performance_kills',
    league='league_position',
    position='position',
)


df = df[
    [column_names_weighted.team_id, column_names_weighted.match_id, column_names_weighted.start_date, column_names_weighted.player_id,
     "kills", "deaths",
     "damagetochampions", "result", "position", "league", "champion", "gamelength", "assists", "duration",
     "kills_rating_avg", "prob",'league_position',
     "total_kills"]]

df['__target'] = df['kills']
df["__target"] = df['__target'].clip(0, 9)

df['kills_per_minute'] = df['kills'] / df['gamelength'] * 60

estimator = LGBMClassifier(max_depth=2, n_estimators=60, verbose=-100)

predictor = Predictor(
    estimator=estimator,
    estimator_features=[
        "prob",
        "kills_rating_avg",
        "position"
    ],
    categorical_transformers=[ConvertDataFrameToCategoricalTransformer(features=["position"])],
)



rating_generator_weighted = UpdateRatingGenerator(
    column_names=column_names_weighted,
    match_rating_generator=MatchRatingGenerator(
        league_rating_adjustor_multiplier=0,
        performance_predictor=RatingNonOpponentPerformancePredictor(),
        start_rating_generator=StartRatingGenerator(league_quantile=0.5, min_count_for_percentiles=5),
    ),
)

rating_generator_kpm = UpdateRatingGenerator(
    column_names=column_names_kpm ,
    match_rating_generator=MatchRatingGenerator(
        league_rating_adjustor_multiplier=0,
        performance_predictor=RatingNonOpponentPerformancePredictor(),
        start_rating_generator=StartRatingGenerator(league_quantile=0.5, min_count_for_percentiles=5),
    ),
)
column_weights = [
    [ColumnWeight(name='kills_per_minute', weight=1), ColumnWeight(name='kills', weight=1), ColumnWeight(name='assists', weight=1)],
    [ColumnWeight("kills", weight=1)],
]

pipeline = Pipeline(
    rating_generators=[rating_generator_weighted, rating_generator_kpm],
    predictor=predictor,
    performances_generator=PerformancesGenerator(
        column_names=[column_names_weighted, column_names_kpm],
        column_weights=column_weights
    )
)


weighter_search_range = {
    column_names_weighted.performance:
        [
            ParameterSearchRange(
                name='kills_per_minute',
                type='uniform',
                low=0,
                high=1
            ),
            ParameterSearchRange(
                name='kills',
                type='uniform',
                low=0,
                high=1
            ),
            ParameterSearchRange(
                name='assists',
                type='uniform',
                low=0,
                high=0.1
            ),
        ]
}

performance_generator_tuner = PerformancesGeneratorTuner(performances_weight_search_ranges=weighter_search_range,
                                                         n_trials=15)

cross_validator = MatchKFoldCrossValidator(
    scorer=OrdinalLossScorer(pred_column=pipeline.predictor.pred_column, targets_to_measure=[t for t in range(9)]),
    match_id_column_name=column_names_weighted.match_id,
    n_splits=1,
    date_column_name=column_names_weighted.start_date,
)

rating_generator_tuner_weighted = UpdateRatingGeneratorTuner(
    team_rating_search_ranges=get_default_team_rating_search_range(),
    team_rating_n_trials=20,
    start_rating_n_trials=0,
)
predictor_tuner = PredictorTuner(
    search_ranges=get_default_lgbm_classifier_search_range(),
    date_column_name=column_names_weighted.start_date,
)
tuner = PipelineTuner(
    performances_generator_tuners=[performance_generator_tuner, None],
    rating_generator_tuners=[rating_generator_tuner_weighted, None],
    fit_best=True,
    pipeline=pipeline,
    predictor_tuner=predictor_tuner,
    cross_validator=cross_validator,
)
tuner.tune(df)
