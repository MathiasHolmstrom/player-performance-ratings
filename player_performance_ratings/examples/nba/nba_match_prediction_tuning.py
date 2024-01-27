import pickle

import pandas as pd
from lightgbm import LGBMClassifier

from player_performance_ratings.cross_validator.cross_validator import MatchKFoldCrossValidator
from player_performance_ratings.transformation.pre_transformers import ConvertDataFrameToCategoricalTransformer

from sklearn.metrics import log_loss

from player_performance_ratings.predictor import GameTeamPredictor
from player_performance_ratings.scorer.score import SklearnScorer

from player_performance_ratings.ratings.rating_calculators import MatchTeatingGenerator
from player_performance_ratings.ratings.rating_calculators.performance_predictor import \
    RatingDifferencePerformancePredictor

from player_performance_ratings.transformation import LagTransformer, RollingMeanTransformer

from player_performance_ratings.ratings import ColumnWeight, UpdateRatingGenerator
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner
from player_performance_ratings.tuner.rating_generator_tuner import OpponentAdjustedRatingGeneratorTuner


from player_performance_ratings import ColumnNames, Pipeline

from player_performance_ratings.tuner.utils import ParameterSearchRange, get_default_team_rating_search_range, \
    get_default_lgbm_classifier_search_range_by_learning_rate
from venn_abers import VennAbersCalibrator

from player_performance_ratings.consts import PredictColumnNames

from player_performance_ratings.predictor.sklearn_models import SkLearnWrapper


from player_performance_ratings.tuner import MatchPredictorTuner


column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_id",
    performance="plus_minus_per_minute",
    participation_weight="participation_weight",
    projected_participation_weight="projected_participation_weight",

)

df = pd.read_pickle("data/game_player_full_with_minutes_prediction.pickle")
poss = pd.read_pickle(r"C:\Users\Admin\PycharmProjects\nba-api-wrapper\examples\data\game_player_rotations.pickle")

df = df[df['game_id'].isin(poss['game_id'].unique().tolist())]

df[PredictColumnNames.TARGET] = df['won']
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])
df = df[df['game_minutes'] > 46]
df = df.rename(columns={'__target_prediction': 'minutes_prediction'})
df['plus_minus_per_minute'] = df['plus_minus'] / df['game_minutes']
df['participation_weight'] = df['minutes'] / df['game_minutes']
df['projected_participation_weight'] = df['minutes_prediction'] / df['game_minutes'].mean()
df['score_difference'] = df['score'] - df['score_opponent']

df = (
    df.assign(team_count=df.groupby('game_id')['team_id'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)
df = df.drop_duplicates(subset=['game_id', 'player_id'])

start_rating_search_range = [
    ParameterSearchRange(
        name='league_quantile',
        type='uniform',
        low=0.04,
        high=0.3,
    ),
    ParameterSearchRange(
        name='min_count_for_percentiles',
        type='int',
        low=50,
        high=300,
    ),
]
df.loc[df['minutes'] > 0, 'plus_minus_per_minute'] = df['plus_minus'] / df['minutes']
df.loc[df['minutes'] == 0, 'plus_minus_per_minute'] = 0

estimator = SkLearnWrapper(
    VennAbersCalibrator(
        estimator=LGBMClassifier(max_depth=2, learning_rate=0.1, n_estimators=200, verbose=-100, reg_alpha=1),
        inductive=True, cal_size=0.2, random_state=101))


estimator = SkLearnWrapper(

        estimator=LGBMClassifier(max_depth=2, learning_rate=0.1, n_estimators=200, verbose=-100, reg_alpha=1))

predictor = GameTeamPredictor(
    estimator=estimator,
    game_id_colum=column_names.match_id,
    team_id_column=column_names.team_id,
    categorical_transformers=[ConvertDataFrameToCategoricalTransformer(features=["location"])],
    #weight_column="projected_participation_weight",
)


performance_predictor = RatingDifferencePerformancePredictor(
    rating_diff_team_from_entity_coef=0.00425,
)

rating_generator = UpdateRatingGenerator(column_names=column_names, match_rating_generator=MatchTeatingGenerator(
    performance_predictor=performance_predictor))

column_weights = [ColumnWeight(name='plus_minus', weight=1)]

post_rating_transformers = [
    LagTransformer(
        features=["score_difference"],
        lag_length=5,
        granularity=[column_names.player_id],
        column_names=column_names,
        days_between_lags=[1, 2, 3, 4, 5],
    ),
    LagTransformer(
        features=[],
        lag_length=4,
        granularity=[column_names.player_id],
        column_names=column_names,
        prefix="future_lag",
        future_lag=True,
        days_between_lags=[1, 2, 3, 4]
    ),

    RollingMeanTransformer(
        features=["score_difference"],
        window=800,
        min_periods=300,
        granularity=["location"],
        column_names=column_names,
    )
]


pipeline = Pipeline(
    #post_rating_transformers=post_rating_transformers,
    rating_generators=rating_generator,
    column_weights=column_weights,
    predictor=predictor,
)

rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
    team_rating_search_ranges=get_default_team_rating_search_range(),
    start_rating_search_ranges=start_rating_search_range,
    team_rating_n_trials=1,
    start_rating_n_trials=1,
)
predictor_tuner = PredictorTuner(
    default_params={'learning_rate': 0.07},
    search_ranges=get_default_lgbm_classifier_search_range_by_learning_rate(learning_rate=0.07),
    n_trials=65,
    date_column_name=column_names.start_date,
    estimator_subclass_level=1,
)

cross_validator = MatchKFoldCrossValidator(
    scorer=SklearnScorer(pred_column=pipeline.predictor.pred_column, scorer_function=log_loss),
    match_id_column_name=column_names.match_id,
    n_splits=5,
    date_column_name=column_names.start_date
)

tuner = MatchPredictorTuner(
    pipeline=pipeline,
    fit_best=True,
    cross_validator=cross_validator,
    rating_generator_tuners=rating_generator_tuner,
    predictor_tuner=predictor_tuner,
)
best_match_predictor = tuner.tune(df=df)
df_with_minutes_prediction = best_match_predictor.generate_historical(df=df)

pickle.dump(best_match_predictor, open("models/nba_game_winner", 'wb'))
df_with_minutes_prediction.to_pickle("data/game_player_predictions.pickle")
