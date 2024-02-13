import pandas as pd
from lightgbm import LGBMClassifier


from player_performance_ratings.predictor.estimators import Predictor

from player_performance_ratings import ColumnNames, PredictColumnNames

from player_performance_ratings.predictor import MatchPredictor
from player_performance_ratings.ratings.enums import RatingEstimatorFeatures
from player_performance_ratings.ratings.time_weight_ratings import BayesianTimeWeightedRating
from player_performance_ratings.scorer.score import OrdinalLossScorer

column_names_kpm = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='kills_per_minute',
    league='league',
    position='position',
)

column_names_kills = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='kills',
    league='league',
    position='position'
)

df = pd.read_parquet("data/subsample_lol_data")
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])
df['__target'] = df['kills']
df["__target"] = df["__target"].clip(0, 8)

df['kills_per_minute'] = df['kills'] / df['gamelength'] * 60

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

time_weighed_rating_kills_per_minute = BayesianTimeWeightedRating(column_names=column_names_kpm)
time_weighed_rating_kills = BayesianTimeWeightedRating(column_names=column_names_kills)

match_predictor = MatchPredictor(
    rating_generators=[time_weighed_rating_kills_per_minute, time_weighed_rating_kills],
    use_auto_create_performance_calculator=True,
    column_weights=[[ColumnWeight(name='kills_per_minute', weight=1)], [ColumnWeight(name='kills', weight=1)]],
    other_categorical_features=["position"],
    predictor=Predictor(
        estimator=LGBMClassifier(verbose=-100),
        features=[
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING + "0",
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING + "1",
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO + "0",
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO + "1",
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING_EVIDENCE + "0",
            RatingEstimatorFeatures.TIME_WEIGHTED_RATING_EVIDENCE + "1",
            "position"
        ],
        target=PredictColumnNames.TARGET,
    )
)

df_predictions = match_predictor.train(df=df)

for idx, kills in enumerate(match_predictor._predictor.estimator.classes_):
    print(df_predictions.iloc[500]['playername'], kills,
          df_predictions.iloc[500][match_predictor._predictor.pred_column][idx])

print(match_predictor._predictor.estimator.feature_importances_)

scorer = OrdinalLossScorer(
    pred_column=match_predictor._predictor.pred_column,
)

print(scorer.score(df_predictions, classes_=match_predictor._predictor.estimator.classes_))
