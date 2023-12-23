from lightgbm import LGBMClassifier
from player_performance_ratings.predictor.estimators import SKLearnClassifierWrapper

from player_performance_ratings import ColumnNames, PredictColumnNames
from player_performance_ratings.examples.utils import load_lol_subsampled_data
from player_performance_ratings.predictor import MatchPredictor
from player_performance_ratings.ratings.enums import RatingColumnNames
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

df = load_lol_subsampled_data()
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])
df['__target'] = df['kills']

df['kills_per_minute'] = df['kills'] / df['gamelength'] * 60

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

time_weighed_rating_kills_per_minute = BayesianTimeWeightedRating()
time_weighed_rating_kills = BayesianTimeWeightedRating()

match_predictor = MatchPredictor(
    rating_generators=[time_weighed_rating_kills_per_minute, time_weighed_rating_kills],
    column_names=[column_names_kpm, column_names_kills],
    predictor=SKLearnClassifierWrapper(
        model=LGBMClassifier(),
        features=[
            RatingColumnNames.TIME_WEIGHTED_RATING + "0",
            RatingColumnNames.TIME_WEIGHTED_RATING + "1",
            RatingColumnNames.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO + "0",
            RatingColumnNames.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO + "1",
            RatingColumnNames.TIME_WEIGHTED_RATING_EVIDENCE + "0",
            RatingColumnNames.TIME_WEIGHTED_RATING_EVIDENCE + "1",
                  column_names_kills.position
        ],
        target=PredictColumnNames.TARGET,
    )
)

df_predictions = match_predictor.generate_historical(df=df)

for idx, kills in enumerate(match_predictor.predictor.model.classes_):
    print(df_predictions.iloc[500]['playername'], kills, df_predictions.iloc[500][match_predictor.predictor.pred_column][idx])

print(match_predictor.predictor.model.feature_importances_)

scorer = OrdinalLossScorer(
    pred_column=match_predictor.predictor.pred_column,
)

print(scorer.score(df_predictions, classes_=match_predictor.predictor.model.classes_))

