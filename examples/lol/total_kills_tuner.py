from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from estimators.ordinal_classifier import OrdinalClassifier
from examples.utils import load_data
from player_performance_ratings import TeamRatingGenerator, PlayerRatingGenerator, MatchPredictor, \
    SKLearnClassifierWrapper, SkLearnTransformerWrapper, RatingColumnNames
from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.ratings.data_prepararer import MatchGenerator
from player_performance_ratings.ratings.match_rating.performance_predictor import RatingMeanPerformancePredictor
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.scorer.score import OrdinalLossScorer
from player_performance_ratings.transformers.common import DiminishingValueTransformer, MinMaxTransformer

df = load_data()
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df['minutes_played'] = df['gamelength'] / 60
df = df[df['position'] != 'team']
df['total_kills'] = df.groupby(['gameid', 'teamname'])['kills'].transform('sum') + df.groupby(['gameid', 'teamname'])[
    'deaths'].transform('sum')
df['total_kills_per_minute'] = df['total_kills'] / df['minutes_played']
df['team_count'] = df.groupby(['gameid'])['teamname'].transform('nunique')

df = df[df['team_count'] == 2]

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance="total_kills_per_minute",
)

match_generator = MatchGenerator(column_names=column_names)
matches = match_generator.generate(df=df)

rating_generator = RatingGenerator(team_rating_generator=TeamRatingGenerator(
    player_rating_generator=PlayerRatingGenerator(performance_predictor=RatingMeanPerformancePredictor())))
match_ratings = rating_generator.generate(matches=matches)

pre_rating_transformers = [
    SkLearnTransformerWrapper(transformer=StandardScaler(), features=['total_kills_per_minute']),
    DiminishingValueTransformer(features=['total_kills_per_minute']),
    MinMaxTransformer(features=['total_kills_per_minute']),
]

predictor = SKLearnClassifierWrapper(
    model=OrdinalClassifier(),
    features=[RatingColumnNames.RATING_MEAN],
    target='total_kills',
    multiclassifier=True
)

match_predictor = MatchPredictor(pre_rating_transformers=pre_rating_transformers, column_names=column_names,
                                 rating_generator=rating_generator, predictor= predictor)
df = match_predictor.generate(df)

score = OrdinalLossScorer(
    pred_column=match_predictor.predictor.pred_column,
    target=match_predictor.predictor.target,
    granularity=[column_names.team_id, column_names.match_id]
)
score.score(df=df, classes_=match_predictor.predictor.model.classes_)

