from examples.utils import load_nba_game_player_data

from sklearn.metrics import log_loss

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor.match_predictor import MatchPredictor
from player_performance_ratings import PlayerRatingGenerator, SKLearnClassifierWrapper
from player_performance_ratings import TeamRatingGenerator
from player_performance_ratings import RatingGenerator
from player_performance_ratings.ratings.enums import RatingColumnNames

column_names = ColumnNames(
    team_id='TEAM_ID',
    match_id='GAME_ID',
    start_date="START_DATE",
    player_id="PLAYER_NAME",
    performance="WON",

)
df = load_nba_game_player_data()
df[PredictColumnNames.TARGET] = df['WON']
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])

df = (
    df.assign(team_count=df.groupby(column_names.match_id)[column_names.team_id].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)


rating_generator = RatingGenerator(
    store_game_ratings=True,
    column_names=column_names,
    team_rating_generator=TeamRatingGenerator(
        player_rating_generator=PlayerRatingGenerator(
        )
    )
)

predictor = SKLearnClassifierWrapper(features=[RatingColumnNames.RATING_DIFFERENCE], granularity=[column_names.match_id, column_names.team_id])

match_predictor = MatchPredictor(column_names=column_names, rating_generator=rating_generator, predictor=predictor)
df = match_predictor.generate(df)

print(log_loss(df[PredictColumnNames.TARGET], df[match_predictor.predictor.pred_column]))
