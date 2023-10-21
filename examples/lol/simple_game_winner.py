from examples.utils import load_data
from src import MatchPredictor
from src import ColumnNames
from sklearn.metrics import log_loss

from src import PlayerRatingGenerator
from player_performance_ratings.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src import RatingGenerator
from src import StartRatingGenerator

df = load_data()
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance="result",
    league='league'
)

start_rating_generator = StartRatingGenerator(
    league_ratings={'LPL': 2600, 'LCK': 2500, 'LEC': 1950, 'LCS': 1850, 'TCL': 1450, 'VCS': 1650,
                          'PCS': 1550,
                          'LLA': 1300,
                          'CBLOL': 1450,
                          'EM': 1300,
                          'LJL': 1400, 'LFL': 1235, 'PRM': 1200, 'UL': 1200, 'LCSA': 1340, 'SL': 1220, 'LHE': 1200,
                          'GLL': 1100,
                          'EBL': 1100,
                          'NLC': 1100,
                          'UKLC': 1100,
                          'PGN': 1100,
                          'HM': 950,
                          'WCS': 2200,
                          'HC': 800,
                          'AL': 1100,
                          })

rating_generator = RatingGenerator(
    team_rating_generator=TeamRatingGenerator(
        player_rating_generator=PlayerRatingGenerator(
            start_rating_generator=start_rating_generator
        )
    )
)

match_predictor = MatchPredictor(column_names=column_names, rating_generator=rating_generator)
df = match_predictor.generate(df)

print(log_loss(df['result'], df[match_predictor.predictor.pred_column]))
