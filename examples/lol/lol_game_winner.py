import pandas as pd

from src.ratings.data_prepararer import get_matches_from_df
from src.ratings.data_structures import ColumnNames
from src.ratings.match_rating.player_rating_generator import PlayerRatingGenerator
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator

from src.ratings.rating_generator import RatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator

df = pd.read_csv("data/2023_LoL.csv")
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date_time="date",
    player_id="playername",
    performance="result",
)

matches = get_matches_from_df(df=df, column_names=column_names)

start_rating_generator = StartRatingGenerator()

player_rating_generator = PlayerRatingGenerator()
team_rating_generator = TeamRatingGenerator(player_rating_generator=player_rating_generator)

rating_generator = RatingGenerator(team_rating_generator=team_rating_generator)
rating_generator.generate(matches)
