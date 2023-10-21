import pandas as pd

from examples.utils import load_data
from src.ratings.data_structures import ColumnNames

from src.ratings.rating_generator import RatingGenerator

df = load_data()
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df = df[df['position'] != 'team']
df['team_count'] = df.groupby(['gameid'])['teamname'].transform('nunique')

df = df[df['team_count'] == 2]

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance="result",
)

rating_generator = RatingGenerator()
match_ratings = rating_generator.generate(df)
