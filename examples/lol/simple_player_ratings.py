import pandas as pd

from src.ratings.data_structures import ColumnNames

from src.ratings.rating_generator import RatingGenerator


df = pd.read_csv("data/2023_LoL.csv")
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
