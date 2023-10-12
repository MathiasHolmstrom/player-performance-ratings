import pandas as pd

from src.ratings.data_prepararer import get_matches_from_df
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

matches = get_matches_from_df(df=df, column_names=column_names)

rating_generator = RatingGenerator(generate_leagues=False)
match_ratings = rating_generator.generate(matches)

df = (df
      .assign(player_rating=match_ratings.pre_match_player_rating_values,
              team_rating=match_ratings.pre_match_team_rating_values,
              oppponent_rating=match_ratings.pre_match_opponent_rating_values
              )
      )
