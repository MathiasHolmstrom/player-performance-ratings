from dataclasses import dataclass
import pandas as pd

from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.rating_generator import RatingGenerator

PERFORMANCE = "performance"


@dataclass
class Config:
    league: bool
    hardcoded_entity_rating: bool



class PlayerRatingGeneratorOld():

    def __init__(self,
                 column_names: ColumnNames,
                 rating_generator: RatingGenerator
                 ):
        self.column_names = column_names
        self.rating_generator = rating_generator

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.performance_generator:
            df = self.performance_generator.generate(df)

        if PERFORMANCE not in df.columns:
            raise ValueError("performance column not created")

        matches = generate_matches(column_names=self.column_names, df=df)

        match_ratings = self.rating_generator.generate(matches)
        df[RatingColumnNames.PLAYER_RATING] = match_ratings.pre_match_player_rating_values
        df[RatingColumnNames.TEAM_RATING] = match_ratings.pre_match_team_rating_values
        df[RatingColumnNames.OPPONENT_RATING] = match_ratings.pre_match_opponent_rating_values
        return df

    @property
    def player_ratings(self):
        return

    @property
    def team_ratings(self):
        return
