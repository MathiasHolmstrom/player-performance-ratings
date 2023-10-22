import copy
from typing import Optional

from player_performance_ratings.ratings.match_rating.performance_predictor import PerformancePredictor
from player_performance_ratings.ratings.match_rating.player_rating.player_rating_generator import PlayerRatingGenerator
from player_performance_ratings.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from src import RatingGenerator
from src import StartRatingGenerator


class RatingGeneratorFactory():

    def __init__(self,
                 start_rating_generator: Optional[StartRatingGenerator] = None,
                 player_rating_generator: Optional[PlayerRatingGenerator] = None,
                 performance_predictor: Optional[PerformancePredictor] = None,
                 team_rating_generator: Optional[TeamRatingGenerator] = None,
                 ):
        self.start_rating_generator = start_rating_generator or StartRatingGenerator()
        self.player_rating_generator = player_rating_generator or PlayerRatingGenerator()
        self.performance_predictor = performance_predictor or PerformancePredictor()
        self.team_rating_generator = team_rating_generator or TeamRatingGenerator()

    def create(self) -> RatingGenerator:
        player_rating_generator = copy.deepcopy(self.player_rating_generator)
        player_rating_generator.start_rating_generator = self.start_rating_generator
        player_rating_generator.performance_predictor = self.performance_predictor
        team_rating_generator = copy.deepcopy(self.team_rating_generator)
        team_rating_generator.player_rating_generator = player_rating_generator

        return RatingGenerator(
            team_rating_generator=team_rating_generator,
        )
