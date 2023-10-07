from src.ratings.data_structures import RatingUpdateParameters, StartRatingParameters, PerformancePredictorParameters, \
    TeamRatingParameters
from src.ratings.enums import PredictedRatingMethod
from src.ratings.match_rating.entity_rating_generator import TeamRatingGenerator, BasePlayerRatingUpdater
from src.ratings.match_rating.match_rating_calculator import RatingMeanPerformancePredictor, \
    PerformancePredictor
from src.ratings.rating_generator import RatingGenerator
from src.ratings.start_rating_calculator import StartRatingGenerator


class RatingGeneratorFactory():

    def __init__(self,
                 rating_update_params: RatingUpdateParameters,
                 start_rating_params: StartRatingParameters,
                 performance_predictor_params: PerformancePredictorParameters,
                 team_rating_parameters: TeamRatingParameters
                 ):
        self.rating_update_params = rating_update_params
        self.start_rating_params = start_rating_params
        self.performance_predictor_params = performance_predictor_params
        self.team_rating_params = team_rating_parameters

    def create(self) -> RatingGenerator:
        performance_predictor = PerformancePredictor(params=self.performance_predictor_params)
        start_rating_generator = StartRatingGenerator(params=self.start_rating_params)
        player_rating_updater = BasePlayerRatingUpdater(update_params=self.rating_update_params,
                                                        start_rating_generator=start_rating_generator,
                                                        performance_predictor=performance_predictor)
        team_rating_generator = TeamRatingGenerator(player_rating_updater=player_rating_updater,
                                                    params=self.team_rating_params)
        return RatingGenerator(team_rating_generator=team_rating_generator)
