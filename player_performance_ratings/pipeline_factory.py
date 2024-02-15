import copy
from typing import Optional, List, Union



from player_performance_ratings import Pipeline
from player_performance_ratings.ratings import PerformancesGenerator, ColumnWeight

from player_performance_ratings.predictor import BasePredictor
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.transformation.base_transformer import BaseTransformer, BasePostTransformer


class PipelineFactory():

    def __init__(self,
                 predictor: BasePredictor,
                 rating_generators: Optional[Union[RatingGenerator, list[RatingGenerator]]] = None,
                 performances_generator: Optional[PerformancesGenerator] = None,
                 post_rating_transformers: Optional[List[BasePostTransformer]] = None,
                 ):

        self.post_rating_transformers = post_rating_transformers or []


        self.rating_generators = rating_generators or []
        if isinstance(self.rating_generators, RatingGenerator):
            self.rating_generators = [self.rating_generators]

        self.predictor = predictor
        self.performances_generator = performances_generator


    def create(self,
               performances_generator: Optional[PerformancesGenerator] = None,
               rating_generators: Optional[list[RatingGenerator]] = None,
               post_rating_transformers: Optional[List[BaseTransformer]] = None,
               predictor: Optional[BasePredictor] = None,
               ) -> Pipeline:

        rating_generators = rating_generators if rating_generators is not None else [copy.deepcopy(r) for r in self.rating_generators]
        performances_generator = performances_generator if performances_generator is not None else copy.deepcopy(self.performances_generator)
        post_rating_transformers = post_rating_transformers if post_rating_transformers is not None else [copy.deepcopy(r) for r in self.post_rating_transformers]
        predictor = predictor if predictor is not None else self.predictor

        return Pipeline(
            rating_generators=rating_generators,
            performances_generator=performances_generator,
            post_rating_transformers=copy.deepcopy(post_rating_transformers),
            predictor=predictor,
        )
