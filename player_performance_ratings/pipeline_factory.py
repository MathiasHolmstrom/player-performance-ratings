import copy
from typing import Optional, List, Union


from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.ratings import PerformancesGenerator, ColumnWeight

from player_performance_ratings.predictor import BaseMLWrapper
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.transformation.base_transformer import BaseTransformer, BasePostTransformer


class PipelineFactory():

    def __init__(self,
                 predictor: BaseMLWrapper,
                 rating_generators: Optional[Union[RatingGenerator, list[RatingGenerator]]] = None,
                 performances_generator: Optional[PerformancesGenerator] = None,
                 post_rating_transformers: Optional[List[BasePostTransformer]] = None,
                 column_weights: Optional[Union[List[List[ColumnWeight]], list[ColumnWeight]]] = None,
                 ):

        if rating_generators and performances_generator is None and not column_weights:
            raise ValueError("If performance generator is None, column weights must be specified")

        self.rating_generators = rating_generators or []
        if isinstance(self.rating_generators, RatingGenerator):
            self.rating_generators = [self.rating_generators]

        self.post_rating_transformers = post_rating_transformers or []

        self.predictor = predictor


        self.performances_generator = performances_generator
        self.column_weights = column_weights if isinstance(column_weights, list) else [
            column_weights] if column_weights else None



        if self.performances_generator is None and self.rating_generators:
            self.performances_generator = PerformancesGenerator(column_weights=self.column_weights,
                                                                column_names=self.rating_generators[0].column_names)

    def create(self,
               performances_generator: Optional[PerformancesGenerator] = None,
               rating_generators: Optional[list[RatingGenerator]] = None,
               post_rating_transformers: Optional[List[BaseTransformer]] = None,
               predictor: Optional[BaseMLWrapper] = None,
               ) -> Pipeline:

        rating_generators = rating_generators if rating_generators is not None else self.rating_generators
        performances_generator = performances_generator if performances_generator is not None else self.performances_generator
        post_rating_transformers = post_rating_transformers if post_rating_transformers is not None else self.post_rating_transformers
        predictor = predictor if predictor is not None else self.predictor

        return Pipeline(
            rating_generators=rating_generators,
            performances_generator=performances_generator,
            post_rating_transformers=copy.deepcopy(post_rating_transformers),
            predictor=predictor,
        )
