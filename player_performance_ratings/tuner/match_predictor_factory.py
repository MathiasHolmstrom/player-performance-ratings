from typing import Optional, List, Union, Tuple

import pendulum

from player_performance_ratings import MatchPredictor, ColumnNames, BaseMLWrapper, BaseTransformer
from player_performance_ratings.ratings.rating_generator import RatingGenerator


class MatchPredictorFactory():

    def __init__(self,
                 column_names: Union[ColumnNames, list[ColumnNames]],
                 rating_generators: Union[RatingGenerator, list[RatingGenerator]],
                 pre_transformers: Optional[List[BaseTransformer]] = None,
                 post_transformers: Optional[List[BaseTransformer]] = None,
                 predictor: BaseMLWrapper = None,
                 train_split_date: Optional[pendulum.datetime] = None
                 ):
        self.rating_generators = rating_generators
        if isinstance(self.rating_generators, RatingGenerator):
            self.rating_generators = [self.rating_generators]
        self.pre_transformers = pre_transformers or []
        self.post_transformers = post_transformers or []
        self.column_names = column_names
        self.predictor = predictor
        self.train_split_date = train_split_date

    def create(self,
               pre_rating_transformers: Optional[List[BaseTransformer]] = None,
               rating_generators: Optional[list[RatingGenerator]] = None,
               post_rating_transformers: Optional[List[BaseTransformer]] = None) -> MatchPredictor:

        rating_generators = rating_generators if rating_generators is not None else self.rating_generators
        pre_rating_transformers = pre_rating_transformers if pre_rating_transformers is not None else self.pre_transformers
        post_rating_transformers = post_rating_transformers if post_rating_transformers is not None else self.post_transformers

        return MatchPredictor(column_names=self.column_names, rating_generators=rating_generators,
                              pre_rating_transformers=pre_rating_transformers,
                              post_rating_transformers=post_rating_transformers,
                              predictor=self.predictor,
                              train_split_date=self.train_split_date)
