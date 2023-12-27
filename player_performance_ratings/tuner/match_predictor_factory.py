from typing import Optional, List, Union

import pendulum
from player_performance_ratings.transformation.factory import auto_create_pre_transformers

from player_performance_ratings.transformation import ColumnWeight

from player_performance_ratings import ColumnNames
from player_performance_ratings.predictor import MatchPredictor
from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper
from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import RatingGenerator
from player_performance_ratings.transformation.base_transformer import BaseTransformer


class MatchPredictorFactory():

    def __init__(self,
                 column_names: Union[ColumnNames, list[ColumnNames]],
                 rating_generators: Optional[Union[RatingGenerator, list[RatingGenerator]]] = None,
                 pre_transformers: Optional[List[BaseTransformer]] = None,
                 post_transformers: Optional[List[BaseTransformer]] = None,
                 predictor: BaseMLWrapper = None,
                 train_split_date: Optional[pendulum.datetime] = None,
                 use_auto_pre_transformers: bool = False,
                 column_weights: Optional[Union[list[list[ColumnWeight]], list[ColumnWeight]]] = None
                 ):

        self.rating_generators = rating_generators or []
        if isinstance(self.rating_generators, RatingGenerator):
            self.rating_generators = [self.rating_generators]

        if len(self.rating_generators) > 1 and predictor is None:
            raise ValueError(
                "If multiple rating generators are used, a predictor must be specified."
                " Otherwise it is not clear which features from which rating-model is used ")


        self.post_transformers = post_transformers or []
        self.column_names = column_names
        if isinstance(self.column_names, ColumnNames):
            self.column_names = [self.column_names]
        self.predictor = predictor
        self.train_split_date = train_split_date
        self.use_auto_pre_transformers = use_auto_pre_transformers
        self.column_weights = column_weights
        self.pre_transformers = pre_transformers or []
        if self.use_auto_pre_transformers:
            self.pre_transformers = auto_create_pre_transformers(column_weights=self.column_weights,
                                                                        column_names=self.column_names)

    def create(self,
               pre_rating_transformers: Optional[List[BaseTransformer]] = None,
               rating_generators: Optional[list[RatingGenerator]] = None,
               post_rating_transformers: Optional[List[BaseTransformer]] = None,
               predictor: Optional[BaseMLWrapper] = None,
               ) -> MatchPredictor:

        rating_generators = rating_generators if rating_generators is not None else self.rating_generators
        pre_rating_transformers = pre_rating_transformers if pre_rating_transformers is not None else self.pre_transformers
        post_rating_transformers = post_rating_transformers if post_rating_transformers is not None else self.post_transformers
        predictor = predictor if predictor is not None else self.predictor



        return MatchPredictor(column_names=self.column_names,
                              rating_generators=rating_generators,
                              pre_rating_transformers=pre_rating_transformers,
                              post_rating_transformers=post_rating_transformers,
                              predictor=predictor,
                              train_split_date=self.train_split_date,
                              )
