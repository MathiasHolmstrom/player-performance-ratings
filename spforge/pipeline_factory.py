import copy
from typing import Optional, List, Union

from spforge import Pipeline, ColumnNames


from spforge.predictor._base import BasePredictor
from spforge.ratings.rating_generator import RatingGenerator
from spforge.transformers.base_transformer import (
    BaseTransformer,
)
from spforge.transformers.fit_transformers import PerformanceWeightsManager
from spforge.transformers.lag_transformers import BaseLagTransformer


class PipelineFactory:

    def __init__(
        self,
        predictor: BasePredictor,
        column_names: ColumnNames,
        rating_generators: Optional[
            Union[RatingGenerator, list[RatingGenerator]]
        ] = None,
        performances_generator: Optional[PerformanceWeightsManager] = None,
        pre_lag_transformers: Optional[List[BaseTransformer]] = None,
        lag_generators: Optional[List[BaseLagTransformer]] = None,
        post_lag_transformers: Optional[List[BaseTransformer]] = None,
    ):
        self.post_lag_transformers = post_lag_transformers or []
        self.pre_lag_transformers = pre_lag_transformers or []
        self.lag_generators = lag_generators or []
        self.column_names = column_names
        self.rating_generators = rating_generators or []
        if isinstance(self.rating_generators, RatingGenerator):
            self.rating_generators = [self.rating_generators]

        self.predictor = predictor
        self.performances_generator = performances_generator

    def create(
        self,
        performances_generator: Optional[PerformanceWeightsManager] = None,
        rating_generators: Optional[list[RatingGenerator]] = None,
        lag_generators: Optional[List[BaseLagTransformer]] = None,
        pre_lag_transformers: Optional[List[BaseTransformer]] = None,
        post_lag_transformers: Optional[List[BaseTransformer]] = None,
        predictor: Optional[BasePredictor] = None,
    ) -> Pipeline:
        rating_generators = (
            rating_generators
            if rating_generators is not None
            else [copy.deepcopy(r) for r in self.rating_generators]
        )
        performances_generator = (
            performances_generator
            if performances_generator is not None
            else copy.deepcopy(self.performances_generator)
        )
        pre_lag_transformers = (
            pre_lag_transformers
            if pre_lag_transformers is not None
            else [copy.deepcopy(r.reset()) for r in self.pre_lag_transformers]
        )
        post_lag_transformers = (
            post_lag_transformers
            if post_lag_transformers is not None
            else [copy.deepcopy(r.reset()) for r in self.post_lag_transformers]
        )
        lag_generators = (
            lag_generators
            if lag_generators is not None
            else [copy.deepcopy(r.reset()) for r in self.lag_generators]
        )

        predictor = predictor if predictor is not None else self.predictor

        return Pipeline(
            rating_generators=rating_generators,
            lag_transformers=copy.deepcopy(lag_generators),
            pre_lag_transformers=copy.deepcopy(pre_lag_transformers),
            post_lag_transformers=copy.deepcopy(post_lag_transformers),
            predictor=predictor,
            column_names=self.column_names,
        )
