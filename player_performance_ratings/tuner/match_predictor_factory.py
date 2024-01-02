from typing import Optional, List, Union

import pendulum

from player_performance_ratings.ratings import PerformancesGenerator, ColumnWeight

from player_performance_ratings.predictor import MatchPredictor
from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper
from player_performance_ratings.ratings.opponent_adjusted_rating.rating_generator import RatingGenerator
from player_performance_ratings.transformation.base_transformer import BaseTransformer
from player_performance_ratings.transformation.factory import auto_create_performance_generator


class MatchPredictorFactory():

    def __init__(self,
                 rating_generators: Optional[Union[RatingGenerator, list[RatingGenerator]]] = None,
                 performances_generator: Optional[PerformancesGenerator] = None,
                 post_transformers: Optional[List[BaseTransformer]] = None,
                 predictor: BaseMLWrapper = None,
                 estimator: Optional = None,
                 other_features: Optional[list[str]] = None,
                 other_categorical_features: Optional[list[str]] = None,
                 group_predictor_by_game_team: bool = False,
                 train_split_date: Optional[pendulum.datetime] = None,
                 date_column_name: Optional[str] = None,
                 match_id_column_name: Optional[str] = None,
                 team_id_column_name: Optional[str] = None,
                 use_auto_create_performance_calculator: bool = False,
                 column_weights: Optional[Union[List[List[ColumnWeight]], list[ColumnWeight]]] = None,
                 ):


        self.rating_generators = rating_generators or []
        if isinstance(self.rating_generators, RatingGenerator):
            self.rating_generators = [self.rating_generators]

        self.post_transformers = post_transformers or []

        self.predictor = predictor
        self.estimator = estimator
        self.date_column_name = date_column_name
        self.other_features = other_features or []
        self.other_categorical_features = other_categorical_features or []
        self.group_predictor_by_game_team = group_predictor_by_game_team
        self.match_id_column_name = match_id_column_name
        self.team_id_column_name = team_id_column_name

        self.train_split_date = train_split_date
        self.use_auto_create_performance_calculator = use_auto_create_performance_calculator
        self.performances_generator = performances_generator
        self.column_weights = column_weights if isinstance(column_weights, list) else [column_weights] if column_weights else None
        if self.use_auto_create_performance_calculator:
            if not self.rating_generators:
                raise ValueError("If auto pre transformers are used, rating generators must be specified")
            column_names = [rating_generator.column_names for rating_generator in self.rating_generators]
            self.performances_generator = auto_create_performance_generator(column_weights=self.column_weights,
                                                                                 column_names=column_names)

    def create(self,
               performances_generator: Optional[PerformancesGenerator] = None,
               rating_generators: Optional[list[RatingGenerator]] = None,
               post_rating_transformers: Optional[List[BaseTransformer]] = None,
               predictor: Optional[BaseMLWrapper] = None,
               ) -> MatchPredictor:

        rating_generators = rating_generators if rating_generators is not None else self.rating_generators
        performances_generator = performances_generator if performances_generator is not None else self.performances_generator
        post_rating_transformers = post_rating_transformers if post_rating_transformers is not None else self.post_transformers
        predictor = predictor if predictor is not None else self.predictor

        return MatchPredictor(
            rating_generators=rating_generators,
            performances_generator=performances_generator,
            post_rating_transformers=post_rating_transformers,
            predictor=predictor,
            train_split_date=self.train_split_date,
            estimator=self.estimator,
            other_features=self.other_features,
            other_categorical_features=self.other_categorical_features,
            group_predictor_by_game_team=self.group_predictor_by_game_team,
            match_id_column_name=self.match_id_column_name,
            team_id_column_name=self.team_id_column_name,
            date_column_name=self.date_column_name,
        )
