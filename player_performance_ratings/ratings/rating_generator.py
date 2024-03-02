from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.ratings.enums import RatingEstimatorFeatures, RatingHistoricalFeatures

from player_performance_ratings.data_structures import Match, PlayerRating, \
    TeamRating, ColumnNames


class RatingGenerator(ABC):

    def __init__(self,
                 performance_column: str,
                 estimator_features_pass_through: Optional[list[RatingEstimatorFeatures]],
                 historical_features_out: Optional[list[RatingHistoricalFeatures]]
                 ):
        self.performance_column = performance_column
        self._estimator_features_out = []
        self._historical_features_out = historical_features_out or []
        self._estimator_features_pass_through = estimator_features_pass_through
        self._ratings_df = None
        self.column_names = None

    def reset_ratings(self):
        self._ratings_df = None

    @abstractmethod
    def generate_historical(self, df: Optional[pd.DataFrame], column_names: ColumnNames,
                            matches: Optional[list[Match]] = None) -> dict[
        RatingEstimatorFeatures, list[float]]:
        pass

    @abstractmethod
    def generate_future(self, df: Optional[pd.DataFrame], matches: Optional[list[Match]] = None) -> dict[
        RatingEstimatorFeatures, list[float]]:
        pass

    @property
    @abstractmethod
    def player_ratings(self) -> dict[str, PlayerRating]:
        pass

    @property
    @abstractmethod
    def team_ratings(self) -> list[TeamRating]:
        pass

    @property
    def estimator_features_out(self) -> list[RatingEstimatorFeatures]:
        return self._estimator_features_out

    @property
    def features_out(self) -> list[Union[RatingEstimatorFeatures, RatingHistoricalFeatures]]:
        return self._estimator_features_out + self._historical_features_out

    @property
    def estimator_features_return(self) -> list[RatingEstimatorFeatures]:
        if self._estimator_features_pass_through:
            return list(set(self._estimator_features_pass_through + self.estimator_features_out))
        return self.estimator_features_out

    @property
    def ratings_df(self) -> pd.DataFrame:
        return self._ratings_df
