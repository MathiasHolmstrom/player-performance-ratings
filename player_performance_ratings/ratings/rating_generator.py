from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.ratings.enums import RatingEstimatorFeatures, RatingHistoricalFeatures

from player_performance_ratings.data_structures import Match, PlayerRating, \
    TeamRating, ColumnNames


class RatingGenerator(ABC):

    def __init__(self, column_names: ColumnNames):
        self.column_names = column_names
        self._ratings_df = None

    @abstractmethod
    def generate_historical(self, df: Optional[pd.DataFrame] = None, matches: Optional[list[Match]] = None) -> dict[
        RatingEstimatorFeatures, list[float]]:
        pass

    @abstractmethod
    def generate_future(self, df: Optional[pd.DataFrame] = None, matches: Optional[list[Match]] = None) -> dict[
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
    @abstractmethod
    def estimator_features_out(self) -> list[RatingEstimatorFeatures]:
        pass

    @property
    @abstractmethod
    def features_out(self) -> list[Union[RatingEstimatorFeatures, RatingHistoricalFeatures]]:
        pass


    @property
    def ratings_df(self) -> pd.DataFrame:
        return self._ratings_df


