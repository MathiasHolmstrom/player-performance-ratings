from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd

from player_performance_ratings.ratings.rating_calculators import MatchRatingGenerator
from player_performance_ratings.ratings.enums import (
    RatingKnownFeatures,
    RatingHistoricalFeatures,
)

from player_performance_ratings.data_structures import (
    Match,
    PlayerRating,
    TeamRating,
    ColumnNames,
)


class RatingGenerator(ABC):

    def __init__(
        self,
        performance_column: str,
        non_estimator_known_features_out: Optional[list[RatingKnownFeatures]],
        historical_features_out: Optional[list[RatingHistoricalFeatures]],
        match_rating_generator: MatchRatingGenerator,
        seperate_player_by_position: Optional[bool] = False,
        prefix: str = "",
    ):
        self.performance_column = performance_column
        self.seperate_player_by_position = seperate_player_by_position
        self.match_rating_generator = match_rating_generator
        self._known_features_out = []
        self._historical_features_out = historical_features_out or []
        self.non_estimator_known_features_out = non_estimator_known_features_out
        self.prefix = prefix
        self.column_names = None
        self._calculated_match_ids = []

    def reset_ratings(self):
        self._calculated_match_ids = []

    @abstractmethod
    def generate_historical_by_matches(
        self,
        matches: list[Match],
        column_names: ColumnNames,
        historical_features_out: Optional[list[RatingHistoricalFeatures]] = None,
        known_features_out: Optional[list[RatingKnownFeatures]] = None,
    ) -> dict[Union[RatingKnownFeatures, RatingHistoricalFeatures], list[float]]:
        pass

    @abstractmethod
    def generate_historical(
        self,
        df: pd.DataFrame,
        column_names: ColumnNames,
        historical_features_out: Optional[list[RatingHistoricalFeatures]] = None,
        known_features_out: Optional[list[RatingKnownFeatures]] = None,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_future(
        self,
        df: Optional[pd.DataFrame],
        matches: Optional[list[Match]] = None,
        historical_features_out: Optional[list[RatingHistoricalFeatures]] = None,
        known_features_out: Optional[list[RatingKnownFeatures]] = None,
    ) -> pd.DataFrame:
        pass

    @property
    def known_features_out(self) -> list[str]:
        """
        Rating features that do not contain leakage. Thus they can be passed into an estimator
        """
        return self._known_features_out

    @property
    def features_out(
        self,
    ) -> list[str]:
        """
        Contains both known and historical features
        """
        return [*self.known_features_out, *self.historical_features_out]

    @property
    def historical_features_out(self) -> list[str]:
        """
        Rating Features that contain leakge. Thus, they must not be passed into an estimator.
        They are only inteded to be used for data-analysis
        """
        if self._historical_features_out:
            return self._historical_features_out
        return []

    @property
    def known_features_return(self) -> list[str]:
        """
        Rating features that do not contain leakage.
        Contains both features intended to be passed to the estimator
         and other features that are not intended to be passed to the estimator
        """
        if self.non_estimator_known_features_out:
            return list(
                set(self.non_estimator_known_features_out + self.known_features_out)
            )
        return self.known_features_out

    @property
    def player_ratings(self) -> dict[str, PlayerRating]:
        return dict(
            sorted(
                self.match_rating_generator.player_ratings.items(),
                key=lambda item: item[1].rating_value,
                reverse=True,
            )
        )

    @property
    def team_ratings(self) -> list[TeamRating]:
        team_id_ratings: list[TeamRating] = []
        teams = self.match_rating_generator.teams
        player_ratings = self.player_ratings
        for id, team in teams.items():
            team_player_ratings = [player_ratings[p] for p in team.player_ids]
            team_rating_value = sum(
                [p.rating_value for p in team_player_ratings]
            ) / len(team_player_ratings)
            team_id_ratings.append(
                TeamRating(
                    id=team.id,
                    name=team.name,
                    players=team_player_ratings,
                    last_match_day_number=team.last_match_day_number,
                    rating_value=team_rating_value,
                )
            )

        return list(
            sorted(team_id_ratings, key=lambda team: team.rating_value, reverse=True)
        )

    @property
    def calculated_match_ids(self) -> list[str]:
        return self._calculated_match_ids

    def _validate_match(self, match: Match):
        if len(match.teams) < 2:
            raise ValueError(f"{match.id} only contains {len(match.teams)} teams")
