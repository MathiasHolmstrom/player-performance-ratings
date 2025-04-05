from abc import ABC, abstractmethod
from typing import Optional, Union

from narwhals.typing import FrameT, IntoFrameT

from spforge.ratings.rating_calculators import MatchRatingGenerator
from spforge.ratings.enums import (
    RatingKnownFeatures,
    RatingUnknownFeatures,
)

from spforge.data_structures import (
    Match,
    PlayerRating,
    TeamRating,
    ColumnNames,
)


class RatingGenerator(ABC):

    def __init__(
        self,
        performance_column: str,
        features_out: Optional[list[RatingKnownFeatures]],
        non_estimator_known_features_out: Optional[list[RatingKnownFeatures]],
        unknown_features_out: Optional[list[RatingUnknownFeatures]],
        match_rating_generator: MatchRatingGenerator,
        column_names: Optional[ColumnNames],
        distinct_positions: Optional[list[str]] = None,
        seperate_player_by_position: Optional[bool] = False,
        prefix: str = "",
        suffix: str = "",
    ):
        self._features_out = features_out or []
        self._non_estimator_known_features_out = non_estimator_known_features_out or []
        self._unknown_features_out = unknown_features_out or []
        self.performance_column = performance_column
        self.seperate_player_by_position = seperate_player_by_position
        self.match_rating_generator = match_rating_generator
        self.distinct_positions = distinct_positions
        self.prefix = prefix
        self.suffix = suffix
        self.column_names = column_names
        self._calculated_match_ids = []
        self._df = None

    def reset_ratings(self):
        self._calculated_match_ids = []

    @abstractmethod
    def fit_transform(self, df: FrameT, column_names: ColumnNames) -> IntoFrameT:
        pass

    @abstractmethod
    def transform_historical(
        self,
        df: FrameT,
        column_names: ColumnNames,
    ) -> IntoFrameT:
        pass

    @abstractmethod
    def transform_future(
        self,
        df: Optional[FrameT],
        matches: Optional[list[Match]] = None,
    ) -> IntoFrameT:
        pass

    @property
    def features_out(
        self,
    ) -> list[str]:
        """
        Contains features to be passed into the estimator
        """
        if self.distinct_positions:
            return [
                self.prefix + RatingKnownFeatures.RATING_DIFFERENCE_POSITION + "_" + p
                for p in self.distinct_positions
            ] + [self.prefix + f + self.suffix for f in self._features_out]
        return [self.prefix + f + self.suffix for f in self._features_out]

    @property
    def unknown_features_out(self) -> list[str]:
        """
        Rating Features that contain leakge. Thus, they must not be passed into an estimator.
        They are only inteded to be used for data-analysis
        """

        return [self.prefix + f + self.suffix for f in self._unknown_features_out]

    @property
    def non_estimator_known_features_out(self) -> list[str]:
        """
        Known features that are not used in the estimator
        """
        return [
            self.prefix + f + self.suffix
            for f in self._non_estimator_known_features_out
        ]

    @property
    def all_rating_features_out(self) -> list[str]:
        return (
            self.features_out
            + self.unknown_features_out
            + self.non_estimator_known_features_out
        )

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

    @property
    def historical_df(self) -> FrameT:
        return self._df
