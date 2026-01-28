import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from spforge.data_structures import (
    PlayerRatingChange,
)

DEFAULT_START_RATING = 1000


@dataclass
class LeagueTeamRatings:
    league: str
    ratings: list[float]


class TeamStartRatingGenerator:

    def __init__(
        self,
        league_ratings: dict[str, float] | None = None,
        league_quantile: float = 0.2,
        min_count_for_percentiles: int = 50,
        max_days_ago_league_entities: int = 600,
        min_match_count_team_rating: int = 2,
        harcoded_start_rating: float | None = None,
    ):
        self.league_ratings = league_ratings or {}
        self.league_quantile = league_quantile
        self.min_count_for_percentiles = min_count_for_percentiles
        self.max_days_ago_league_entities = max_days_ago_league_entities
        self.min_match_count_team_rating = min_match_count_team_rating
        self.harcoded_start_rating = harcoded_start_rating
        if self.harcoded_start_rating is not None:
            logging.warning(
                "Hardcoded start ratings are used."
                " This will usually result in worse accuracy when new teams are expected to perform worse"
            )

        self._league_to_last_day_number: dict[str, list[Any]] = {}
        self._league_to_team_ids: dict[str, list[str]] = {}
        self._league_team_ratings: dict[str, list] = {}
        self._team_to_league: dict[str, str] = {}

    def reset(self):
        self._league_to_last_day_number = {}
        self._league_to_team_ids = {}
        self._league_team_ratings = {}
        self._team_to_league = {}

    def generate_rating_value(
        self,
        day_number: int,
        league: str,
    ) -> float:

        if self.harcoded_start_rating is not None:
            return self.harcoded_start_rating

        if league not in self.league_ratings:
            self.league_ratings[league] = DEFAULT_START_RATING

        if league not in self._league_to_team_ids:
            self._league_to_team_ids[league] = []
            self._league_to_last_day_number[league] = []

        if league not in self._league_team_ratings:
            self._league_team_ratings[league] = []

        return self._calculate_start_rating_value(
            match_day_number=day_number,
            league=league,
        )

    def _calculate_start_rating_value(
        self,
        match_day_number: int,
        league: str,
    ) -> float:
        new_team_ratings = self._get_new_teams_ratings(
            match_day_number=match_day_number,
            league=league,
        )
        region_team_count = len(new_team_ratings)
        if region_team_count < self.min_count_for_percentiles:
            return self.league_ratings[league]
        else:
            return self._start_rating_value_for_above_threshold(new_team_ratings)

    def _get_new_teams_ratings(
        self,
        match_day_number: int,
        league: str,
    ) -> list[float]:
        team_ratings: list[float] = []

        for index, last_day_number in enumerate(self._league_to_last_day_number[league]):
            days_ago = match_day_number - last_day_number

            if days_ago <= self.max_days_ago_league_entities:
                team_ratings.append(self._league_team_ratings[league][index])

        return team_ratings

    def _start_rating_value_for_above_threshold(self, team_ratings: list) -> float:
        percentile = np.percentile(team_ratings, self.league_quantile * 100)
        return float(percentile)

    def update_teams_to_leagues(self, rating_change: PlayerRatingChange):
        league = rating_change.league
        id = rating_change.id
        day_number = rating_change.day_number
        rating_value = rating_change.pre_match_rating_value + rating_change.rating_change_value

        league_data = self._league_team_ratings.setdefault(league, [])
        league_team_ids = self._league_to_team_ids.setdefault(league, [])
        league_last_day_numbers = self._league_to_last_day_number.setdefault(league, [])

        if id not in league_team_ids:
            league_data.append(rating_value)
            league_team_ids.append(id)
            league_last_day_numbers.append(day_number)
            self._team_to_league[id] = league
        else:

            index = league_team_ids.index(id)
            league_last_day_numbers[index] = day_number
            league_data[index] = rating_value

        current_team_league = self._team_to_league.get(id, league)
        if league != current_team_league:
            team_index = self._league_to_team_ids[current_team_league].index(id)

            for data_structure in (
                self._league_team_ratings[current_team_league],
                self._league_to_last_day_number[current_team_league],
                self._league_to_team_ids[current_team_league],
            ):
                del data_structure[team_index]

            self._team_to_league[id] = league

    @property
    def league_team_ratings(self) -> dict[str, list]:
        return self._league_team_ratings
