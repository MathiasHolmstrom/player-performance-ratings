import logging
from dataclasses import dataclass

import numpy as np
from typing import Dict, Any, List, Optional

from player_performance_ratings.data_structures import (
    MatchPlayer,
    PreMatchPlayerRating,
    PlayerRatingChange,
    MatchTeam,
    TeamRatingChange,
)

DEFAULT_START_RATING = 1000


@dataclass
class LeaguePlayerRatings:
    league: str
    ratings: List[float]


class StartRatingGenerator:

    def __init__(
        self,
        league_ratings: Optional[dict[str, float]] = None,
        league_quantile: float = 0.2,
        min_count_for_percentiles: int = 50,
        team_rating_subtract: float = 80,
        team_weight: float = 0,
        max_days_ago_league_entities: int = 120,
        min_match_count_team_rating: int = 2,
        harcoded_start_rating: Optional[float] = None,
    ):

        self.league_ratings = league_ratings or {}
        self.league_quantile = league_quantile
        self.min_count_for_percentiles = min_count_for_percentiles
        self.team_rating_subtract = team_rating_subtract
        self.team_weight = team_weight
        self.max_days_ago_league_entities = max_days_ago_league_entities
        self.min_match_count_team_rating = min_match_count_team_rating
        self.harcoded_start_rating = harcoded_start_rating
        if self.harcoded_start_rating is not None:
            logging.warning(
                f"Hardcoded start ratings are used."
                f" This will usually result in worse accuracy when new players are expected to perform worse"
            )

        self._league_to_last_day_number: Dict[str, List[Any]] = {}
        self._league_to_player_ids: Dict[str, List[str]] = {}
        self._league_player_ratings: dict[str, list] = {}
        self._player_to_league: Dict[str, str] = {}

    def reset(self):
        self._league_to_last_day_number = {}
        self._league_to_player_ids = {}
        self._league_player_ratings = {}
        self._player_to_league = {}

    def generate_rating_value(
        self,
        day_number: int,
        match_player: MatchPlayer,
        team_pre_match_player_ratings: list[PreMatchPlayerRating],
    ) -> float:

        if self.harcoded_start_rating is not None:
            return self.harcoded_start_rating

        league = match_player.league
        if league not in self.league_ratings:
            self.league_ratings[league] = DEFAULT_START_RATING

        if league not in self._league_to_player_ids:
            self._league_to_player_ids[league] = []
            self._league_to_last_day_number[league] = []

        if league not in self._league_player_ratings:
            self._league_player_ratings[league] = []

        existing_team_rating = None
        if self.team_weight > 0:
            tot_player_game_count = sum(
                [p.games_played for p in team_pre_match_player_ratings]
            )
            if tot_player_game_count >= self.min_match_count_team_rating:
                sum_team_rating = sum(
                    player.rating_value
                    * player.match_performance.projected_participation_weight
                    for player in team_pre_match_player_ratings
                )

                sum_participation_weight = sum(
                    player.match_performance.projected_participation_weight
                    for player in team_pre_match_player_ratings
                )

                existing_team_rating = (
                    sum_team_rating / sum_participation_weight
                    if sum_participation_weight > 0
                    else 0
                )

        if existing_team_rating is None:
            team_weight = 0
            adjusted_team_start_rating = 0
        else:
            team_weight = self.team_weight

            adjusted_team_start_rating = (
                existing_team_rating - self.team_rating_subtract
            )
        start_rating_league = self._calculate_start_rating_value(
            match_day_number=day_number,
            league=league,
        )
        return (
            start_rating_league * (1 - team_weight)
            + team_weight * adjusted_team_start_rating
        )

    def _calculate_start_rating_value(
        self,
        match_day_number: int,
        league: str,
    ) -> float:
        new_player_ratings = self._get_new_players_ratings(
            match_day_number=match_day_number,
            league=league,
        )
        region_player_count = len(new_player_ratings)
        if region_player_count < self.min_count_for_percentiles:
            return self.league_ratings[league]
        else:
            return self._start_rating_value_for_above_threshold(new_player_ratings)

    def _get_new_players_ratings(
        self,
        match_day_number: int,
        league: str,
    ) -> List[float]:
        player_ratings: List[float] = []

        for index, last_day_number in enumerate(
            self._league_to_last_day_number[league]
        ):
            days_ago = match_day_number - last_day_number

            if days_ago <= self.max_days_ago_league_entities:
                player_ratings.append(self._league_player_ratings[league][index])

        return player_ratings

    def _start_rating_value_for_above_threshold(self, player_ratings: List) -> float:
        percentile = np.percentile(player_ratings, self.league_quantile * 100)
        return percentile

    def update_players_to_leagues(self, rating_change: PlayerRatingChange):
        league = rating_change.league
        id = rating_change.id
        day_number = rating_change.day_number
        rating_value = (
            rating_change.pre_match_rating_value + rating_change.rating_change_value
        )

        league_data = self._league_player_ratings.setdefault(league, [])
        league_player_ids = self._league_to_player_ids.setdefault(league, [])
        league_last_day_numbers = self._league_to_last_day_number.setdefault(league, [])

        if id not in league_player_ids:
            league_data.append(rating_value)
            league_player_ids.append(id)
            league_last_day_numbers.append(day_number)
            self._player_to_league[id] = league
        else:

            index = league_player_ids.index(id)
            league_last_day_numbers[index] = day_number
            league_data[index] = rating_value

        current_player_league = self._player_to_league.get(id, league)
        if league != current_player_league:
            player_index = self._league_to_player_ids[current_player_league].index(id)

            for data_structure in (
                self._league_player_ratings[current_player_league],
                self._league_to_last_day_number[current_player_league],
                self._league_to_player_ids[current_player_league],
            ):
                del data_structure[player_index]

            self._player_to_league[id] = league

    @property
    def league_player_ratings(self) -> dict[str, list]:
        return self._league_player_ratings
