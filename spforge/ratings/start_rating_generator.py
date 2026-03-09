import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from spforge.data_structures import (
    MatchPlayer,
    PlayerRatingChange,
    PreMatchPlayerRating,
)

DEFAULT_START_RATING = 1000


@dataclass
class LeaguePlayerRatings:
    league: str
    ratings: list[float]


class StartRatingGenerator:
    def __init__(
        self,
        league_ratings: dict[str, float] | None = None,
        league_quantile: float = 0.2,
        min_count_for_percentiles: int = 50,
        team_rating_subtract: float = 80,
        team_weight: float = 0,
        max_days_ago_league_entities: int = 600,
        min_match_count_team_rating: int = 2,
        harcoded_start_rating: float | None = None,
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
                "Hardcoded start ratings are used."
                " This will usually result in worse accuracy when new players are expected to perform worse"
            )

        self._league_to_last_day_number: dict[str, list[Any]] = {}
        self._league_to_player_ids: dict[str, list[str]] = {}
        self._league_player_ratings: dict[str, list] = {}
        self._player_to_league: dict[str, str] = {}
        self._league_to_player_index: dict[str, dict[str, int]] = {}
        self._start_rating_cache: dict[tuple[str, int], float] = {}

    def reset(self):
        self._league_to_last_day_number = {}
        self._league_to_player_ids = {}
        self._league_player_ratings = {}
        self._player_to_league = {}
        self._league_to_player_index = {}
        self._start_rating_cache = {}

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
            self._league_to_player_index[league] = {}

        if league not in self._league_player_ratings:
            self._league_player_ratings[league] = []

        existing_team_rating = None
        if self.team_weight > 0:
            tot_player_game_count = sum([p.games_played for p in team_pre_match_player_ratings])
            if tot_player_game_count >= self.min_match_count_team_rating:
                sum_team_rating = sum(
                    player.rating_value * player.match_performance.projected_participation_weight
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

            adjusted_team_start_rating = existing_team_rating - self.team_rating_subtract
        start_rating_league = self._calculate_start_rating_value(
            match_day_number=day_number,
            league=league,
        )
        return start_rating_league * (1 - team_weight) + team_weight * adjusted_team_start_rating

    def _calculate_start_rating_value(
        self,
        match_day_number: int,
        league: str,
    ) -> float:
        cache_key = (league, match_day_number)
        cached_value = self._start_rating_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        new_player_ratings = self._get_new_players_ratings(
            match_day_number=match_day_number,
            league=league,
        )
        region_player_count = len(new_player_ratings)
        if region_player_count < self.min_count_for_percentiles:
            start_rating = self.league_ratings[league]
        else:
            start_rating = self._start_rating_value_for_above_threshold(new_player_ratings)

        self._start_rating_cache[cache_key] = start_rating
        return start_rating

    def _get_new_players_ratings(
        self,
        match_day_number: int,
        league: str,
    ) -> list[float]:
        player_ratings: list[float] = []

        for index, last_day_number in enumerate(self._league_to_last_day_number[league]):
            days_ago = match_day_number - last_day_number

            if days_ago <= self.max_days_ago_league_entities:
                player_ratings.append(self._league_player_ratings[league][index])

        return player_ratings

    def _start_rating_value_for_above_threshold(self, player_ratings: list) -> float:
        percentile = np.percentile(player_ratings, self.league_quantile * 100)
        return float(percentile)

    def update_players_to_leagues(self, rating_change: PlayerRatingChange):
        self.update_player_rating(
            player_id=rating_change.id,
            league=rating_change.league,
            day_number=rating_change.day_number,
            pre_match_rating_value=rating_change.pre_match_rating_value,
            rating_change_value=rating_change.rating_change_value,
        )

    def update_player_rating(
        self,
        player_id: str,
        league: str | None,
        day_number: int,
        pre_match_rating_value: float,
        rating_change_value: float,
    ) -> None:
        id = player_id
        rating_value = pre_match_rating_value + rating_change_value

        league_data = self._league_player_ratings.setdefault(league, [])
        league_player_ids = self._league_to_player_ids.setdefault(league, [])
        league_last_day_numbers = self._league_to_last_day_number.setdefault(league, [])
        league_player_index = self._league_to_player_index.setdefault(league, {})

        player_index = league_player_index.get(id)
        if player_index is None:
            league_data.append(rating_value)
            league_player_ids.append(id)
            league_last_day_numbers.append(day_number)
            league_player_index[id] = len(league_player_ids) - 1
            self._player_to_league[id] = league
        else:
            league_last_day_numbers[player_index] = day_number
            league_data[player_index] = rating_value

        current_player_league = self._player_to_league.get(id, league)
        if league != current_player_league:
            self._remove_player_from_league(current_player_league, id)

            self._player_to_league[id] = league

        self._invalidate_start_rating_cache(league)
        if league != current_player_league:
            self._invalidate_start_rating_cache(current_player_league)

    def _remove_player_from_league(self, league: str, player_id: str) -> None:
        league_player_ids = self._league_to_player_ids[league]
        league_last_day_numbers = self._league_to_last_day_number[league]
        league_ratings = self._league_player_ratings[league]
        league_player_index = self._league_to_player_index[league]

        remove_index = league_player_index.pop(player_id)
        last_index = len(league_player_ids) - 1

        if remove_index != last_index:
            moved_player_id = league_player_ids[last_index]
            league_player_ids[remove_index] = moved_player_id
            league_last_day_numbers[remove_index] = league_last_day_numbers[last_index]
            league_ratings[remove_index] = league_ratings[last_index]
            league_player_index[moved_player_id] = remove_index

        league_player_ids.pop()
        league_last_day_numbers.pop()
        league_ratings.pop()

    def _invalidate_start_rating_cache(self, league: str) -> None:
        keys_to_remove = [key for key in self._start_rating_cache if key[0] == league]
        for key in keys_to_remove:
            del self._start_rating_cache[key]

    @property
    def league_player_ratings(self) -> dict[str, list]:
        return self._league_player_ratings
