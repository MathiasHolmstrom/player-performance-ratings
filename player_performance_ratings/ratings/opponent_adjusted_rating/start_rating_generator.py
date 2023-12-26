import logging
from dataclasses import dataclass

import numpy as np
from typing import Dict, Any, List, Optional

from player_performance_ratings.data_structures import MatchPlayer, PreMatchPlayerRating, PlayerRatingChange

DEFAULT_START_RATING = 1000


@dataclass
class LeagueEntityRatings:
    league: str
    entity_ratings: List[float]


class StartRatingGenerator():

    def __init__(self,
                 league_ratings: Optional[dict[str, float]] = None,
                 league_quantile: float = 0.2,
                 min_count_for_percentiles: int = 100,
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
                f"Hardcoded ratings are used. This will usually result in worse accuracy."
                f"Although can come with computational benefits if there are many different players in the dataset")

        self.league_to_last_day_number: Dict[str, List[Any]] = {}
        self.league_to_entity_ids: Dict[str, List[str]] = {}
        self.league_player_ratings: dict[str, list] = {}
        self.entity_to_league: Dict[str, str] = {}

    def generate_rating_value(self,
                              day_number: int,
                              match_entity: MatchPlayer,
                              team_pre_match_player_ratings: list[PreMatchPlayerRating],
                              ) -> float:

        if self.harcoded_start_rating is not None:
            return self.harcoded_start_rating

        league = match_entity.league
        if league not in self.league_ratings:
            self.league_ratings[league] = DEFAULT_START_RATING

        if league not in self.league_to_entity_ids:
            self.league_to_entity_ids[league] = []
            self.league_to_last_day_number[league] = []

        if league not in self.league_player_ratings:
            self.league_player_ratings[league] = []

        existing_team_rating = None
        if self.team_weight > 0:
            tot_player_game_count = sum([p.games_played for p in team_pre_match_player_ratings])
            if tot_player_game_count >= self.min_match_count_team_rating:
                sum_team_rating = sum(player.rating_value * player.match_performance.participation_weight for player in
                                      team_pre_match_player_ratings)
                sum_participation_weight = sum(
                    player.match_performance.participation_weight for player in team_pre_match_player_ratings)

                existing_team_rating = sum_team_rating / sum_participation_weight if sum_participation_weight > 0 else 0

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

    def _calculate_start_rating_value(self,
                                      match_day_number: int,
                                      league: str,
                                      ) -> float:
        new_entity_ratings = self._get_new_entities_ratings(
            match_day_number=match_day_number,
            league=league,
        )
        region_entity_count = len(new_entity_ratings)
        if region_entity_count < self.min_count_for_percentiles:
            return self.league_ratings[league]
        else:
            return self._start_rating_value_for_above_threshold(new_entity_ratings)

    def _get_new_entities_ratings(self,
                                  match_day_number: int,
                                  league: str,
                                  ) -> List[float]:
        entity_ratings: List[float] = []

        for index, last_day_number in enumerate(self.league_to_last_day_number[league]):
            days_ago = match_day_number - last_day_number

            if days_ago <= self.max_days_ago_league_entities:
                entity_ratings.append(self.league_player_ratings[
                                          league][index])

        return entity_ratings

    def _start_rating_value_for_above_threshold(self, entity_ratings: List) -> float:
        percentile = np.percentile(entity_ratings, self.league_quantile * 100)
        return percentile

    def update_league_ratings(self, rating_change: PlayerRatingChange):
        league = rating_change.league
        id = rating_change.id
        day_number = rating_change.day_number
        rating_value = rating_change.pre_match_rating_value + rating_change.rating_change_value

        league_data = self.league_player_ratings.setdefault(league, [])
        league_entity_ids = self.league_to_entity_ids.setdefault(league, [])
        league_last_day_numbers = self.league_to_last_day_number.setdefault(league, [])

        if id not in league_entity_ids:
            league_data.append(rating_value)
            league_entity_ids.append(id)
            league_last_day_numbers.append(day_number)
            self.entity_to_league[id] = league
        else:

            index = league_entity_ids.index(id)
            league_last_day_numbers[index] = day_number
            league_data[index] = rating_value

        current_entity_league = self.entity_to_league.get(id, league)
        if league != current_entity_league:
            entity_index = self.league_to_entity_ids[current_entity_league].index(id)

            for data_structure in (self.league_player_ratings[current_entity_league],
                                   self.league_to_last_day_number[current_entity_league],
                                   self.league_to_entity_ids[current_entity_league]):
                del data_structure[entity_index]

            self.entity_to_league[id] = league
