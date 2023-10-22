from dataclasses import dataclass

import numpy as np
from typing import Dict, Any, List, Optional

from player_performance_ratings.data_structures import MatchPlayer, PreMatchPlayerRating

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
                 team_weight: float = 0.2,
                 max_days_ago_league_entities: int = 120,
                 ):

        self.league_ratings = league_ratings or {}
        self.league_quantile = league_quantile
        self.min_count_for_percentiles = min_count_for_percentiles
        self.team_rating_subtract = team_rating_subtract
        self.team_weight = team_weight
        self.max_days_ago_league_entities = max_days_ago_league_entities

        self.league_to_last_day_number: Dict[str, List[Any]] = {}
        self.league_to_entity_ids: Dict[str, List[str]] = {}
        self.league_player_ratings: dict[str, list] = {}
        self.entity_to_league: Dict[str, str] = {}

    def generate_rating_value(self,
                              day_number: int,
                              match_entity: MatchPlayer,
                              team_rating: Optional[float],
                              ) -> float:

        league = match_entity.league
        if league not in self.league_ratings:
            self.league_ratings[league] = DEFAULT_START_RATING

        if league not in self.league_to_entity_ids:
            self.league_to_entity_ids[league] = []
            self.league_to_last_day_number[league] = []

        if league not in self.league_player_ratings:
            self.league_player_ratings[league] = []

        if team_rating is None:
            team_weight = 0
            adjusted_team_start_rating = 0
        else:
            team_weight = self.team_weight
            adjusted_team_start_rating = team_rating - self.team_rating_subtract
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

    def update_league_ratings(self,
                              day_number: int,
                              pre_match_player_rating: PreMatchPlayerRating,
                              rating_value: float
                              ):

        league = pre_match_player_rating.league

        id = pre_match_player_rating.id
        if league not in self.league_to_entity_ids:
            self.league_player_ratings[league] = []
            self.league_to_entity_ids[league] = []
            self.league_to_last_day_number[league] = []

        if id not in self.league_to_entity_ids[league]:
            self.league_player_ratings[league].append(rating_value)

            self.league_to_entity_ids[league].append(id)
            self.league_to_last_day_number[league].append(day_number)
            self.entity_to_league[id] = league
        else:
            index = self.league_to_entity_ids[league].index(id)
            self.league_to_last_day_number[league][index] = day_number

            self.league_player_ratings[league][index] = \
                rating_value

        current_entity_league = self.entity_to_league[id]

        if league != current_entity_league:
            entity_index = self.league_to_entity_ids[current_entity_league].index(id)

            self.league_player_ratings[current_entity_league] = \
                self.league_player_ratings[
                    current_entity_league][:entity_index] + \
                self.league_player_ratings[
                    current_entity_league][entity_index + 1:]

            self.league_to_last_day_number[current_entity_league] = \
                self.league_to_last_day_number[current_entity_league][:entity_index] + \
                self.league_to_last_day_number[current_entity_league][entity_index + 1:]

            self.league_to_entity_ids[current_entity_league] = \
                self.league_to_entity_ids[current_entity_league][:entity_index] + \
                self.league_to_entity_ids[current_entity_league][entity_index + 1:]

            self.entity_to_league[id] = league
