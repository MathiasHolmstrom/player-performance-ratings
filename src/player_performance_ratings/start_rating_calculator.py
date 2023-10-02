import copy
from dataclasses import dataclass

import numpy as np
from typing import Dict, Any, List
from src.rating_model2.data_structures import MatchRating, MatchEntity, RatingType, MatchPerformanceRating

DEFAULT_START_RATING = 1000


@dataclass
class LeagueEntityRatings:
    league: str
    entity_ratings: List[float]


@dataclass
class RatingTypeLeagueEntityRatings:
    rating_type: RatingType
    league_entity_ratings: Dict[str, LeagueEntityRatings]


class StartRatingCalculator():

    def __init__(self,
                 start_league_ratings: Dict[RatingType, Dict[str, float]] = None,
                 min_count_using_percentiles: int = 150,
                 league_quantile: float = 0.2,
                 team_rating_subtract: float = 80,
                 team_weight: float = 0.2,
                 max_days_ago_league_entities: int = 120,
                 ):

        self.team_weight = team_weight
        self.max_days_ago_league_entities = max_days_ago_league_entities

        self.min_count_for_percentiles = min_count_using_percentiles
        self.team_rating_subtract = team_rating_subtract
        self.league_quantile = league_quantile
        if start_league_ratings is None:
            start_league_ratings: Dict[RatingType, Dict[str, float]] = {}
        self.region_ratings_original = start_league_ratings

        self.rating_type_to_league_ratings = copy.deepcopy(self.region_ratings_original)

        self.league_to_last_day_number: Dict[str, List[Any]] = {}
        self.league_to_entity_ids: Dict[str, List[str]] = {}
        self.entity_to_league: Dict[str, str] = {}
        self.ratings_type_to_league_entity_ratings: Dict[RatingType, RatingTypeLeagueEntityRatings] = {}

    def generate_rating(self,
                        day_number: int,
                        match_entity: MatchEntity,
                        team_rating: float,
                        rating_type: RatingType,
                        ) -> float:

        if rating_type not in self.ratings_type_to_league_entity_ratings:
            self.ratings_type_to_league_entity_ratings[rating_type] = RatingTypeLeagueEntityRatings(
                rating_type=rating_type,
                league_entity_ratings={}
            )
        if rating_type not in self.rating_type_to_league_ratings:
            self.rating_type_to_league_ratings[rating_type]: Dict[str, float] = {}

        league = match_entity.league
        if league not in self.rating_type_to_league_ratings[rating_type]:
            self.rating_type_to_league_ratings[rating_type][league] = DEFAULT_START_RATING

        if league not in self.league_to_entity_ids:
            self.league_to_entity_ids[league] = []
            self.league_to_last_day_number[league] = []

        if league not in self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings:
            self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings[league] = LeagueEntityRatings(
                league=league,
                entity_ratings=[],
            )

        if team_rating is None:
            team_weight = 0
            adjusted_team_start_rating = 0
        else:
            team_weight = self.team_weight
            adjusted_team_start_rating = team_rating - self.team_rating_subtract
        start_rating_league = self._calculate_start_rating_value(
            match_day_number=day_number,
            league=league,
            rating_type=rating_type
        )
        return start_rating_league * (1 - team_weight) + team_weight * adjusted_team_start_rating

    def _calculate_start_rating_value(self,
                                      match_day_number: int,
                                      league: str,
                                      rating_type: RatingType
                                      ) -> float:
        new_entity_ratings = self._get_new_entities_ratings(
            match_day_number=match_day_number,
            league=league,
            rating_type=rating_type
        )
        region_entity_count = len(new_entity_ratings)
        if region_entity_count < self.min_count_for_percentiles:
            return self._start_rating_value_for_below_threshold(rating_type=rating_type, league=league)
        else:
            return self._start_rating_value_for_above_threshold(new_entity_ratings)

    def _get_new_entities_ratings(self,
                                  match_day_number: int,
                                  league: str,
                                  rating_type: RatingType
                                  ) -> List[float]:
        entity_ratings: List[float] = []

        for index, last_day_number in enumerate(self.league_to_last_day_number[league]):
            days_ago = match_day_number - last_day_number

            if days_ago <= self.max_days_ago_league_entities:

                entity_ratings.append(self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings[
                                          league].entity_ratings[index])

        return entity_ratings

    def _start_rating_value_for_below_threshold(self, rating_type: RatingType, league: str) -> float:
        return self.rating_type_to_league_ratings[rating_type][league]

    def _start_rating_value_for_above_threshold(self, entity_ratings: List) -> float:
        percentile = np.percentile(entity_ratings, self.league_quantile * 100)
        return percentile

    def update_league_ratings(self,
                              day_number: int,
                              match_entity: MatchEntity
                              ):

        league = match_entity.league
        entity_id = match_entity.entity_id

        if league not in self.league_to_entity_ids:
            for rating_type in match_entity.match_performance_rating:
                if match_entity.match_performance_rating[rating_type].rating.post_match_entity_rating is None:
                    continue
                if rating_type not in self.ratings_type_to_league_entity_ratings:
                    self.ratings_type_to_league_entity_ratings[rating_type] = RatingTypeLeagueEntityRatings(
                        rating_type=rating_type,
                        league_entity_ratings={}
                    )
                self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings[
                    league] = LeagueEntityRatings(
                    league=league,
                    entity_ratings=[],
                )

            self.league_to_entity_ids[league] = []
            self.league_to_last_day_number[league] = []

        entity_has_active_rating = False
        if entity_id not in self.league_to_entity_ids[league]:
            for rating_type, match_performance_rating in match_entity.match_performance_rating.items():
                if match_performance_rating.rating.post_match_entity_rating is None:
                    continue
                entity_has_active_rating = True
                self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings[
                    league].entity_ratings.append(
                    match_performance_rating.rating.post_match_entity_rating)

            if not entity_has_active_rating:
                return
            self.league_to_entity_ids[league].append(entity_id)
            self.league_to_last_day_number[league].append(day_number)
            self.entity_to_league[entity_id] = league
        else:
            index = self.league_to_entity_ids[league].index(entity_id)
            self.league_to_last_day_number[league][index] = day_number
            for rating_type, match_performance_rating in match_entity.match_performance_rating.items():
                if  match_performance_rating.rating.post_match_entity_rating is None:
                    continue
                self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings[league].entity_ratings[
                    index] = \
                    match_performance_rating.rating.post_match_entity_rating

        current_entity_league = self.entity_to_league[entity_id]

        if league != current_entity_league:
            entity_index = self.league_to_entity_ids[current_entity_league].index(entity_id)
            for rating_type, match_performance_rating in match_entity.match_performance_rating.items():
                self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings[
                    current_entity_league].entity_ratings = \
                    self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings[
                        current_entity_league].entity_ratings[:entity_index] + \
                    self.ratings_type_to_league_entity_ratings[rating_type].league_entity_ratings[
                        current_entity_league].entity_ratings[entity_index + 1:]


            self.league_to_last_day_number[current_entity_league] = \
                self.league_to_last_day_number[current_entity_league][:entity_index] + \
                self.league_to_last_day_number[current_entity_league][entity_index + 1:]

            self.league_to_entity_ids[current_entity_league] = \
                self.league_to_entity_ids[current_entity_league][:entity_index] + \
                self.league_to_entity_ids[current_entity_league][entity_index + 1:]

            self.entity_to_league[entity_id] = league
