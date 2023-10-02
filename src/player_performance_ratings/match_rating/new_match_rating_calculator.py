import logging
import math
from typing import Dict, List, Union

from src.player_performance_ratings.data_structures import Match, MatchEntity, EntityRating
from src.player_performance_ratings.match_rating.match_rating_calculator import PerformancePredictor

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class BaseEntityRatingGenerator():

    def __init__(self, min_games_played: int,
                 reference_certain_sum_value: float,
                 certain_weight: float,
                 min_rating_change_multiplier_ratio: float,
                 min_rating_change_for_league: float,
                 certain_value_denom: float,
                 max_certain_sum: float,
                 certain_days_ago_multiplier: float,
                 rating_change_multiplier: float,
                 rating_change_momentum_games_count: int,
                 rating_change_momentum_multiplier: float,
                 max_days_ago: float,
                 performance_predictor: PerformancePredictor
                 ):
        self.min_games_played = min_games_played
        self.reference_certain_sum_value = reference_certain_sum_value
        self.certain_weight = certain_weight
        self.min_rating_change_multiplier_ratio = min_rating_change_multiplier_ratio
        self.min_rating_change_for_league = min_rating_change_for_league
        self.certain_value_denom = certain_value_denom
        self.max_days_ago = max_days_ago
        self.rating_change_multiplier = rating_change_multiplier
        self.max_certain_sum = max_certain_sum
        self.certain_days_ago_multiplier = certain_days_ago_multiplier
        self.rating_change_momentum_games_count = rating_change_momentum_games_count
        self.rating_change_momentum_multiplier = rating_change_momentum_multiplier
        self.entity_ratings: Dict[str, EntityRating] = {}
        self.performance_predictor = performance_predictor
        self.ratings: dict[str, float] = {}

    def generate_rating(self, match: Match, entity_id: str) -> float:
        pass

    def update_rating(self, id: str, match: Match):
        pass

    def _calculate_rating_change_multiplier(self,
                                            entity_rating: EntityRating,
                                            rating_change_multiplier: float
                                            ) -> float:
        certain_multiplier = self._calculate_certain_multiplier(
            entity_rating=entity_rating,
            rating_change_multiplier=rating_change_multiplier
        )
        multiplier = certain_multiplier * self.certain_weight + (
                1 - self.certain_weight) * rating_change_multiplier
        min_rating_change_multiplier = rating_change_multiplier * self.min_rating_change_multiplier_ratio
        return max(min_rating_change_multiplier, multiplier)

    def _calculate_certain_multiplier(self, entity_rating: EntityRating, rating_change_multiplier: float) -> float:
        net_certain_sum_value = entity_rating.certain_sum - self.reference_certain_sum_value
        certain_factor = -sigmoid_subtract_half_and_multiply2(net_certain_sum_value,
                                                              self.certain_value_denom) + MODIFIED_RATING_CHANGE_CONSTANT
        return certain_factor * rating_change_multiplier

    def _calculate_post_match_certain_sum(self,
                                          entity_rating: EntityRating,
                                          match: Match,
                                          particpation_weight: float
                                          ) -> float:

        days_ago = self._calculate_days_ago_since_last_match(entity_rating.last_match_day_number, match)
        certain_sum_value = -min(days_ago,
                                 self.max_days_ago) * self.certain_days_ago_multiplier + entity_rating.certain_sum + \
                            MATCH_CONTRIBUTION_TO_SUM_VALUE * particpation_weight

        return max(0.0, min(certain_sum_value, self.max_certain_sum))

    def _calculate_days_ago_since_last_match(self, last_match_day_number, match: Match) -> float:
        match_day_number = match.day_number
        if last_match_day_number is None:
            return 0.0

        return match_day_number - last_match_day_number



    def _get_team_rating(self, team_id: str) -> float:

    def _rating_by_id(self, id: str) -> float:
        return self.ratings[id]


class RatioEntityRatingGenerator(BaseEntityRatingGenerator):

    def generate_rating(self, match: Match, entity_id: str) -> float:
        pass

    def _calculate_rating_change(self,
                                 match_entity: MatchEntity,
                                 match: Match,
                                 rating_change_multiplier: float,
                                 ):


        pre_match_team_rating = self._calculate_team_rating(
            match_entity=match_entity,
            match=match,
        )


        entity_rating = self.entity_ratings[match_entity.entity_id]

        predicted_performance = self.performance_predictor.predict_performance(
            rating=match_entity.match_performance_rating.rating.pre_match_entity_rating,
            opponent_rating=match_entity.match_performance_rating.rating.pre_match_opponent_rating,
            team_rating=pre_match_team_rating
        )
        performance_difference = match_entity.match_performance_rating.match_performance - predicted_performance

        rating_change_multiplier = self._calculate_rating_change_multiplier(
            entity_rating=entity_rating,
            rating_change_multiplier=rating_change_multiplier
        )

        rating_change = performance_difference * rating_change_multiplier * match_entity.match_performance_rating.participation_weight
        if math.isnan(rating_change):
            logging.warning(f"rating change is nan return 0 entity id {match_entity.entity_id}")
            return 0

        st_idx = max(0, len(entity_rating.prev_rating_changes) - self.rating_change_momentum_games_count)
        prev_rating_changes = entity_rating.prev_rating_changes[st_idx:]

        rating_change += sum(prev_rating_changes) * self.rating_change_momentum_multiplier

        return rating_change

    def _calculate_team_rating(self,
                               match_entity: MatchEntity,
                               match: Match,
                               ) -> Union[None, float]:

        team_id = match_entity.team_id
        pre_match_team_rating = 0
        sum_ratio = 0

        for other_match_entity in match.entities:
            other_entity_id = other_match_entity.entity_id
            if other_entity_id not in self.entity_ratings or self.entity_ratings[
                other_entity_id].games_played < self.min_games_played:
                continue
            if other_match_entity.team_id == team_id and other_entity_id != match_entity.entity_id and \
                    other_entity_id in match_entity.match_performance_rating.ratio:
                ratio = match_entity.match_performance_rating.ratio[other_entity_id]
                sum_ratio += ratio
                pre_match_team_rating += ratio * self.entity_ratings[other_entity_id].rating

        if sum_ratio == 0:
            return self._calculate_team_rating_excl_entity_id(
                team_id=team_id,
                entity_id=match_entity.entity_id,
                match=match,
                min_games_played=self.min_games_played
            )

        return pre_match_team_rating / sum_ratio

    def _calculate_team_rating_excl_entity_id(self,
                                              team_id: str,
                                              entity_id: str,
                                              match: Match,
                                              min_games_played: int = 3
                                              ) -> Union[
        float, None]:
        sum_rating = 0
        count = 0
        for match_entity in match.entities:
            if match_entity.team_id == team_id and match_entity.entity_id != entity_id:
                if match_entity.entity_id not in self.entity_ratings or self.entity_ratings[
                    match_entity.entity_id].games_played < min_games_played:
                    continue

                sum_rating += self.entity_ratings[match_entity.entity_id].rating * \
                              match_entity.match_performance_rating.projected_participation_weight
                count += match_entity.match_performance_rating.projected_participation_weight

        if count == 0:
            return None

        return sum_rating / count


class NewMatchRatingCalculator():

    def __init__(self, entity_rating_generator: EntityRatingGenerator, min_match_ratings_for_team: int):
        self.min_match_ratings_for_team = min_match_ratings_for_team
        self.entity_rating_generator = entity_rating_generator

    def generate_pre_match_ratings(self,
                                   match: Match,
                                   calculate_participation_weight: bool
                                   ) -> Match:

    def _get_pre_match_team_ratings(self, match: Match):
        team_rating: Dict[str, float] = {}
        new_entity_ids: List[str] = []
        match_team_count = {}
        for match_entity_index, match_entity in enumerate(match.entities):
            team_id = match_entity.team_id
            if team_id not in team_rating:
                team_rating[team_id] = 0
                match_team_count[team_id] = 0

            if match_entity.entity_id not in self.entity_ratings:
                new_entity_ids.append(match_entity.entity_id)
                continue

            match_team_count[team_id] += self.entity_ratings[match_entity.entity_id].games_played

            entity_rating = self.entity_ratings[match_entity.entity_id].rating

            team_rating[team_id] += entity_rating

        for team_id, count in match_team_count.items():
            if count < self.min_match_ratings_for_team:
                team_rating[team_id] = None

        return team_rating, new_entity_ids
