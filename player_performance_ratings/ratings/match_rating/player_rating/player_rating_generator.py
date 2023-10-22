import logging
import math
from typing import Dict, Optional

from player_performance_ratings.data_structures import Match, MatchPlayer, PlayerRating, PreMatchTeamRating, PreMatchPlayerRating, \
    PostMatchPlayerRating
from player_performance_ratings.ratings.match_rating.performance_predictor import PerformancePredictor
from player_performance_ratings.ratings.match_rating.player_rating.start_rating.start_rating_generator import StartRatingGenerator

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class PlayerRatingGenerator():

    def __init__(self,
                 start_rating_generator: Optional[StartRatingGenerator] = None,
                 performance_predictor: Optional[PerformancePredictor] = None,
                 certain_weight: float = 0.9,
                 certain_days_ago_multiplier: float = 0.06,
                 max_days_ago: int = 90,
                 max_certain_sum: float = 60,
                 min_rating_change_for_league: float = 4,
                 certain_value_denom: float = 35,
                 min_rating_change_multiplier_ratio: float = 0.1,
                 reference_certain_sum_value: float = 3,
                 rating_change_multiplier: float = 50,
                 league_rating_adjustor_multiplier: float = 5,
                 league_rating_change_sum_count: int = 250
                 ):
        self.certain_weight = certain_weight
        self.certain_days_ago_multiplier = certain_days_ago_multiplier
        # TODO implement below
        self.min_rating_change_for_league = min_rating_change_for_league
        self.certain_value_denom = certain_value_denom
        self.min_rating_change_multiplier_ratio = min_rating_change_multiplier_ratio
        self.reference_certain_sum_value = reference_certain_sum_value
        self.rating_change_multiplier = rating_change_multiplier
        self.max_certain_sum = max_certain_sum
        self.league_rating_adjustor_multiplier = league_rating_adjustor_multiplier
        self.league_rating_change_sum_count = league_rating_change_sum_count
        self.max_days_ago = max_days_ago
        self.player_ratings: Dict[str, PlayerRating] = {}
        self._league_rating_changes: dict[str, float] = {}
        self._league_rating_changes_count: dict[str, float] = {}
        self.performance_predictor = performance_predictor or PerformancePredictor()
        self.start_rating_generator = start_rating_generator or StartRatingGenerator()

    def generate_pre_rating(self, match_player: MatchPlayer) -> PreMatchPlayerRating:
        player_rating = self.get_rating_by_id(id=match_player.id)
        projected_rating_value = match_player.performance.projected_participation_weight * \
                                 player_rating.rating_value
        return PreMatchPlayerRating(
            id=match_player.id,
            rating_value=player_rating.rating_value,
            projected_rating_value=projected_rating_value,
            match_performance=match_player.performance,
            certain_ratio=player_rating.certain_ratio,
            games_played=player_rating.games_played,
            league=match_player.league
        )

    def generate_new_player_rating(self, match: Match, match_player: MatchPlayer,
                                   existing_team_rating: Optional[float]) -> PreMatchPlayerRating:
        id = match_player.id

        rating_value = self.start_rating_generator.generate_rating_value(
            day_number=match.day_number,
            match_entity=match_player,
            team_rating=existing_team_rating,
        )

        self.player_ratings[match_player.id] = PlayerRating(
            id=id,
            rating_value=rating_value
        )

        return PreMatchPlayerRating(
            id=id,
            rating_value=rating_value,
            projected_rating_value=match_player.performance.projected_participation_weight * rating_value,
            match_performance=match_player.performance,
            certain_ratio=self.player_ratings[match_player.id].certain_ratio,
            games_played=self.player_ratings[match_player.id].games_played,
            league=match_player.league
        )

    def generate_post_rating(self,
                             day_number: int,
                             pre_match_player_rating: PreMatchPlayerRating,
                             pre_match_team_rating: PreMatchTeamRating,
                             pre_match_opponent_rating: PreMatchTeamRating) -> PostMatchPlayerRating:
        id = pre_match_player_rating.id

        predicted_performance = self.performance_predictor.predict_performance(
            player_rating=pre_match_player_rating,
            opponent_team_rating=pre_match_opponent_rating,
            team_rating=pre_match_team_rating
        )
        performance_difference = pre_match_player_rating.match_performance.performance_value - predicted_performance

        rating_change_multiplier = self._calculate_rating_change_multiplier(entity_id=id)

        rating_change = performance_difference * rating_change_multiplier * pre_match_player_rating.match_performance.participation_weight
        if math.isnan(rating_change):
            logging.warning(f"rating change is nan return 0 entity id {id}")
            raise ValueError

        self.player_ratings[id].rating_value += rating_change
        self.player_ratings[id].games_played += pre_match_player_rating.match_performance.participation_weight
        self.player_ratings[id].last_match_day_number = day_number

        self.start_rating_generator.update_league_ratings(day_number=day_number,
                                                          pre_match_player_rating=pre_match_player_rating,
                                                          rating_value=self.player_ratings[id].rating_value)

        post_match_player_rating = PostMatchPlayerRating(
            id=pre_match_player_rating.id,
            rating_value=self.player_ratings[id].rating_value,
            predicted_performance=predicted_performance
        )

        self._update_league_ratings(pre_match_player_rating=pre_match_player_rating,
                                    post_match_player_rating=post_match_player_rating)

        return post_match_player_rating

    def _update_league_ratings(self,
                               pre_match_player_rating: PreMatchPlayerRating,
                               post_match_player_rating: PostMatchPlayerRating,
                               ):

        league = pre_match_player_rating.league

        if league not in self._league_rating_changes:
            self._league_rating_changes[pre_match_player_rating.league] = 0
            self._league_rating_changes_count[league] = 0

        pre_match_player = pre_match_player_rating
        rating_change = post_match_player_rating.rating_value - pre_match_player.rating_value
        self._league_rating_changes[pre_match_player_rating.league] += rating_change
        self._league_rating_changes_count[league] += 1

        if self._league_rating_changes[league] > abs(self.league_rating_change_sum_count):
            for player_id in self.start_rating_generator.league_to_entity_ids[league]:
                mean_rating_change = self._league_rating_changes[league] / self._league_rating_changes_count[league]
                self.player_ratings[
                    player_id].rating_value += mean_rating_change * self.league_rating_adjustor_multiplier

            self._league_rating_changes[league] = 0

    def _calculate_rating_change_multiplier(self,
                                            entity_id: str,
                                            ) -> float:
        certain_multiplier = self._calculate_certain_multiplier(entity_rating=self.player_ratings[entity_id])
        multiplier = certain_multiplier * self.certain_weight + (
                1 - self.certain_weight) * self.rating_change_multiplier
        min_rating_change_multiplier = self.rating_change_multiplier * self.min_rating_change_multiplier_ratio
        return max(min_rating_change_multiplier, multiplier)

    def _calculate_certain_multiplier(self, entity_rating: PlayerRating) -> float:
        net_certain_sum_value = entity_rating.certain_sum - self.reference_certain_sum_value
        certain_factor = -sigmoid_subtract_half_and_multiply2(net_certain_sum_value,
                                                              self.certain_value_denom) + MODIFIED_RATING_CHANGE_CONSTANT
        return certain_factor * self.rating_change_multiplier

    def _calculate_post_match_certain_sum(self,
                                          entity_rating: PlayerRating,
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

    def get_rating_by_id(self, id: str):
        if id not in self.player_ratings:
            raise KeyError(f"{id} not in player_ratings")
        return self.player_ratings[id]
