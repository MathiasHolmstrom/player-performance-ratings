import logging
import math
from typing import Dict, List, Union, Optional, Tuple

from src.player_performance_ratings.data_structures import Match, MatchPlayer, PlayerRating, MatchTeam
from src.player_performance_ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.player_performance_ratings.start_rating_calculator import StartRatingGenerator

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class BasePlayerRatingUpdater():

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
                 start_rating_generator: StartRatingGenerator,
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
        self.player_ratings: Dict[str, PlayerRating] = {}
        self.performance_predictor = performance_predictor
        self.start_rating_generator = start_rating_generator
        self.ratings: dict[str, float] = {}

    def update(self, match: Match, match_entity: MatchPlayer) -> float:
        rating_change = self._calculate_rating_change(
            match_entity=match_entity,
            match=match,
        )
        self.player_ratings[match_entity.entity_id].rating += rating_change
        self.player_ratings[
            match_entity.entity_id].games_played += match_entity.match_performance_rating.participation_weight
        return self.player_ratings[match_entity.entity_id].rating

    def _calculate_rating_change(self,
                                 match_entity: MatchPlayer,
                                 match: Match,
                                 ):

        pre_match_team_rating = self._calculate_team_rating(
            entity_id=match_entity.entity_id,
            team_id=match_entity.team_id,
            match=match,
        )

        entity_rating = self.player_ratings[match_entity.entity_id]

        predicted_performance = self.performance_predictor.predict_performance(
            rating=match_entity.match_performance_rating.rating.pre_match_entity_rating,
            opponent_rating=match_entity.match_performance_rating.rating.pre_match_opponent_rating,
            team_rating=pre_match_team_rating
        )
        performance_difference = match_entity.match_performance_rating.match_performance - predicted_performance

        rating_change_multiplier = self._calculate_rating_change_multiplier(entity_id=match_entity.entity_id)

        rating_change = performance_difference * rating_change_multiplier * match_entity.match_performance_rating.participation_weight
        if math.isnan(rating_change):
            logging.warning(f"rating change is nan return 0 entity id {match_entity.entity_id}")
            return 0

        st_idx = max(0, len(entity_rating.prev_rating_changes) - self.rating_change_momentum_games_count)
        prev_rating_changes = entity_rating.prev_rating_changes[st_idx:]

        rating_change += sum(prev_rating_changes) * self.rating_change_momentum_multiplier

        return rating_change

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

    def _calculate_team_rating(self,
                               team_id: str,
                               entity_id: str,
                               match: Match,
                               ) -> Optional[float]:
        sum_rating = 0
        count = 0
        for match_entity in match.entities:
            if match_entity.team_id == team_id and match_entity.entity_id != entity_id:
                if match_entity.entity_id not in self.player_ratings or self.player_ratings[
                    match_entity.entity_id].games_played < self.min_games_played:
                    continue

                sum_rating += self.player_ratings[match_entity.entity_id].rating * \
                              match_entity.match_performance_rating.participation_weight

                count += match_entity.match_performance_rating.participation_weight

        if count == 0:
            return None

        return sum_rating / count

    def get_rating_by_id(self, id: str):
        if id not in self.player_ratings:
            raise KeyError(f"{id} not in player_ratings")
        return self.player_ratings[id]

    def get_rating_for_new_player(self, match: Match, match_player: MatchPlayer, existing_team_rating: float) -> PlayerRating:
        id = match_player.id

        rating = self.start_rating_generator.generate_rating(
            day_number=match.day_number,
            match_entity=match_player,
            team_rating=existing_team_rating,
        )

        self.player_ratings[match_player.id] = PlayerRating(
            id=id,
            rating=rating
        )

        return self.player_ratings[match_player.id]


class ProjectedRatioEntityRatingGenerator(BasePlayerRatingUpdater):

    def _calculate_team_rating(self,
                               team_id: str,
                               entity_id: str,
                               match: Match,
                               ) -> Optional[float]:

        pre_match_team_rating = 0
        sum_ratio = 0

        for other_match_entity in match.entities:
            other_entity_id = other_match_entity.entity_id
            if other_entity_id not in self.player_ratings or self.player_ratings[
                other_entity_id].games_played < self.min_games_played:
                continue
            if other_match_entity.team_id == team_id and other_entity_id != match_entity.entity_id and \
                    other_entity_id in match_entity.match_performance_rating.ratio:
                ratio = match_entity.match_performance_rating.ratio[other_entity_id]
                sum_ratio += ratio
                pre_match_team_rating += ratio * self.player_ratings[other_entity_id].rating

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
                if match_entity.entity_id not in self.player_ratings or self.player_ratings[
                    match_entity.entity_id].games_played < min_games_played:
                    continue

                sum_rating += self.player_ratings[match_entity.entity_id].rating * \
                              match_entity.match_performance_rating.projected_participation_weight
                count += match_entity.match_performance_rating.projected_participation_weight

        if count == 0:
            return None

        return sum_rating / count


class TeamRatingGenerator():

    def __init__(self,
                 entity_rating_generator: BasePlayerRatingUpdater,
                 min_match_count: int
                 ):
        self.player_rating_generator = entity_rating_generator
        self.min_match_count = min_match_count

    def pre_match_rating(self, match: Match, team: MatchTeam) -> float:

        player_ratings, new_players = self._get_player_ratings_and_new_players(team=team)
        tot_player_game_count = sum([p.games_played for p in player_ratings])
        if len(new_players) == 0:
            return sum(player_ratings) / len(player_ratings)

        elif tot_player_game_count < self.min_match_count:
            existing_team_rating = None
        else:
            existing_team_rating = sum(player_ratings) / len(player_ratings)

        new_ratings = self._generate_new_start_ratings(match=match, new_players=new_players,
                                                       existing_team_rating=existing_team_rating)
        player_ratings += new_ratings
        return sum(player_ratings) / len(player_ratings)

    def _get_player_ratings_and_new_players(self, team: MatchTeam) -> Tuple[
        list[PlayerRating], list[MatchPlayer]]:

        player_ratings = []
        player_count = 0

        new_match_entities = []

        for match_entity in team.entites:

            entity_rating = self.player_rating_generator.get_rating_by_id(id=match_entity.id)
            if entity_rating is None:
                new_match_entities.append(match_entity)
                continue

            player_count += entity_rating.games_played
            player_ratings.append(entity_rating)

        return player_ratings, new_match_entities

    def _generate_new_start_ratings(self, match: Match, new_players: List[MatchPlayer],
                                    existing_team_rating: float) -> list[PlayerRating]:

        player_ratings = []

        for match_player in new_players:
            player_rating = self.player_rating_generator.get_rating_for_new_player(match=match, match_player=match_player,
                                                                           existing_team_rating=existing_team_rating)
            player_ratings.append(player_rating)

        return player_ratings


class NewMatchRatingCalculator():

    def __init__(self, team_rating_generator: TeamRatingGenerator):
        self.team_rating_generator = team_rating_generator

    def generate_pre_match_ratings(self,
                                   match: Match,
                                   ) -> Match:
        for match_team in match.teams:
            pre_match_pre_start_team_rating, new_entity_ids = self.team_rating_generator.pre_match_pre_start_rating(
                team=match_team)
            self.up
