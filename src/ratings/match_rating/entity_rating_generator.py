import logging
import math
from typing import Dict, List, Union, Optional, Tuple

from src.ratings.data_structures import Match, MatchPlayer, PlayerRating, MatchTeam, RatingUpdateParameters, \
    TeamRatingParameters, PreMatchTeamRating, PreMatchPlayerRating
from src.ratings.match_rating.match_rating_calculator import PerformancePredictor
from src.ratings.start_rating_calculator import StartRatingGenerator

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class BasePlayerRatingUpdater():

    def __init__(self,
                 update_params: RatingUpdateParameters,
                 start_rating_generator: StartRatingGenerator,
                 performance_predictor: PerformancePredictor
                 ):
        self.update_params = update_params
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
            match_entity.entity_id].games_played += match_entity.match_player_performance.participation_weight
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

        entity_rating = self.player_ratings[match_entity.id]

        predicted_performance = self.performance_predictor.predict_performance(
            rating=match_entity.match_player_performance.rating.pre_match_entity_rating,
            opponent_rating=match_entity.match_player_performance.rating.pre_match_opponent_rating,
            team_rating=pre_match_team_rating
        )
        performance_difference = match_entity.match_player_performance.match_performance - predicted_performance

        rating_change_multiplier = self._calculate_rating_change_multiplier(entity_id=match_entity.id)

        rating_change = performance_difference * rating_change_multiplier * match_entity.match_player_performance.participation_weight
        if math.isnan(rating_change):
            logging.warning(f"rating change is nan return 0 entity id {match_entity.id}")
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
                               team: MatchTeam,
                               ) -> Optional[float]:
        sum_rating = 0
        count = 0
        for match_player in team.players:
            if match_player.id != entity_id:
                if match_player.id not in self.player_ratings or self.player_ratings[
                    match_player.id].games_played < self.min_games_played:
                    continue

                sum_rating += self.player_ratings[match_player.id].rating * \
                              match_player.match_player_performance.participation_weight

                count += match_player.match_player_performance.participation_weight

        if count == 0:
            return None

        return sum_rating / count

    def get_rating_by_id(self, id: str):
        if id not in self.player_ratings:
            raise KeyError(f"{id} not in player_ratings")
        return self.player_ratings[id]

    def get_rating_for_new_player(self, match: Match, match_player: MatchPlayer,
                                  existing_team_rating: Optional[float]) -> PlayerRating:
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


class ProjectedRatioEntityRatingUpdater(BasePlayerRatingUpdater):

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
                    other_entity_id in match_entity.match_player_performance.ratio:
                ratio = match_entity.match_player_performance.ratio[other_entity_id]
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
                              match_entity.match_player_performance.projected_participation_weight
                count += match_entity.match_player_performance.projected_participation_weight

        if count == 0:
            return None

        return sum_rating / count


class TeamRatingGenerator():

    def __init__(self,
                 player_rating_updater: BasePlayerRatingUpdater,
                 params: TeamRatingParameters
                 ):
        self.player_rating_generator = player_rating_updater
        self.params = params

    def pre_match_rating(self, match: Match, team: MatchTeam) -> PreMatchTeamRating:

        pre_match_player_ratings, new_players = self._get_pre_match_player_ratings_and_new_players(team=team)
        tot_player_game_count = sum([p.games_played for p in pre_match_player_ratings])
        if len(new_players) == 0:
            return PreMatchTeamRating(
                id=team.id,
                players=pre_match_player_ratings,
                rating=sum(pre_match_player_ratings) / len(pre_match_player_ratings),
            )

        elif tot_player_game_count < self.params.min_match_count:
            existing_team_rating = None
        else:
            existing_team_rating = sum(pre_match_player_ratings) / len(pre_match_player_ratings)

        new_player_pre_match_ratings = self._generate_new_player_pre_match_ratings(match=match, new_players=new_players,
                                                                                   existing_team_rating=existing_team_rating)
        pre_match_player_ratings += new_player_pre_match_ratings
        return PreMatchTeamRating(
            id=team.id,
            players=pre_match_player_ratings,
            rating=sum(pre_match_player_ratings) / len(pre_match_player_ratings),
        )

    def _get_pre_match_player_ratings_and_new_players(self, team: MatchTeam) -> Tuple[
        dict[str, PreMatchPlayerRating], list[MatchPlayer]]:

        pre_match_player_ratings = {}
        player_count = 0

        new_match_entities = []

        for match_player in team.players:

            player_rating = self.player_rating_generator.get_rating_by_id(id=match_player.id)
            if player_rating is None:
                new_match_entities.append(match_player)
                continue

            player_count += player_rating.games_played

            pre_match_player_rating = PreMatchPlayerRating(
                id=match_player.id,
                rating=player_rating.rating,
                projected_rating=match_player.match_player_performance.projected_participation_weight * player_rating.rating
            )

            pre_match_player_ratings[match_player.id] = pre_match_player_rating

        return pre_match_player_ratings, new_match_entities

    def _generate_new_player_pre_match_ratings(self, match: Match, new_players: List[MatchPlayer],
                                               existing_team_rating: Optional[float]) -> dict[
        str, PreMatchPlayerRating]:

        pre_match_player_ratings = {}

        for match_player in new_players:
            player_rating = self.player_rating_generator.get_rating_for_new_player(match=match,
                                                                                   match_player=match_player,
                                                                                   existing_team_rating=existing_team_rating)
            pre_match_player_rating = PreMatchPlayerRating(
                id=player_rating.id,
                rating=player_rating.rating,
                projected_rating=match_player.match_player_performance.projected_participation_weight * player_rating.rating
            )
            pre_match_player_ratings[match_player.id] = pre_match_player_rating

        return pre_match_player_ratings

    def predict_performance(self, match: Match):
        for team in match.teams:
