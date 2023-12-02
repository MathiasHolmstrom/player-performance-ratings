import logging
import math
from typing import Dict, Optional, Tuple

from player_performance_ratings.data_structures import Match, MatchPlayer, PlayerRating, PreMatchTeamRating, \
    PreMatchPlayerRating, \
    PlayerRatingChange, Team, MatchTeam, TeamRatingChange
from player_performance_ratings.ratings.match_rating.performance_predictor import RatingDifferencePerformancePredictor, \
    PerformancePredictor
from player_performance_ratings.ratings.match_rating.start_rating.start_rating_generator import \
    StartRatingGenerator

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class TeamRatingGenerator():

    def __init__(self,
                 start_rating_generator: Optional[StartRatingGenerator] = None,
                 performance_predictor: Optional[PerformancePredictor] = None,
                 min_match_count: int = 1,
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
        self.min_match_count = min_match_count
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
        self.performance_predictor = performance_predictor or RatingDifferencePerformancePredictor()
        self.start_rating_generator = start_rating_generator or StartRatingGenerator()

        self._teams: dict[str, Team] = {}

    def generate_pre_match_team_rating(self, match: Match, match_team: MatchTeam) -> PreMatchTeamRating:

        self._teams[match_team.id] = Team(
            id=match_team.id,
            last_match_day_number=match.day_number,
            player_ids=[p.id for p in match_team.players]
        )

        pre_match_player_ratings, pre_match_player_rating_values, new_players = self._get_pre_match_player_ratings_and_new_players(
            team=match_team)
        tot_player_game_count = sum([p.games_played for p in pre_match_player_ratings])
        if len(new_players) == 0:
            return PreMatchTeamRating(
                id=match_team.id,
                players=pre_match_player_ratings,
                rating_value=sum(pre_match_player_rating_values) / len(pre_match_player_rating_values),
                league=match_team.league
            )

        elif tot_player_game_count < self.min_match_count:
            existing_team_rating = None
        else:
            existing_team_rating = sum(pre_match_player_rating_values) / len(pre_match_player_rating_values)

        new_player_pre_match_ratings = self._generate_new_player_pre_match_ratings(match=match, new_players=new_players,
                                                                                   existing_team_rating=existing_team_rating)
        pre_match_player_ratings += new_player_pre_match_ratings

        return PreMatchTeamRating(
            id=match_team.id,
            players=pre_match_player_ratings,
            rating_value=sum([p.rating_value for p in pre_match_player_ratings]) / len(pre_match_player_ratings),
            league=match_team.league
        )

    def _get_pre_match_player_ratings_and_new_players(self, team: MatchTeam) -> Tuple[
        list[PreMatchPlayerRating], list[float], list[MatchPlayer]]:

        pre_match_player_ratings = []
        player_count = 0

        new_match_entities = []
        pre_match_player_ratings_values = []

        for match_player in team.players:
            if match_player.id in self.player_ratings:

                player_rating = self._get_rating_by_id(id=match_player.id)

                pre_match_player_rating = PreMatchPlayerRating(
                    id=match_player.id,
                    rating_value=player_rating.rating_value,
                    match_performance=match_player.performance,
                    games_played=player_rating.games_played,
                    league=match_player.league
                )

            else:
                new_match_entities.append(match_player)
                continue

            player_count += self._get_rating_by_id(match_player.id).games_played

            pre_match_player_ratings.append(pre_match_player_rating)
            pre_match_player_ratings_values.append(pre_match_player_rating.rating_value)

        return pre_match_player_ratings, pre_match_player_ratings_values, new_match_entities

    def _generate_new_player_pre_match_ratings(self, match: Match, new_players: list[MatchPlayer],
                                               existing_team_rating: Optional[float]) -> list[PreMatchPlayerRating]:

        pre_match_player_ratings = []

        for match_player in new_players:
            pre_match_player_rating = self._generate_new_player_rating(match=match,
                                                                       match_player=match_player,
                                                                       existing_team_rating=existing_team_rating)

            pre_match_player_ratings.append(pre_match_player_rating)

        return pre_match_player_ratings

    def _generate_new_player_rating(self, match: Match, match_player: MatchPlayer,
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
            match_performance=match_player.performance,
            games_played=self.player_ratings[match_player.id].games_played,
            league=match_player.league
        )

    def generate_rating_change(self,
                               day_number: int,
                               pre_match_team_ratings: list[PreMatchTeamRating],
                               team_idx: int
                               ) -> TeamRatingChange:

        pre_opponent_team_rating = pre_match_team_ratings[-team_idx + 1]
        pre_team_rating = pre_match_team_ratings[team_idx]
        player_rating_changes = []
        sum_participation_weight = 0
        sum_predicted_performance = 0
        sum_performance_value = 0
        sum_rating_change = 0

        for pre_player_rating in pre_match_team_ratings[team_idx].players:

            predicted_performance = self.performance_predictor.predict_performance(
                player_rating=pre_player_rating,
                opponent_team_rating=pre_opponent_team_rating,
                team_rating=pre_team_rating
            )

            rating_change_multiplier = self._calculate_rating_change_multiplier(entity_id=pre_player_rating.id)
            performance_difference = pre_player_rating.match_performance.performance_value - predicted_performance
            rating_change_value = performance_difference * rating_change_multiplier * pre_player_rating.match_performance.participation_weight
            if math.isnan(rating_change_value):
                logging.warning(f"rating change is nan return 0 entity id {id}")
                raise ValueError

            player_rating_change = PlayerRatingChange(
                id=pre_player_rating.id,
                predicted_performance=predicted_performance,
                participation_weight=pre_player_rating.match_performance.participation_weight,
                performance=pre_player_rating.match_performance.performance_value,
                rating_change_value=rating_change_value,
                league=pre_player_rating.league,
                day_number=day_number,
                pre_match_rating_value=pre_player_rating.rating_value
            )
            player_rating_changes.append(player_rating_change)
            sum_predicted_performance += player_rating_change.predicted_performance * pre_player_rating.match_performance.participation_weight
            sum_performance_value += pre_player_rating.match_performance.performance_value * pre_player_rating.match_performance.participation_weight
            sum_rating_change += player_rating_change.rating_change_value * pre_player_rating.match_performance.participation_weight
            sum_participation_weight += pre_player_rating.match_performance.participation_weight

        rating_change_value = sum_rating_change / sum_participation_weight if sum_participation_weight > 0 else 0
        predicted_performance = sum_predicted_performance / sum_participation_weight if sum_participation_weight > 0 else 0
        performance = sum_performance_value / sum_participation_weight if sum_participation_weight > 0 else 0

        return TeamRatingChange(
            players=player_rating_changes,
            id=pre_match_team_ratings[0].id,
            rating_change_value=rating_change_value,
            predicted_performance=predicted_performance,
            pre_match_rating_value=pre_match_team_ratings[team_idx].rating_value,
            league=pre_match_team_ratings[team_idx].league,
            performance=performance
        )

    def update_rating_by_team_rating_change(self,
                                            team_rating_change: TeamRatingChange,
                                            ) -> None:

        for player_rating_change in team_rating_change.players:
            id = player_rating_change.id
            self.player_ratings[id].certain_sum = self._calculate_post_match_certain_sum(
                entity_rating=self.player_ratings[id],
                day_number=player_rating_change.day_number,
                particpation_weight=player_rating_change.participation_weight
            )

            self.player_ratings[id].rating_value += player_rating_change.rating_change_value
            self.player_ratings[id].games_played += player_rating_change.participation_weight
            self.player_ratings[id].last_match_day_number = player_rating_change.day_number

            self.start_rating_generator.update_league_ratings(rating_change=player_rating_change)
            self._update_league_ratings(rating_change=player_rating_change)

    def _update_league_ratings(self,
                               rating_change: PlayerRatingChange
                               ):

        league = rating_change.league

        if league not in self._league_rating_changes:
            self._league_rating_changes[rating_change.league] = 0
            self._league_rating_changes_count[league] = 0

        rating_change_value = rating_change.rating_change_value
        self._league_rating_changes[rating_change.league] += rating_change_value
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
                                          day_number: int,
                                          particpation_weight: float
                                          ) -> float:
        days_ago = self._calculate_days_ago_since_last_match(last_match_day_number=entity_rating.last_match_day_number,
                                                             day_number=day_number)
        certain_sum_value = -min(days_ago,
                                 self.max_days_ago) * self.certain_days_ago_multiplier + entity_rating.certain_sum + \
                            MATCH_CONTRIBUTION_TO_SUM_VALUE * particpation_weight

        return max(0.0, min(certain_sum_value, self.max_certain_sum))

    def _calculate_days_ago_since_last_match(self, last_match_day_number, day_number: int) -> float:
        match_day_number = day_number
        if last_match_day_number is None:
            return 0.0

        return match_day_number - last_match_day_number

    def _get_rating_by_id(self, id: str):
        if id not in self.player_ratings:
            raise KeyError(f"{id} not in player_ratings")
        return self.player_ratings[id]

    @property
    def teams(self) -> dict[str, Team]:
        return self._teams
