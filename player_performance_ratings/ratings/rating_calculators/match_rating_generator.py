import logging
import math
from typing import Dict, Optional, Tuple

from player_performance_ratings.data_structures import (
    MatchPlayer,
    PlayerRating,
    PreMatchTeamRating,
    PreMatchPlayerRating,
    PlayerRatingChange,
    Team,
    MatchTeam,
    TeamRatingChange,
)
from player_performance_ratings.ratings.rating_calculators.performance_predictor import (
    RatingDifferencePerformancePredictor,
    PerformancePredictor,
)
from player_performance_ratings.ratings.rating_calculators.start_rating_generator import (
    StartRatingGenerator,
)

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
EXPECTED_MEAN_CONFIDENCE_SUM = 30


class MatchRatingGenerator:

    def __init__(
        self,
        start_rating_generator: Optional[StartRatingGenerator] = None,
        performance_predictor: Optional[PerformancePredictor] = None,
        rating_change_multiplier: float = 50,
        confidence_days_ago_multiplier: float = 0.06,
        confidence_max_days: int = 90,
        confidence_value_denom: float = 140,
        confidence_max_sum: float = 150,
        confidence_weight: float = 0.9,
        min_rating_change_multiplier_ratio: float = 0.1,
        league_rating_change_update_threshold: float = 100,
        league_rating_adjustor_multiplier: float = 0.05,
        team_id_change_confidence_sum_decrease: float = 3,
    ):
        """
        :param start_rating_generator:
            Class object for generating start rating (the rating a player receives in its very first match).

        :param performance_predictor:
            Class object for predicting performance of a player in a match.
        This is used to adjust for the current quality of opponents faced or other contexts dependant on the class used.

        :param rating_change_multiplier:
            The base multiplier for how much a player's rating changes after a match.
            Equivalent of the K-factor in Elo.


        :param confidence_days_ago_multiplier:
            Determinutes how much the confidence_sum is affected by how long ago the last match was.
            If a player has not played in a long time, the confidence in a players rating declines.
            This means his future rating changes faster. A higher confidence_days_ago_multiplier  therefore leads to more volatile future rating changes.

        :param confidence_max_days:
            A players confidence_sum decreases by confidence_days_ago_multiplier * min(confidence_max_days, days_ago_since_last_match).
            confidence_max_days therefore puts a limit on how much a players confidence_sum can decrease.
            This ensures a non-linear relationship between inactivity and our confidence in a players rating.

        :param confidence_value_denom:
            Higher confidence_value_denom results in a more volatile rating change multipler based on the confidence_sum.

        :param confidence_max_sum:
            The maximum confidence_sum a player can have.
             If confidence_max_sum is very high, a player with many historical games, can be inactive for a while and still have a high confidence_sum.
             Thus a lower confidence_max_sum increases the speed at which a players confidence_sum decreases when inactive.


        :param min_rating_change_multiplier_ratio:
            To ensure that a player's rating changes at least a little bit after a match,
            min_rating_change_multiplier_ratio ensures that the applied_rating_change_multiplier is at least a certain ratio of the rating_change_multiplier.

        :param confidence_weight:
            Determines how much the applied rating change multiplier is affected by the confidence_sum.
            The confidence_sum being how much data (and how recent) we have on a player.
            If the confidence_ratio is 0, the applied rating_change_multipler is equal to the rating_change_multiplier.
            The higher ratio it is, the more volatile the applied rating_change_multiplier is based on the data we have on a player.
            Recommended is between 0.7 and 1.0

        :param league_rating_change_update_threshold:
            All players within a league gets their rating updated when the sum of all rating changes within a league exceeds this threshold.
            A lower value means that the ratings within a league are updated more often but is more computationally expensive.

        :param league_rating_adjustor_multiplier:
            The amount of which the players within a league gets their rating updated with, is league_rating_adjustor_multiplier * mean_rating_change.
            A higher value makes the performance of other players within a league have a larger impact on a players rating - even if the player hasn't played a match.

        """

        self.confidence_weight = confidence_weight
        self.confidence_days_ago_multiplier = confidence_days_ago_multiplier
        self.confidence_value_denom = confidence_value_denom
        self.min_rating_change_multiplier_ratio = min_rating_change_multiplier_ratio
        self.rating_change_multiplier = rating_change_multiplier
        self.confidence_max_sum = confidence_max_sum
        self.league_rating_adjustor_multiplier = league_rating_adjustor_multiplier
        self.league_rating_change_update_threshold = (
            league_rating_change_update_threshold
        )
        self.confidence_max_days = confidence_max_days
        self.player_ratings: Dict[str, PlayerRating] = {}
        self._league_rating_changes: dict[Optional[str], float] = {}
        self._league_rating_changes_count: dict[str, float] = {}
        self.performance_predictor = (
            performance_predictor or RatingDifferencePerformancePredictor()
        )
        self.start_rating_generator = start_rating_generator or StartRatingGenerator()
        self.team_id_change_confidence_sum_decrease = (
            team_id_change_confidence_sum_decrease
        )

        self._teams: dict[str, Team] = {}

    def generate_pre_match_team_rating(
        self, day_number: int, match_team: MatchTeam
    ) -> PreMatchTeamRating:

        self._teams[match_team.id] = Team(
            id=match_team.id,
            last_match_day_number=day_number,
            player_ids=[p.id for p in match_team.players],
        )

        (
            existing_pre_match_player_ratings,
            pre_match_player_rating_values,
            new_players,
        ) = self._get_pre_match_player_ratings_and_new_players(team=match_team)

        new_player_pre_match_ratings = self._generate_new_player_pre_match_ratings(
            day_number=day_number,
            new_players=new_players,
            team_pre_match_player_ratings=existing_pre_match_player_ratings,
        )
        if (
            len(new_player_pre_match_ratings) > 0
            and len(existing_pre_match_player_ratings) > 0
        ):

            pre_match_player_ratings = []
            for match_player in match_team.players:

                pre_match_player = [
                    p for p in new_player_pre_match_ratings if p.id == match_player.id
                ]
                if len(pre_match_player) == 0:
                    pre_match_player = [
                        p
                        for p in existing_pre_match_player_ratings
                        if p.id == match_player.id
                    ]

                pre_match_player_ratings.append(pre_match_player[0])

        elif len(new_player_pre_match_ratings) == 0:
            pre_match_player_ratings = existing_pre_match_player_ratings
        else:
            pre_match_player_ratings = new_player_pre_match_ratings

        # pre_match_player_ratings += new_player_pre_match_ratings
        pre_match_team_rating_value = self._generate_pre_match_team_rating_value(
            pre_match_player_ratings=pre_match_player_ratings
        )
        pre_match_team_rating_projected_value = (
            self._generate_pre_match_team_rating_projected_value(
                pre_match_player_ratings=pre_match_player_ratings
            )
        )

        return PreMatchTeamRating(
            id=match_team.id,
            players=pre_match_player_ratings,
            rating_value=pre_match_team_rating_value,
            projected_rating_value=pre_match_team_rating_projected_value,
            league=match_team.league,
        )

    def _get_pre_match_player_ratings_and_new_players(
        self, team: MatchTeam
    ) -> Tuple[list[PreMatchPlayerRating], list[float], list[MatchPlayer]]:

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
                    league=match_player.league,
                    position=match_player.position,
                    other=match_player.others,
                )

            else:
                new_match_entities.append(match_player)
                continue

            player_count += self._get_rating_by_id(match_player.id).games_played

            pre_match_player_ratings.append(pre_match_player_rating)
            pre_match_player_ratings_values.append(pre_match_player_rating.rating_value)

        return (
            pre_match_player_ratings,
            pre_match_player_ratings_values,
            new_match_entities,
        )

    def _generate_new_player_pre_match_ratings(
        self,
        day_number: int,
        new_players: list[MatchPlayer],
        team_pre_match_player_ratings: list[PreMatchPlayerRating],
    ) -> list[PreMatchPlayerRating]:

        pre_match_player_ratings = []

        for match_player in new_players:
            id = match_player.id

            rating_value = self.start_rating_generator.generate_rating_value(
                day_number=day_number,
                match_player=match_player,
                team_pre_match_player_ratings=team_pre_match_player_ratings,
            )

            self.player_ratings[match_player.id] = PlayerRating(
                id=id, rating_value=rating_value
            )

            pre_match_player_rating = PreMatchPlayerRating(
                id=id,
                rating_value=rating_value,
                match_performance=match_player.performance,
                games_played=self.player_ratings[match_player.id].games_played,
                league=match_player.league,
                position=match_player.position,
                other=match_player.others,
            )

            pre_match_player_ratings.append(pre_match_player_rating)

        return pre_match_player_ratings

    def _generate_pre_match_team_rating_projected_value(
        self, pre_match_player_ratings: list[PreMatchPlayerRating]
    ) -> float:
        team_rating = sum(
            player.rating_value
            * player.match_performance.projected_participation_weight
            for player in pre_match_player_ratings
        )
        sum_participation_weight = sum(
            player.match_performance.projected_participation_weight
            for player in pre_match_player_ratings
        )

        return (
            team_rating / sum_participation_weight
            if sum_participation_weight > 0
            else 0
        )

    def _generate_pre_match_team_rating_value(
        self, pre_match_player_ratings: list[PreMatchPlayerRating]
    ) -> Optional[float]:
        if (
            len(pre_match_player_ratings) > 0
            and pre_match_player_ratings[0].match_performance.participation_weight
            is None
        ):
            return None
        team_rating = sum(
            player.rating_value * player.match_performance.participation_weight
            for player in pre_match_player_ratings
        )
        sum_participation_weight = sum(
            player.match_performance.participation_weight
            for player in pre_match_player_ratings
        )

        return (
            team_rating / sum_participation_weight
            if sum_participation_weight > 0
            else 0
        )

    def generate_rating_change(
        self,
        day_number: int,
        pre_match_team_rating: PreMatchTeamRating,
        pre_match_opponent_team_rating: PreMatchTeamRating,
    ) -> TeamRatingChange:

        player_rating_changes = []
        sum_participation_weight = 0
        sum_predicted_performance = 0
        sum_performance_value = 0
        sum_rating_change = 0

        for pre_player_rating in pre_match_team_rating.players:

            predicted_performance = self.performance_predictor.predict_performance(
                player_rating=pre_player_rating,
                opponent_team_rating=pre_match_opponent_team_rating,
                team_rating=pre_match_team_rating,
            )

            applied_rating_change_multiplier = (
                self._calculate_applied_rating_change_multiplier(
                    player_id=pre_player_rating.id, team_id=pre_match_team_rating.id
                )
            )
            performance_difference = (
                pre_player_rating.match_performance.performance_value
                - predicted_performance
            )
            rating_change_value = (
                performance_difference
                * applied_rating_change_multiplier
                * pre_player_rating.match_performance.participation_weight
            )
            if math.isnan(rating_change_value):
                raise ValueError(
                    f"rating_change_value is nan for {pre_player_rating.id}"
                )

            player_rating_change = PlayerRatingChange(
                id=pre_player_rating.id,
                predicted_performance=predicted_performance,
                participation_weight=pre_player_rating.match_performance.participation_weight,
                performance=pre_player_rating.match_performance.performance_value,
                rating_change_value=rating_change_value,
                league=pre_player_rating.league,
                day_number=day_number,
                pre_match_rating_value=pre_player_rating.rating_value,
            )
            player_rating_changes.append(player_rating_change)
            sum_predicted_performance += (
                player_rating_change.predicted_performance
                * pre_player_rating.match_performance.participation_weight
            )
            sum_performance_value += (
                pre_player_rating.match_performance.performance_value
                * pre_player_rating.match_performance.participation_weight
            )
            sum_rating_change += (
                player_rating_change.rating_change_value
                * pre_player_rating.match_performance.participation_weight
            )
            sum_participation_weight += (
                pre_player_rating.match_performance.participation_weight
            )

        predicted_performance = (
            sum_predicted_performance / sum_participation_weight
            if sum_participation_weight > 0
            else 0
        )
        performance = (
            sum_performance_value / sum_participation_weight
            if sum_participation_weight > 0
            else 0
        )

        return TeamRatingChange(
            players=player_rating_changes,
            id=pre_match_team_rating.id,
            rating_change_value=sum_rating_change,
            predicted_performance=predicted_performance,
            pre_match_projected_rating_value=pre_match_team_rating.rating_value,
            league=pre_match_team_rating.league,
            performance=performance,
        )

    def update_rating_by_team_rating_change(
        self,
        team_rating_change: TeamRatingChange,
        opponent_team_rating_change: TeamRatingChange,
    ) -> None:

        for player_rating_change in team_rating_change.players:
            id = player_rating_change.id
            self.player_ratings[id].confidence_sum = (
                self._calculate_post_match_confidence_sum(
                    entity_rating=self.player_ratings[id],
                    day_number=player_rating_change.day_number,
                    particpation_weight=player_rating_change.participation_weight,
                )
            )

            self.player_ratings[
                id
            ].rating_value += player_rating_change.rating_change_value
            self.player_ratings[
                id
            ].games_played += player_rating_change.participation_weight
            self.player_ratings[id].last_match_day_number = (
                player_rating_change.day_number
            )
            self.player_ratings[id].most_recent_team_id = team_rating_change.id

            self.start_rating_generator.update_players_to_leagues(
                rating_change=player_rating_change
            )
            if player_rating_change.league != opponent_team_rating_change.league:
                self._update_player_ratings_from_league_changes(
                    rating_change=player_rating_change
                )

    def _update_player_ratings_from_league_changes(
        self, rating_change: PlayerRatingChange
    ):

        league = rating_change.league

        if league not in self._league_rating_changes:
            self._league_rating_changes[rating_change.league] = 0
            self._league_rating_changes_count[league] = 0

        rating_change_value = rating_change.rating_change_value
        self._league_rating_changes[rating_change.league] += rating_change_value
        self._league_rating_changes_count[league] += 1

        if self._league_rating_changes[league] > abs(
            self.league_rating_change_update_threshold
        ):
            for player_id in self.start_rating_generator._league_to_player_ids[league]:
                self.player_ratings[player_id].rating_value += (
                    self._league_rating_changes[league]
                    * self.league_rating_adjustor_multiplier
                )

            self._league_rating_changes[league] = 0

    def _calculate_applied_rating_change_multiplier(
        self,
        player_id: str,
        team_id: str,
    ) -> float:
        if (
            self.player_ratings[player_id].most_recent_team_id
            and self.player_ratings[player_id].most_recent_team_id != team_id
        ):
            self.player_ratings[
                player_id
            ].confidence_sum -= self.team_id_change_confidence_sum_decrease

        min_applied_rating_change_multiplier = (
            self.rating_change_multiplier * self.min_rating_change_multiplier_ratio
        )
        confidence_change_multiplier = self.rating_change_multiplier * (
            (
                EXPECTED_MEAN_CONFIDENCE_SUM
                - self.player_ratings[player_id].confidence_sum
            )
            / self.confidence_value_denom
            + 1
        )
        applied_rating_change_multiplier = (
            confidence_change_multiplier * self.confidence_weight
            + (1 - self.confidence_weight) * self.rating_change_multiplier
        )

        # net_certain_sum_value = self.player_ratings[
        #                              player_id].confidence_sum - 20
        #  certain_factor = -(1 / (1 + math.exp(-net_certain_sum_value / 14)) - 0.5) * 2 + 1
        #   certain_multiplier = certain_factor * self.rating_change_multiplier
        #  multiplier = certain_multiplier * self.confidence_weight + (
        #          1 - self.confidence_weight) * self.rating_change_multiplier

        #   min_rating_change_multiplier = self.rating_change_multiplier * self.min_rating_change_multiplier_ratio
        return max(
            min_applied_rating_change_multiplier, applied_rating_change_multiplier
        )

    #     return max(min_applied_rating_change_multiplier, applied_rating_change_multiplier)

    def _calculate_post_match_confidence_sum(
        self, entity_rating: PlayerRating, day_number: int, particpation_weight: float
    ) -> float:
        days_ago = self._calculate_days_ago_since_last_match(
            last_match_day_number=entity_rating.last_match_day_number,
            day_number=day_number,
        )
        confidence_sum_value = (
            -min(days_ago, self.confidence_max_days)
            * self.confidence_days_ago_multiplier
            + entity_rating.confidence_sum
            + MATCH_CONTRIBUTION_TO_SUM_VALUE * particpation_weight
        )

        return max(0.0, min(confidence_sum_value, self.confidence_max_sum))

    def _calculate_days_ago_since_last_match(
        self, last_match_day_number, day_number: int
    ) -> float:
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
