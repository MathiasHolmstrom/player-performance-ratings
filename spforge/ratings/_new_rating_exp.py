import math
from typing import Tuple, Optional

import polars as pl
import time

from spforge.data_structures import MatchPlayer, MatchPerformance, MatchTeam, PreMatchPlayerRating, PlayerRating, \
    PlayerRatingChange, TeamRatingChange, PreMatchTeamRating, ColumnNames
from spforge.ratings import StartRatingGenerator, RatingDifferencePerformancePredictor
from spforge.ratings.rating_calculators.match_rating_generator import MATCH_CONTRIBUTION_TO_SUM_VALUE, \
    EXPECTED_MEAN_CONFIDENCE_SUM
from spforge.transformers.fit_transformers import PerformanceWeightsManager
from spforge.transformers.fit_transformers._performance_manager import ColumnWeight, PerformanceManager


class PlayerRatingGeneratorNew():

    def __init__(self, column_names: ColumnNames):
        self.performances_generator = PerformanceManager(
            performance_column_name='performance',
            features=['plus_minus']
        )
        self.performance_predictor = RatingDifferencePerformancePredictor()
        self.start_rating_generator = StartRatingGenerator()
        self.column_names = column_names
        self._player_ratings = {}

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:

        df = self.performances_generator.fit_transform(df)
        return self._transform(df)

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:

        df = df.with_columns([
            pl.struct(["player_id", "performance"]).alias("player_struct")
        ])

        agg_df = df.group_by(["game_id", "team_id"]).agg([
            pl.col("player_struct").alias("players_with_stats"),
        ]
        )

        match_df = agg_df.join(
            agg_df,
            on="game_id",
            how="inner",
            suffix="_opponent"
        ).filter(
            pl.col("team_id") != pl.col("team_id_opponent")
        ).unique("game_id")

        for r in match_df.iter_rows(named=True):
            existing_pre_match_player_ratings_team1, rating_values, new_players_team1 = self._create_pre_match_player_ratings_and_new_players(
                r=r,
                players_col_name='players_with_stats')
            existing_pre_match_player_ratings_team2, rating_values, new_players_team2 = self._create_pre_match_player_ratings_and_new_players(
                r=r,
                players_col_name='players_with_stats_opponent')
            new_player_pre_match_ratings_team1 = self._generate_new_player_pre_match_ratings(
                day_number=r['__day_number'],
                new_players=new_players_team1,
                team_pre_match_player_ratings=[])

    def _create_pre_match_team_rating(self,
                                      player_ids: list[str],
                                      new_player_pre_match_ratings: list[PreMatchPlayerRating],
                                      existing_pre_match_player_ratings: list[
                                          PreMatchPlayerRating]) -> PreMatchTeamRating:
        if (
                len(new_player_pre_match_ratings) > 0
                and len(existing_pre_match_player_ratings) > 0
        ):

            pre_match_player_ratings = []
            for player_id in player_ids:

                pre_match_player = [
                    p for p in new_player_pre_match_ratings if p.id == player_id
                ]
                if len(pre_match_player) == 0:
                    pre_match_player = [
                        p
                        for p in existing_pre_match_player_ratings
                        if p.id == player_id
                    ]

                pre_match_player_ratings.append(pre_match_player[0])

        elif len(new_player_pre_match_ratings) == 0:
            pre_match_player_ratings = existing_pre_match_player_ratings
        else:
            pre_match_player_ratings = new_player_pre_match_ratings

        pre_match_team_rating_value = self._generate_pre_match_team_rating_value(
            pre_match_player_ratings=pre_match_player_ratings
        )
        pre_match_team_rating_projected_value = (
            self._generate_pre_match_team_rating_projected_value(
                pre_match_player_ratings=pre_match_player_ratings
            )
        )

        return PreMatchTeamRating(
            id=r[self.column_names.match_id],
            players=pre_match_player_ratings,
            rating_value=pre_match_team_rating_value,
            projected_rating_value=pre_match_team_rating_projected_value,
            league=r[self.column_names.league],
        )

    def _create_pre_match_player_ratings_and_new_players(
            self, r: dict, players_col_name: str
    ) -> Tuple[list[PreMatchPlayerRating], list[float], list[MatchPlayer]]:

        pre_match_player_ratings = []
        player_count = 0

        new_match_entities = []
        pre_match_player_ratings_values = []

        for team_player in r[players_col_name]:
            player_id = team_player['player_id']
            match_performance = MatchPerformance(
                performance_value=team_player['performance'],
                projected_participation_weight=1,
                participation_weight=1,
            )
            if player_id in self._player_ratings:

                player_rating = self._player_ratings[player_id]

                pre_match_player_rating = PreMatchPlayerRating(
                    id=player_id,
                    rating_value=player_rating.rating_value,
                    match_performance=match_performance,
                    games_played=player_rating.games_played,
                    league=r[self.column_names.league],
                    position=r[self.column_names.position],
                )

            else:
                new_match_entities.append(
                    MatchPlayer(
                        id=player_id,
                        performance=match_performance,
                        league=r[self.column_names.league],
                        position=r[self.column_names.position],
                    )
                )
                continue

            player_count += self._player_ratings[player_id].games_played

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

            self._player_ratings[match_player.id] = PlayerRating(
                id=id, rating_value=rating_value
            )

            pre_match_player_rating = PreMatchPlayerRating(
                id=id,
                rating_value=rating_value,
                match_performance=match_player.performance,
                games_played=self._player_ratings[match_player.id].games_played,
                league=match_player.league,
                position=match_player.position,
                other=match_player.others,
            )

            pre_match_player_ratings.append(pre_match_player_rating)

        return pre_match_player_ratings

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
            self._player_ratings[id].confidence_sum = (
                self._calculate_post_match_confidence_sum(
                    entity_rating=self._player_ratings[id],
                    day_number=player_rating_change.day_number,
                    particpation_weight=player_rating_change.participation_weight,
                )
            )

            self._player_ratings[
                id
            ].rating_value += player_rating_change.rating_change_value
            self._player_ratings[
                id
            ].games_played += player_rating_change.participation_weight
            self._player_ratings[id].last_match_day_number = (
                player_rating_change.day_number
            )
            self._player_ratings[id].most_recent_team_id = team_rating_change.id

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
                self._player_ratings[player_id].rating_value += (
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
                self._player_ratings[player_id].most_recent_team_id
                and self._player_ratings[player_id].most_recent_team_id != team_id
        ):
            self._player_ratings[
                player_id
            ].confidence_sum -= self.team_id_change_confidence_sum_decrease

        min_applied_rating_change_multiplier = (
                self.rating_change_multiplier * self.min_rating_change_multiplier_ratio
        )
        confidence_change_multiplier = self.rating_change_multiplier * (
                (
                        EXPECTED_MEAN_CONFIDENCE_SUM
                        - self._player_ratings[player_id].confidence_sum
                )
                / self.confidence_value_denom
                + 1
        )
        applied_rating_change_multiplier = (
                confidence_change_multiplier * self.confidence_weight
                + (1 - self.confidence_weight) * self.rating_change_multiplier
        )

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
        if id not in self._player_ratings:
            raise KeyError(f"{id} not in player_ratings")
        return self._player_ratings[id]


if __name__ == '__main__':
    df = pl.read_parquet(
        r"C:\Users\HOM\PycharmProjects\player-performance-ratings\examples\nba\data\game_player_subsample.parquet")

    player_rating_generator = PlayerRatingGeneratorNew()
    df = player_rating_generator.fit_transform(df)
