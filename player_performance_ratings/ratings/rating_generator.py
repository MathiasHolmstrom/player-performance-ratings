from typing import Optional

import numpy as np
import pandas as pd

from player_performance_ratings import TeamRatingGenerator
from player_performance_ratings.ratings.enums import RatingColumnNames

from player_performance_ratings.data_structures import Match, PreMatchRating, PreMatchTeamRating, PostMatchRatingChange, \
    MatchRating, \
    PlayerRating, TeamRating, ColumnNames, TeamRatingChange


class RatingGenerator():

    def __init__(self,
                 team_rating_generator: Optional[TeamRatingGenerator] = None,
                 store_game_ratings: bool = False,
                 column_names: ColumnNames = None

                 ):
        self.team_rating_generator = team_rating_generator or TeamRatingGenerator()
        self.store_game_ratings = store_game_ratings
        self.ratings_df = None
        self.column_names = column_names
        if self.store_game_ratings and not self.column_names:
            raise ValueError("in order to store ratings, column_names must be passed to constructor")

    def generate(self, matches: list[Match], df: Optional[pd.DataFrame] = None) -> dict[RatingColumnNames, list[float]]:

        pre_match_player_rating_values = []
        pre_match_team_rating_values = []
        pre_match_opponent_rating_values = []
        team_opponent_leagues = []
        match_ids = []
        player_rating_changes = []
        player_leagues = []
        performances = []

        for match in matches:
            self._validate_match(match)
            match_team_rating_changes = self._create_match_team_rating_changes(match=match)

            for team_idx, team_rating_change in enumerate(match_team_rating_changes):
                opponent_team = match_team_rating_changes[-team_idx + 1]
                for player_idx, player_rating_change in enumerate(team_rating_change.players):

                    pre_match_player_rating_values.append(player_rating_change.pre_match_rating_value)
                    pre_match_team_rating_values.append(team_rating_change.pre_match_rating_value)
                    pre_match_opponent_rating_values.append(opponent_team.pre_match_rating_value)
                    player_rating_changes.append(player_rating_change.rating_change_value)
                    player_leagues.append(player_rating_change.league)
                    team_opponent_leagues.append(match_team_rating_changes[-team_idx + 1].league)
                    match_ids.append(match.id)
                    performances.append(player_rating_change.performance)

        rating_differences = np.array(pre_match_team_rating_values) - (
            pre_match_opponent_rating_values)
        rating_means = np.array(pre_match_team_rating_values) * 0.5 + 0.5 * np.array(
            pre_match_opponent_rating_values)

        if self.store_game_ratings:
            if df is None:
                raise ValueError(
                    "df must be passed in order to store ratings. Set store_ratings to False or pass df to method")
            self.ratings_df = df[
                [self.column_names.team_id, self.column_names.player_id, self.column_names.match_id]].assign(
                **{
                    RatingColumnNames.RATING_DIFFERENCE: rating_differences,
                    RatingColumnNames.PLAYER_LEAGUE: player_leagues,
                    RatingColumnNames.OPPONENT_LEAGUE: team_opponent_leagues,
                    RatingColumnNames.PLAYER_RATING: pre_match_player_rating_values,
                    RatingColumnNames.PLAYER_RATING_CHANGE: player_rating_changes,
                    RatingColumnNames.TEAM_RATING: pre_match_team_rating_values,
                    RatingColumnNames.OPPONENT_RATING: pre_match_opponent_rating_values,
                    RatingColumnNames.PERFORMANCE: performances,
                    RatingColumnNames.RATING_MEAN: rating_means,

                })

        return {
            RatingColumnNames.RATING_DIFFERENCE: rating_differences,
            RatingColumnNames.PLAYER_LEAGUE: player_leagues,
            RatingColumnNames.OPPONENT_LEAGUE: team_opponent_leagues,
            RatingColumnNames.PLAYER_RATING: pre_match_player_rating_values,
            RatingColumnNames.PLAYER_RATING_CHANGE: player_rating_changes,
            RatingColumnNames.MATCH_ID: match_ids,
            RatingColumnNames.TEAM_RATING: pre_match_team_rating_values,
            RatingColumnNames.OPPONENT_RATING: pre_match_opponent_rating_values,
            RatingColumnNames.RATING_MEAN: rating_means,
        }

    def _create_match_team_rating_changes(self, match: Match) -> list[TeamRatingChange]:

        team_rating_changes = []
        pre_match_rating = PreMatchRating(
            id=match.id,
            teams=self._get_pre_match_team_ratings(match=match),
            day_number=match.day_number
        )

        for team_idx in range(len(pre_match_rating.teams)):

            team_rating_change = self.team_rating_generator.generate_rating_change(day_number=match.day_number,
                                                                                      pre_match_team_ratings=pre_match_rating.teams,
                                                                                      team_idx=team_idx)
            team_rating_changes.append(team_rating_change)

        return team_rating_changes

    def update_ratings(self, team_rating_changes: list[TeamRatingChange]):

        for team_rating_change in team_rating_changes:
            self.team_rating_generator.update_rating_by_team_rating_change(team_rating_change=team_rating_change)

    def _get_pre_match_team_ratings(self, match: Match) -> list[PreMatchTeamRating]:
        pre_match_team_ratings = []
        for match_team in match.teams:
            pre_match_team_ratings.append(self.team_rating_generator.generate_pre_match_team_rating(
                match_team=match_team, match=match))

        return pre_match_team_ratings


    def _validate_match(self, match: Match):
        if len(match.teams) < 2:
            print(f"{match.id} only contains {len(match.teams)} teams")
            raise ValueError

    @property
    def player_ratings(self) -> dict[str, PlayerRating]:
        return dict(sorted(self.team_rating_generator.player_ratings.items(),
                           key=lambda item: item[1].rating_value, reverse=True))

    @property
    def team_ratings(self) -> list[TeamRating]:
        team_id_ratings: list[TeamRating] = []
        teams = self.team_rating_generator.teams
        player_ratings = self.player_ratings
        for id, team in teams.items():
            team_player_ratings = [player_ratings[p] for p in team.player_ids]
            team_rating_value = sum([p.rating_value for p in team_player_ratings]) / len(team_player_ratings)
            team_id_ratings.append(TeamRating(id=team.id, name=team.name, players=team_player_ratings,
                                              last_match_day_number=team.last_match_day_number,
                                              rating_value=team_rating_value))

        return list(sorted(team_id_ratings,
                           key=lambda team: team.rating_value, reverse=True))
