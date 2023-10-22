from typing import Optional

import numpy as np
import pandas as pd

from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.match_rating.team_rating_generator import TeamRatingGenerator
from player_performance_ratings.data_structures import Match, PreMatchRating, PreMatchTeamRating, PostMatchRating, \
    MatchRating, \
    PostMatchTeamRating, PlayerRating, TeamRating, ColumnNames


class RatingGenerator():

    def __init__(self,
                 team_rating_generator: Optional[TeamRatingGenerator] = None,
                 store_game_ratings: bool = False,
                 column_names: Optional[ColumnNames] = None

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

        for match in matches:
            self._validate_match(match)

            match_rating = self._create_match_rating(match=match)

            for team_idx, team in enumerate(match_rating.pre_match_rating.teams):
                opponent_team = match_rating.pre_match_rating.teams[-team_idx + 1]
                for player_idx, player in enumerate(team.players):
                    pre_match_player_rating_values.append(player.rating_value)
                    pre_match_team_rating_values.append(team.rating_value)
                    pre_match_opponent_rating_values.append(opponent_team.rating_value)
                    player_rating_changes.append(match_rating.post_match_rating.teams[team_idx].players[
                                                     player_idx].rating_value - player.rating_value)
                    player_leagues.append(player.league)
                    team_opponent_leagues.append(match_rating.pre_match_rating.teams[-team_idx + 1].league)
                    match_ids.append(match.id)

        rating_differences = np.array(pre_match_team_rating_values) - (
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

                })

        return {
            RatingColumnNames.RATING_DIFFERENCE: rating_differences,
            RatingColumnNames.PLAYER_LEAGUE: player_leagues,
            RatingColumnNames.OPPONENT_LEAGUE: team_opponent_leagues,
            RatingColumnNames.PLAYER_RATING: pre_match_player_rating_values,
            RatingColumnNames.PLAYER_RATING_CHANGE: player_rating_changes,
            RatingColumnNames.MATCH_ID: match_ids,
            RatingColumnNames.TEAM_RATING: pre_match_team_rating_values,
            RatingColumnNames.OPPONENT_RATING: pre_match_opponent_rating_values
        }

    def _create_match_rating(self, match: Match) -> MatchRating:

        pre_match_rating = PreMatchRating(
            id=match.id,
            teams=self._get_pre_match_team_ratings(match=match),
            day_number=match.day_number
        )

        post_match_rating = PostMatchRating(
            id=match.id,
            teams=self._get_post_match_team_ratings(pre_match_rating=pre_match_rating)
        )

        return MatchRating(
            id=match.id,
            pre_match_rating=pre_match_rating,
            post_match_rating=post_match_rating
        )

    def _get_pre_match_team_ratings(self, match: Match) -> list[PreMatchTeamRating]:
        pre_match_team_ratings = []
        for match_team in match.teams:
            pre_match_team_ratings.append(self.team_rating_generator.pre_match_rating(
                match_team=match_team, match=match))

        return pre_match_team_ratings

    def _get_post_match_team_ratings(self, pre_match_rating: PreMatchRating) -> list[PostMatchTeamRating]:
        post_match_team_ratings = []
        for team_idx, _ in enumerate(pre_match_rating.teams):
            post_match_team_ratings.append(self.team_rating_generator.generate_post_match_rating(
                pre_match_team_ratings=pre_match_rating.teams, team_idx=team_idx,
                day_number=pre_match_rating.day_number))

        return post_match_team_ratings

    def _validate_match(self, match: Match):
        if len(match.teams) < 2:
            print(f"{match.id} only contains {len(match.teams)} teams")
            raise ValueError

    @property
    def player_ratings(self) -> dict[str, PlayerRating]:
        return dict(sorted(self.team_rating_generator.player_rating_generator.player_ratings.items(),
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
