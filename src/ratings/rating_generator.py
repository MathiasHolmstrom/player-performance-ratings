from typing import Optional

import numpy as np

from src.ratings.data_structures import Match, PreMatchRating, PreMatchTeamRating, PostMatchRating, MatchRating, \
    MatchRatings, PostMatchTeamRating
from src.ratings.enums import RatingColumnNames
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator


class RatingGenerator():

    def __init__(self,
                 team_rating_generator: Optional[TeamRatingGenerator] = None,

                 ):
        self.team_rating_generator = team_rating_generator or TeamRatingGenerator()

    def generate(self, matches: list[Match]) -> dict[RatingColumnNames, list[float]]:

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

        return {
            RatingColumnNames.rating_difference:np.array(pre_match_team_rating_values) - (
            pre_match_opponent_rating_values),
            RatingColumnNames.player_league: player_leagues,
            RatingColumnNames.opponent_league: team_opponent_leagues,
            RatingColumnNames.player_rating: pre_match_player_rating_values,
            RatingColumnNames.player_rating_change: player_rating_changes,
            RatingColumnNames.match_id: match_ids,
            RatingColumnNames.team_rating: pre_match_team_rating_values,
            RatingColumnNames.opponent_rating: pre_match_opponent_rating_values

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
                team=match_team, match=match))

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
