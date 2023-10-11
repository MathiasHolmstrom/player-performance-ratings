from dataclasses import dataclass

import pandas as pd

from src.ratings.data_structures import Match, PreMatchRating, PreMatchTeamRating, PostMatchRating, MatchRating, \
    MatchRatings, PostMatchTeamRating
from src.ratings.league_identifier import LeagueIdentifier
from src.ratings.match_rating.team_rating_generator import TeamRatingGenerator


class RatingGenerator():

    def __init__(self, team_rating_generator: TeamRatingGenerator, league_identifier: LeagueIdentifier):
        self.team_rating_generator = team_rating_generator
        self.league_identifier = league_identifier

    def generate(self, matches: list[Match]) -> MatchRatings:

        pre_match_player_rating_values = []
        pre_match_team_rating_values = []
        pre_match_opponent_rating_values = []

        for match in matches:
            self._validate_match(match)
            match = self.league_identifier.update_entity_leagues(match=match)
            match.league = self.league_identifier.get_primary_league(match)
            match_rating = self._create_match_rating(match=match)
            for team_idx, team in enumerate(match_rating.pre_match_rating.teams):
                opponent_team = match_rating.pre_match_rating.teams[-team_idx + 1]
                for player in team.players:
                    pre_match_player_rating_values.append(player.rating_value)
                    pre_match_team_rating_values.append(team.rating_value)
                    pre_match_opponent_rating_values.append(opponent_team.rating_value)

        return MatchRatings(
            pre_match_team_rating_values=pre_match_team_rating_values,
            pre_match_player_rating_values=pre_match_team_rating_values,
            pre_match_opponent_rating_values=pre_match_opponent_rating_values
        )

    def _create_match_rating(self, match: Match) -> MatchRating:

        pre_match_rating = PreMatchRating(
            id=match.id,
            teams=self._get_pre_match_team_ratings(match=match)
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
                pre_match_team_ratings=pre_match_rating.teams, team_idx=team_idx))

        return post_match_team_ratings


    def _validate_match(self, match: Match):
        if len(match.team_ids) < 2:
            raise ValueError


