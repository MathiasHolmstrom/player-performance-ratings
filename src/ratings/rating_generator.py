from dataclasses import dataclass

import pandas as pd

from src.ratings.data_structures import Match, PreMatchRating, PreMatchTeamRating, PostMatchRating, MatchRating, \
    MatchRatings
from src.ratings.match_rating.entity_rating_generator import TeamRatingGenerator


class RatingGenerator():

    def __init__(self, team_rating_generator: TeamRatingGenerator):
        self.team_rating_generator = team_rating_generator

    def generate(self, matches: list[Match]) -> MatchRatings:

        pre_match_player_ratings: list[float] = []
        pre_match_team_ratings: list[float] = []
        pre_match_opponent_ratings: list[float] = []

        for match in matches:
            match_rating = self._create_match_rating(match=match)
            pre_match_player_ratings += match_rating.pre_match_player_ratings
            pre_match_team_ratings += match_rating.pre_match_team_ratings
            pre_match_opponent_ratings += match_rating.pre_match_opponent_ratings

        return MatchRatings(
            pre_match_team_ratings=pre_match_team_ratings,
            pre_match_player_ratings=pre_match_player_ratings,
            pre_match_opponent_ratings=pre_match_opponent_ratings
        )

    def _create_match_rating(self, match: Match) -> MatchRating:
        pre_team_ratings = self._get_pre_match_team_ratings(match=match)
        self.team_rating_generator.predict_performance(pre_team_ratings)

        pre_match_rating = PreMatchRating(
            id=match.id,
            teams=self._get_pre_match_team_ratings(match=match)
        )

        post_match_rating = PostMatchRating()

        return MatchRating(
            id=match.id,
            pre_match_rating=pre_match_rating,
            post_match_rating=post_match_rating
        )

    def _get_pre_match_team_ratings(self, match: Match) -> dict[str, PreMatchTeamRating]:
        pre_match_team_ratings: dict[str, PreMatchTeamRating] = {}
        for match_team in match.teams:
            pre_match_team_ratings[match_team.id] = self.team_rating_generator.pre_match_rating(
                team=match_team, match=match)

        return pre_match_team_ratings
