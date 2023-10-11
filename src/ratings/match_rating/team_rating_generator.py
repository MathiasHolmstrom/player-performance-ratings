from typing import Tuple, List, Optional

from src.ratings.data_structures import MatchPlayer, MatchTeam, PreMatchPlayerRating, Match, PreMatchTeamRating, \
   PostMatchTeamRating
from src.ratings.match_rating.player_rating_generator import BasePlayerRatingUpdater


class TeamRatingGenerator():

    def __init__(self,
                 player_rating_updater: BasePlayerRatingUpdater,
                 min_match_count: int = 1

                 ):
        self.player_rating_generator = player_rating_updater
        self.min_match_count = min_match_count

    def pre_match_rating(self, match: Match, team: MatchTeam) -> PreMatchTeamRating:

        pre_match_player_ratings, new_players = self._get_pre_match_player_ratings_and_new_players(team=team)
        tot_player_game_count = sum([p.games_played for p in pre_match_player_ratings])
        if len(new_players) == 0:
            return PreMatchTeamRating(
                id=team.id,
                players=pre_match_player_ratings,
                rating_value=sum([p.rating_value for p in pre_match_player_ratings]) / len(pre_match_player_ratings),
                projected_rating_value=sum([p.projected_rating_value for p in pre_match_player_ratings]) / len(
                    pre_match_player_ratings),
            )

        elif tot_player_game_count < self.min_match_count:
            existing_team_rating = None
        else:
            existing_team_rating = sum(pre_match_player_ratings) / len(pre_match_player_ratings)

        new_player_pre_match_ratings = self._generate_new_player_pre_match_ratings(match=match, new_players=new_players,
                                                                                   existing_team_rating=existing_team_rating)
        pre_match_player_ratings += new_player_pre_match_ratings
        return PreMatchTeamRating(
            id=team.id,
            players=pre_match_player_ratings,
            rating_value=sum([p.rating_value for p in pre_match_player_ratings])/len(pre_match_player_ratings),
            projected_rating_value=sum([p.projected_rating_value for p in pre_match_player_ratings]) / len(pre_match_player_ratings),
        )

    def _get_pre_match_player_ratings_and_new_players(self, team: MatchTeam) -> Tuple[
        list[PreMatchPlayerRating], list[MatchPlayer]]:

        pre_match_player_ratings = []
        player_count = 0

        new_match_entities = []

        for match_player in team.players:
            if match_player.id in self.player_rating_generator.player_ratings:
                pre_match_player_rating = self.player_rating_generator.generate_pre_rating(match_player=match_player)
            else:
                new_match_entities.append(match_player)
                continue

            player_count += self.player_rating_generator.get_rating_by_id(match_player.id).games_played

            pre_match_player_ratings.append(pre_match_player_rating)

        return pre_match_player_ratings, new_match_entities

    def _generate_new_player_pre_match_ratings(self, match: Match, new_players: List[MatchPlayer],
                                               existing_team_rating: Optional[float]) -> dict[
        str, PreMatchPlayerRating]:

        pre_match_player_ratings = {}

        for match_player in new_players:
            pre_match_player_rating = self.player_rating_generator.generate_new_player_rating(match=match,
                                                                                              match_player=match_player,
                                                                                              existing_team_rating=existing_team_rating)

            pre_match_player_ratings[match_player.id] = pre_match_player_rating

        return pre_match_player_ratings

    def generate_post_match_rating(self, pre_match_team_ratings: list[PreMatchTeamRating], team_idx: int) -> PostMatchTeamRating:
        pre_opponent_team_rating = pre_match_team_ratings[-team_idx+1]
        pre_team_rating = pre_match_team_ratings[team_idx]
        post_player_ratings = []
        post_team_rating = 0
        for pre_player_rating in pre_match_team_ratings[team_idx].players:
            post_player_rating = self.player_rating_generator.generate_post_rating(
                pre_match_player_rating=pre_player_rating,
                pre_match_team_rating=pre_team_rating,
                pre_match_opponent_rating=pre_opponent_team_rating)
            post_player_ratings.append(post_player_rating)
            post_team_rating += post_player_rating.rating_value * pre_player_rating.match_performance.participation_weight

        return PostMatchTeamRating(
            players=post_player_ratings,
            id=pre_team_rating.id,
            rating_value=post_team_rating,
        )
