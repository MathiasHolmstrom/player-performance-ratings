import math

from typing import List, Dict

from player_performance_ratings.data_structures import Match


class LeagueIdentifier:

    def __init__(self, matches_back: int = 25):
        self.matches_back = matches_back
        self.entity_to_match_leagues: Dict[str, List[str]] = {}
        self.entity_to_match_league_counts: Dict[str, Dict[str, int]] = {}
        self.entity_id_to_most_league_count: Dict[str, int] = {}
        self.entity_id_to_most_league_name: Dict[str, str] = {}

    def identify(self, player_id: str, league_match: str) -> str:
        if player_id not in self.entity_to_match_leagues:
            self.entity_to_match_leagues[player_id] = []
            self.entity_to_match_league_counts[player_id] = {}
            self.entity_id_to_most_league_count[player_id] = 0
            self.entity_id_to_most_league_name[player_id] = ""

        self.entity_to_match_leagues[player_id].append(league_match)
        if league_match not in self.entity_to_match_league_counts[player_id]:
            self.entity_to_match_league_counts[player_id][league_match] = 0

        if len(self.entity_to_match_leagues[player_id]) > self.matches_back:
            league_drop_out = self.entity_to_match_leagues[player_id][0]
            self.entity_to_match_league_counts[player_id][league_drop_out] -= 1
            self.entity_to_match_leagues[player_id] = self.entity_to_match_leagues[
                player_id
            ][1:]
            if league_drop_out == self.entity_id_to_most_league_name[player_id]:
                self.entity_id_to_most_league_count[player_id] -= 1

        self.entity_to_match_league_counts[player_id][league_match] += 1
        if (
            self.entity_to_match_league_counts[player_id][league_match]
            > self.entity_id_to_most_league_count[player_id]
        ):
            self.entity_id_to_most_league_name[player_id] = league_match

        if self.entity_id_to_most_league_name[player_id] == league_match:
            self.entity_id_to_most_league_count[player_id] += 1

        return self.entity_id_to_most_league_name[player_id]

    def _generate_teams_to_leagues(
        self, team_ids: List[str], team_league_counts: Dict[str, Dict[str, int]]
    ) -> Dict[str, str]:
        team_leagues: Dict[str, str] = {}
        for team_id in team_ids:
            league = self._identify_primary_league_for_team(team_league_counts[team_id])
            team_leagues[team_id] = league

        return team_leagues

    def _identify_primary_league_for_team(self, league_counts: Dict[str, int]) -> str:
        max_league: str = ""
        max_count = -math.inf
        for league, count in league_counts.items():
            if count > max_count:
                max_count = count
                max_league = league

        return max_league

    def _get_opponent_league(self, team_id: str, team_leagues: Dict[str, str]) -> str:

        for team_id2, league in team_leagues.items():
            if team_id2 != team_id:
                return team_leagues[team_id2]

        raise KeyError

    def _generate_team_league_counts(self, match: Match) -> Dict[str, Dict[str, int]]:

        team_league_counts: Dict[str, Dict[str, int]] = {}
        for team in match.teams:
            for player in team.players:
                if team.id not in team_league_counts:
                    team_league_counts[team.id] = {}

                if player.league not in team_league_counts[team.id]:
                    team_league_counts[team.id][player.league] = 0

                team_league_counts[team.id][player.league] += 1

        return team_league_counts

    def get_primary_league(self, match: Match) -> str:
        region_counts: dict[str, int] = {}
        max_count: int = 0
        primary_league: str = ""
        for team in match.teams:
            for player in team.players:
                region = player.league

                if region not in region_counts:
                    region_counts[region] = 0
                region_counts[region] += 1

                if region_counts[region] > max_count:
                    max_count = region_counts[region]
                    primary_league = region

        return primary_league
