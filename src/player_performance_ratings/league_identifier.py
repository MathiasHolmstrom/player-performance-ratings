import math

from typing import List, Dict



class LeagueIdentifier():

    def __init__(self,
                 matches_back: int = 25
                 ):
        self.matches_back = matches_back
        self.entity_to_match_leagues: Dict[str, List[str]] = {}
        self.entity_to_match_league_counts: Dict[str, Dict[str, int]] = {}
        self.entity_id_to_most_league_count: Dict[str, int] = {}
        self.entity_id_to_most_league_name: Dict[str, str] = {}

    def update_and_return_entity_league(self, entity_id: str, league_match: str) -> str:
        if entity_id not in self.entity_to_match_leagues:
            self.entity_to_match_leagues[entity_id] = []
            self.entity_to_match_league_counts[entity_id] = {}
            self.entity_id_to_most_league_count[entity_id] = 0
            self.entity_id_to_most_league_name[entity_id] = ""

        self.entity_to_match_leagues[entity_id].append(league_match)
        if league_match not in self.entity_to_match_league_counts[entity_id]:
            self.entity_to_match_league_counts[entity_id][league_match] = 0

        if len(self.entity_to_match_leagues[entity_id]) > self.matches_back:
            league_drop_out = self.entity_to_match_leagues[entity_id][0]
            self.entity_to_match_league_counts[entity_id][league_drop_out] -= 1
            self.entity_to_match_leagues[entity_id] = self.entity_to_match_leagues[entity_id][1:]
            if league_drop_out == self.entity_id_to_most_league_name[entity_id]:
                self.entity_id_to_most_league_count[entity_id] -= 1

        self.entity_to_match_league_counts[entity_id][league_match] += 1
        if self.entity_to_match_league_counts[entity_id][league_match] > self.entity_id_to_most_league_count[entity_id]:
            self.entity_id_to_most_league_name[entity_id] = league_match

        if self.entity_id_to_most_league_name[entity_id] == league_match:
            self.entity_id_to_most_league_count[entity_id] += 1

        return self.entity_id_to_most_league_name[entity_id]
