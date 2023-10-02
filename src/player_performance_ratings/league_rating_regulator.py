from typing import Dict

from src.player_performance_ratings.data_structures import MatchEntity, MatchRating


class LeagueRatingAdjustor():

    def __init__(self,
                 league_rating_regularizer: float,

                 ):
        self.rating_regularizer = league_rating_regularizer
        self.league_ratings: Dict[str, float] = {}

    def update_league_ratings(self,
                              match_entity: MatchEntity,
                              match_entity_rating: MatchRating,
                              ) -> Dict[str, float]:

        if match_entity_rating.post_match_entity_rating is None:
            return self.league_ratings
        if match_entity.league == match_entity.opponent_league:
            return self.league_ratings
        rating_change = match_entity_rating.post_match_entity_rating - match_entity_rating.pre_match_entity_rating
        if match_entity.league not in self.league_ratings:
            self.league_ratings[match_entity.league] = 0
        self.league_ratings[match_entity.league] += rating_change * self.rating_regularizer

        return self.league_ratings
