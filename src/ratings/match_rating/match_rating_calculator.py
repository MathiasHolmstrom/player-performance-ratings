import copy
import logging
import math
import time
from typing import Dict, List, Union
import math

from src.ratings.data_structures import PlayerRating, Match, MatchPlayer, PerformancePredictorParameters

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
MODIFIED_RATING_CHANGE_CONSTANT = 1
CERTAIN_SUM = 'certain_sum'


def sigmoid_subtract_half_and_multiply2(value: float, x: float) -> float:
    return (1 / (1 + math.exp(-value / x)) - 0.5) * 2


class PerformancePredictor:

    def __init__(self,
                 params: PerformancePredictorParameters,
                 ):
        self.params = params

    def predict_performance(self, rating: float, opponent_rating: float, team_rating: float = 0) -> float:
        rating_difference = rating - opponent_rating
        if team_rating is not None:
            rating_diff_team_from_entity = team_rating - rating
            team_rating_diff = team_rating - opponent_rating
        else:
            rating_diff_team_from_entity = 0
            team_rating_diff = 0

        value = self.rating_diff_coef * rating_difference + \
                self.rating_diff_team_from_entity_coef * rating_diff_team_from_entity + team_rating_diff * self.team_rating_diff_coef
        prediction = (math.exp(value)) / (1 + math.exp(value))
        if prediction > self.max_predict_value:
            return self.max_predict_value
        elif prediction < (1 - self.max_predict_value):
            return (1 - self.max_predict_value)
        return prediction


class RatingMeanPerformancePredictor:

    def __init__(self,
                 rating_diff_coef,
                 rating_diff_team_from_entity_coef,
                 team_rating_diff_coef: float,
                 max_predict_value: float = 1,
                 last_sample_count: int = 1500
                 ):
        self.rating_diff_coef = rating_diff_coef
        self.team_rating_diff_coef = team_rating_diff_coef
        self.rating_diff_team_from_entity_coef = rating_diff_team_from_entity_coef
        self.max_predict_value = max_predict_value
        self.last_sample_count = last_sample_count
        self.sum_ratings = []
        self.sum_rating = 0
        self.rating_count = 0

    def predict_performance(self, rating: float, opponent_rating: float, team_rating: float = 0) -> float:

        self.sum_ratings.append(rating)
        self.rating_count += 1
        self.sum_rating += rating
        start_index = max(0, len(self.sum_ratings) - self.last_sample_count)
        self.sum_ratings = self.sum_ratings[start_index:]
        #  average_rating = sum(self.sum_ratings) / len(self.sum_ratings)
        average_rating = self.sum_rating / self.rating_count
        mean_rating = rating * 0.5 + opponent_rating * 0.5 - average_rating

        value = self.rating_diff_coef * mean_rating
        prediction = (math.exp(value)) / (1 + math.exp(value))
        if prediction > self.max_predict_value:
            return self.max_predict_value
        elif prediction < (1 - self.max_predict_value):
            return (1 - self.max_predict_value)
        return prediction


class MatchRatingCalculatorMixin:

    def __init__(self,
                 reference_certain_sum_value: float,
                 certain_weight: float,
                 min_rating_change_multiplier_ratio: float,
                 min_rating_change_for_league: float,
                 certain_value_denom: float,
                 max_certain_sum: float,
                 certain_days_ago_multiplier: float,
                 rating_change_multiplier: float,
                 max_days_ago: float,
                 offense_rating_change_multiplier: float,
                 defense_rating_change_multiplier: float,
                 past_team_ratings_for_average_rating: int,
                 rating_change_momentum_games_count: int,
                 rating_change_momentum_multiplier: float,
                 start_rating_calculator: StartRatingCalculator,
                 league_rating_adjustor: LeagueRatingAdjustor,
                 performance_predictor: PerformancePredictor
                 ):

        self.rating_change_momentum_games_count = rating_change_momentum_games_count
        self.rating_change_momentum_multiplier = rating_change_momentum_multiplier
        self.reference_certain_sum_value = reference_certain_sum_value
        self.certain_weight = certain_weight
        self.past_team_ratings_for_average_rating = past_team_ratings_for_average_rating
        self.min_rating_change_multiplier_ratio = min_rating_change_multiplier_ratio
        self.min_rating_change_for_league = min_rating_change_for_league
        self.certain_value_denom = certain_value_denom
        self.max_days_ago = max_days_ago
        self.offense_rating_change_multiplier = offense_rating_change_multiplier
        self.defense_rating_change_multiplier = defense_rating_change_multiplier
        self.rating_change_multiplier = rating_change_multiplier
        self.max_certain_sum = max_certain_sum
        self.certain_days_ago_multiplier = certain_days_ago_multiplier
        self.start_rating_calculator = start_rating_calculator
        self.league_rating_adjustor = league_rating_adjustor
        self.performance_predictor = performance_predictor
        self.entity_ratings: Dict[str, PlayerRating] = {}

    def _calculate_rating_change_multiplier(self,
                                            entity_rating: PlayerRating,
                                            rating_change_multiplier: float
                                            ) -> float:
        certain_multiplier = self._calculate_certain_multiplier(
            entity_rating=entity_rating,
            rating_change_multiplier=rating_change_multiplier
        )
        multiplier = certain_multiplier * self.certain_weight + (
                1 - self.certain_weight) * rating_change_multiplier
        min_rating_change_multiplier = rating_change_multiplier * self.min_rating_change_multiplier_ratio
        return max(min_rating_change_multiplier, multiplier)

    def _calculate_certain_multiplier(self, entity_rating: PlayerRating, rating_change_multiplier: float) -> float:
        net_certain_sum_value = entity_rating.certain_sum - self.reference_certain_sum_value
        certain_factor = -sigmoid_subtract_half_and_multiply2(net_certain_sum_value,
                                                              self.certain_value_denom) + MODIFIED_RATING_CHANGE_CONSTANT
        return certain_factor * rating_change_multiplier

    def _calculate_post_match_certain_sum(self,
                                          entity_rating: PlayerRating,
                                          match: Match,
                                          particpation_weight: float
                                          ) -> float:

        days_ago = self._calculate_days_ago_since_last_match(entity_rating.last_match_day_number, match)
        certain_sum_value = -min(days_ago,
                                 self.max_days_ago) * self.certain_days_ago_multiplier + entity_rating.certain_sum + \
                            MATCH_CONTRIBUTION_TO_SUM_VALUE * particpation_weight

        return max(0.0, min(certain_sum_value, self.max_certain_sum))

    def _calculate_days_ago_since_last_match(self, last_match_day_number, match: Match) -> float:
        match_day_number = match.day_number
        if last_match_day_number is None:
            return 0.0

        return match_day_number - last_match_day_number

    def _calculate_team_rating_by_rating_type_and_excluding_entity_id(self,
                                                                      team_id: str,
                                                                      entity_id: str,
                                                                      match: Match,
                                                                      min_games_played: int,
                                                                      use_projected_participation_weight: bool
                                                                      ) -> Union[
        float, None]:
        sum_rating = 0
        count = 0
        for match_entity in match.entities:
            if match_entity.team_id == team_id and match_entity.entity_id != entity_id:
                if match_entity.entity_id not in self.entity_ratings or self.entity_ratings[
                    match_entity.entity_id].games_played < min_games_played:
                    continue
                if use_projected_participation_weight:
                    sum_rating += self.entity_ratings[match_entity.entity_id].rating * \
                                  match_entity.match_player_performance.projected_participation_weight
                else:
                    if match_entity.match_player_performance.participation_weight is None:
                        continue
                    sum_rating += self.entity_ratings[match_entity.entity_id].rating * \
                                  match_entity.match_player_performance.participation_weight

                if use_projected_participation_weight:
                    count += match_entity.match_player_performance.projected_participation_weight
                else:
                    count += match_entity.match_player_performance.participation_weight

        #  elif match_entity.entity_id == match_entity.team_id and len(match.entities) == len(
        #         match.team_ids) and match_entity.team_id == team_id:
        #      return self.entity_ratings[match_entity.entity_id].ratings[rating_type]

        if count == 0:
            return None
        return sum_rating / count

    def _calculate_team_rating_by_rating_type_and_excluding_entity_id_using_projected_participation_weight(self,
                                                                                                           team_id: str,
                                                                                                           entity_id: str,
                                                                                                           match: Match,
                                                                                                           min_games_played: int = 3
                                                                                                           ) -> Union[
        float, None]:
        sum_rating = 0
        count = 0
        for match_entity in match.entities:
            if match_entity.team_id == team_id and match_entity.entity_id != entity_id:
                if match_entity.entity_id not in self.entity_ratings or self.entity_ratings[
                    match_entity.entity_id].games_played < min_games_played:
                    continue

                sum_rating += self.entity_ratings[match_entity.entity_id].rating * \
                              match_entity.match_player_performance.projected_participation_weight
                count += match_entity.match_player_performance.projected_participation_weight

        if count == 0:
            return None

        return sum_rating / count

    def update_entity_ratings_by_league_result_and_rating_type(self, match: Match):
        for match_entity in match.entities:
            entity_id = match_entity.entity_id
            if match_entity.league != match_entity.opponent_league:
                league_ratings_change = self.league_rating_adjustor.update_league_ratings(
                    match_entity_rating=match_entity.match_player_performance.rating,
                    match_entity=match_entity
                )

                self._update_all_entity_ratings_for_league(
                    league_ratings_change=league_ratings_change,
                    current_entity_id=entity_id,
                )

    def _update_all_entity_ratings_for_league(self,
                                              league_ratings_change: Dict[str, float],
                                              current_entity_id: str,
                                              ):
        for league, rating_change in league_ratings_change.items():
            if abs(rating_change) > self.min_rating_change_for_league:
                entity_ids = self.start_rating_calculator.league_to_entity_ids[league]
                for entity_id in entity_ids:
                    if entity_id == current_entity_id:
                        continue

                    self.entity_ratings[entity_id].rating += rating_change
                self.league_rating_adjustor.league_ratings[league] = 0

    def _calculate_team_rating_by_ratio(self,
                                        match_entity: MatchPlayer,
                                        match: Match,
                                        min_games_played: int,
                                        ) -> Union[None, float]:

        team_id = match_entity.team_id
        pre_match_team_rating = 0
        sum_ratio = 0

        for other_match_entity in match.entities:
            other_entity_id = other_match_entity.entity_id
            if other_entity_id not in self.entity_ratings or self.entity_ratings[
                other_entity_id].games_played < min_games_played:
                continue
            if other_match_entity.team_id == team_id and other_entity_id != match_entity.entity_id and \
                    other_entity_id in match_entity.match_player_performance.ratio:
                ratio = match_entity.match_player_performance.ratio[other_entity_id]
                sum_ratio += ratio
                pre_match_team_rating += ratio * self.entity_ratings[other_entity_id].rating

        if sum_ratio == 0:
            return self._calculate_team_rating_by_rating_type_and_excluding_entity_id(
                team_id=team_id,
                entity_id=match_entity.entity_id,
                match=match,
                min_games_played=min_games_played,
                use_projected_participation_weight=False
            )

        return pre_match_team_rating / sum_ratio

    def _calculate_rating_change(self,
                                 match_entity: MatchPlayer,
                                 match: Match,
                                 rating_change_multiplier: float,
                                 ):

        if match_entity.match_player_performance.ratio is not None \
                and len(match_entity.match_player_performance.ratio) > 0:
            pre_match_team_rating = self._calculate_team_rating_by_ratio(
                match_entity=match_entity,
                match=match,
                min_games_played=0,
            )

        else:
            pre_match_team_rating = self._calculate_team_rating_by_rating_type_and_excluding_entity_id(
                entity_id=match_entity.entity_id,
                match=match,
                team_id=match_entity.team_id,
                min_games_played=0,
                use_projected_participation_weight=False
            )

        entity_rating = self.entity_ratings[match_entity.entity_id]

        predicted_performance = self.performance_predictor.predict_performance(
            rating=match_entity.match_player_performance.rating.pre_match_entity_rating,
            opponent_rating=match_entity.match_player_performance.rating.pre_match_opponent_rating,
            team_rating=pre_match_team_rating
        )
        performance_difference = match_entity.match_player_performance.match_performance - predicted_performance

        rating_change_multiplier = self._calculate_rating_change_multiplier(
            entity_rating=entity_rating,
            rating_change_multiplier=rating_change_multiplier
        )

        rating_change = performance_difference * rating_change_multiplier * match_entity.match_player_performance.participation_weight
        if math.isnan(rating_change):
            logging.debug(f"rating change is nan return 0 entity id {match_entity.entity_id}")
            return 0

        st_idx = max(0, len(entity_rating.prev_rating_changes) - self.rating_change_momentum_games_count)
        prev_rating_changes = entity_rating.prev_rating_changes[st_idx:]

        rating_change += sum(prev_rating_changes) * self.rating_change_momentum_multiplier

        return rating_change

    def update_entity_ratings(self, match_entity, rating_change: float,
                              match: Match):

        self.entity_ratings[match_entity.entity_id].rating += rating_change
        self.entity_ratings[match_entity.entity_id].prev_rating_changes.append(rating_change)
        match_entity.match_player_performance.rating.post_match_entity_rating = \
            self.entity_ratings[match_entity.entity_id].rating

        self.entity_ratings[match_entity.entity_id].certain_sum = self._calculate_post_match_certain_sum(
            entity_rating=self.entity_ratings[match_entity.entity_id],
            match=match,
            particpation_weight=match_entity.match_player_performance.participation_weight
        )

        self.entity_ratings[match_entity.entity_id].last_match_day_number = match.day_number
        self.entity_ratings[match_entity.entity_id].games_played += 1


class MatchRatingCalculator(MatchRatingCalculatorMixin):

    def generate_pre_match_ratings(self,
                                   match: Match,
                                   calculate_participation_weight: bool
                                   ) -> Match:

        # team_entity_ratings, entity_id_to_rating = self._generate_pre_match_ratings(match=match)

        pre_start_rating_team_ratings, new_entity_ids = self._generate_team_ratings_before_start_rating(
            match)
        self._calculate_and_update_start_ratings(team_ratings=pre_start_rating_team_ratings, entity_ids=new_entity_ids,
                                                 match=match)
        #  team_ratings_proj, team_ratings = self._generate_team_ratings(match=match)

        team_id_to_opponent_team_id = self.generate_team_id_to_opponent_team_id(match)
        team_ratings_proj = {}
        team_ratings = {}
        for team_id in team_id_to_opponent_team_id:
            team_ratings[team_id] = self._calculate_team_rating_by_rating_type_and_excluding_entity_id(
                team_id=team_id,
                entity_id="ghrfrt",
                match=match,
                min_games_played=0,
                use_projected_participation_weight=False
            )
            if calculate_participation_weight:
                team_ratings_proj[
                    team_id] = self._calculate_team_rating_by_rating_type_and_excluding_entity_id_using_projected_participation_weight(
                    team_id=team_id,
                    entity_id="ghrfrt",
                    match=match,
                    min_games_played=0,
                )

        team_id_to_rating_change: Dict[str, float] = {}

        for match_entity_index, match_entity in enumerate(match.entities):
            team_id = match_entity.team_id
            team_id_opponent = team_id_to_opponent_team_id[team_id]
            if team_id not in team_id_to_rating_change:
                team_id_to_rating_change[team_id] = 0

            match.entities[match_entity_index].match_player_performance.rating = MatchRating(
                pre_match_entity_rating=self.entity_ratings[match_entity.entity_id].rating,
                pre_match_opponent_rating=team_ratings[team_id_opponent],
                pre_match_team_rating=team_ratings[team_id],
                rating_type=RatingType.DEFAULT,
            )
            if team_id in team_ratings_proj:
                match.entities[match_entity_index].match_player_performance[
                    RatingType.DEFAULT].rating.pre_match_projected_team_rating = team_ratings_proj[team_id]
                match.entities[match_entity_index].match_player_performance[
                    RatingType.DEFAULT].rating.pre_match_projected_opponent_rating = team_ratings_proj[
                    team_id_opponent]

            match_entity.match_player_performance[RatingType.DEFAULT].rating.pre_match_opponent_rating = \
                team_ratings[team_id_opponent]

            self.rating_type_to_ratings[RatingType.DEFAULT].append(
                self.entity_ratings[match_entity.entity_id].ratings[RatingType.DEFAULT])
            if self.past_team_ratings_for_average_rating is not None:
                start_index = max(0, len(
                    self.rating_type_to_ratings[RatingType.DEFAULT]) - self.past_team_ratings_for_average_rating)
                self.rating_type_to_ratings[RatingType.DEFAULT] = self.rating_type_to_ratings[RatingType.DEFAULT][
                                                                  start_index:]

        return match

    def update_entity_ratings_by_league_result(self, match: Match):
        self.update_entity_ratings_by_league_result_and_rating_type(match)

    def generate_team_id_to_opponent_team_id(self, match):
        team_ids = []
        for match_entity in match.entities:
            if match_entity.team_id not in team_ids:
                team_ids.append(match_entity.team_id)

        return {
            team_ids[0]: team_ids[1],
            team_ids[1]: team_ids[0]
        }

    def update_entity_ratings_for_matches(self, matches: List[Match]):

        entity_rating_changes = {}

        match_count = 0
        for match in matches:
            match_count += 1
            for match_entity_index, match_entity in enumerate(match.entities):

                if math.isnan(match_entity.match_player_performance.match_performance) is False:

                    if match_entity.entity_id not in entity_rating_changes:
                        entity_rating_changes[match_entity.entity_id] = 0

                    rating_change = self._calculate_rating_change(
                        match_entity=match_entity,
                        match=match,
                        rating_change_multiplier=self.rating_change_multiplier,
                    )

                    entity_rating_changes[match_entity.entity_id] += rating_change

                    if len(matches) == match_count:
                        self.update_entity_ratings(match_entity=match_entity,
                                                   rating_change=entity_rating_changes[
                                                       match_entity.entity_id],
                                                   match=match
                                                   )
                    match.entities[match_entity_index] = match_entity

        try:
            if len(self.rating_type_to_ratings[RatingType.DEFAULT]) > 0:
                self.average_rating = sum(self.rating_type_to_ratings[RatingType.DEFAULT]) / \
                                      len(self.rating_type_to_ratings[RatingType.DEFAULT])
        except TypeError:
            # temp band aid fix for projected games
            pass

    def _generate_team_ratings_before_start_rating(self, match: Match):
        team_rating: Dict[str, float] = {}
        new_entity_ids: List[str] = []
        match_team_count = {}
        for match_entity_index, match_entity in enumerate(match.entities):
            team_id = match_entity.team_id
            if team_id not in team_rating:
                team_rating[team_id] = 0
                match_team_count[team_id] = 0

            if match_entity.entity_id not in self.entity_ratings:
                new_entity_ids.append(match_entity.entity_id)
                continue

            match_team_count[team_id] += self.entity_ratings[match_entity.entity_id].games_played

            entity_rating = self.entity_ratings[match_entity.entity_id].rating

            team_rating[team_id] += entity_rating

        for team_id, count in match_team_count.items():
            if count < self.start_rating_calculator.min_match_ratings_for_team:
                team_rating[team_id] = None

        return team_rating, new_entity_ids

    def _calculate_and_update_start_ratings(self, team_ratings: Dict[str, float], entity_ids: List[str], match: Match):
        for match_entity in match.entities:
            if match_entity.entity_id not in entity_ids:
                continue
            team_rating = team_ratings[match_entity.team_id]
            start_rating = self.start_rating_calculator.update_rating(
                day_number=match.day_number,
                match_entity=match_entity,
                team_rating=team_rating,
            )

            self.entity_ratings[match_entity.entity_id] = PlayerRating(
                id=match_entity.entity_id,
                rating=start_rating,
                prev_rating_changes=[],
            )

    def _generate_team_ratings(self, match: Match):
        team_ratings_proj = {}
        team_ratings = {}
        for match_entity in match.entities:
            if match_entity.team_id not in team_ratings_proj:
                team_ratings[match_entity.team_id] = 0
                team_ratings_proj[match_entity.team_id] = 0

            team_ratings_proj[match_entity.team_id] += self.entity_ratings[match_entity.entity_id].ratings[
                                                           RatingType.DEFAULT] * match_entity.match_player_performance[
                                                           RatingType.DEFAULT].projected_participation_weight

            team_ratings[match_entity.team_id] += self.entity_ratings[match_entity.entity_id].ratings[
                                                      RatingType.DEFAULT] * match_entity.match_player_performance[
                                                      RatingType.DEFAULT].participation_weight

        return team_ratings_proj, team_ratings

    def update_league_ratings(self, day_number: int, match_entity: MatchPlayer):
        self.start_rating_calculator.update_league_ratings(day_number=day_number, match_entity=match_entity)


class MatchGenerator():

    def __init__(self,

                 league_identifier: LeagueIdentifier,
                 match_rating_calculator: DefaultMatchRatingCalculator,
                 ):
        self.league_identifier = league_identifier
        self.match_rating_calculator = match_rating_calculator

    def generate(self, match: Match, calculate_participation_weight: bool):
        try:
            self._validate_match(match)
        except ValueError:
            raise

        self._update_entity_leagues(match)

        match = self.match_rating_calculator.generate_pre_match_ratings(
            match=match,
            calculate_participation_weight=calculate_participation_weight
        )
        self._set_match_league(match)
        return match

    def update_ratings_for_matches(self, matches: List[Match]):
        self.match_rating_calculator.update_entity_ratings_for_matches(matches)

        for match in matches:
            if match.league is not None:
                self._update_entity_leagues(match)

            for match_entity in match.entities:
                self.match_rating_calculator.update_league_ratings(match.day_number, match_entity)

            if match.league is not None:
                self.match_rating_calculator.update_entity_ratings_by_league_result(match=match)

    def _validate_match(self, match: Match):
        if len(match.team_ids) < 2:
            raise ValueError

    def _set_match_league(self, match: Match) -> None:
        region_counts: Dict[str, int] = {}
        max_count: int = 0
        max_region: str = ""
        for entity in match.entities:
            region = entity.league

            if region not in region_counts:
                region_counts[region] = 0
            region_counts[region] += 1

            if region_counts[region] > max_count:
                max_count = region_counts[region]
                max_region = region

        match.league = max_region

    def _update_entity_leagues(self, match: Match) -> Match:

        for index, match_entity in enumerate(match.entities):
            entity_league = self.league_identifier.update_and_return_entity_league(match_entity.entity_id,
                                                                                   match.league)
            match.entities[index].league = entity_league

        match = self._update_opponent_leagues(match)
        return match

    def _update_opponent_leagues(self, match: Match) -> Match:
        team_league_counts = self._generate_team_league_counts(match)
        team_ids = [t for t in team_league_counts]
        team_leagues = self._generate_teams_to_leagues(team_ids, team_league_counts)
        for entity_index, entity in enumerate(match.entities):
            try:
                opponent_league = self._get_opponent_league(entity.team_id, team_leagues)
            except KeyError:
                opponent_league = None
            match.entities[entity_index].opponent_league = opponent_league

        return match

    def _generate_teams_to_leagues(self, team_ids: List[str], team_league_counts: Dict[str, Dict[str, int]]) -> Dict[
        str, str]:
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

    def _get_opponent_league(self, team_id: str, team_leagues: Dict[
        str, str]) -> str:

        for team_id2, league in team_leagues.items():
            if team_id2 != team_id:
                return team_leagues[team_id2]

        raise KeyError

    def _generate_team_league_counts(self, match: Match) -> Dict[str, Dict[str, int]]:

        team_league_counts: Dict[str, Dict[str, int]] = {}
        for entity in match.entities:
            if entity.team_id not in team_league_counts:
                team_league_counts[entity.team_id] = {}

            if entity.league not in team_league_counts[entity.team_id]:
                team_league_counts[entity.team_id][entity.league] = 0

            team_league_counts[entity.team_id][entity.league] += 1

        return team_league_counts
