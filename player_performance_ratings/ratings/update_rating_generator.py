from typing import Optional, Any, Union

import numpy as np
import pandas as pd

from player_performance_ratings.ratings import convert_df_to_matches
from player_performance_ratings.ratings.rating_calculators import RatingMeanPerformancePredictor
from player_performance_ratings.ratings.rating_calculators.performance_predictor import \
    RatingNonOpponentPerformancePredictor
from player_performance_ratings.ratings.rating_calculators.match_rating_generator import MatchRatingGenerator
from player_performance_ratings.ratings.enums import RatingEstimatorFeatures, RatingHistoricalFeatures

from player_performance_ratings.data_structures import Match, PreMatchRating, PreMatchTeamRating, PlayerRating, \
    TeamRating, ColumnNames, TeamRatingChange
from player_performance_ratings.ratings.rating_generator import RatingGenerator


class UpdateRatingGenerator(RatingGenerator):
    """
    Generates ratings for players and teams based on the match-performance of the player and the ratings of the players and teams.
    Ratings are updated after a match is finished
    """

    def __init__(self,
                 column_names: ColumnNames,
                 match_rating_generator: Optional[MatchRatingGenerator] = None,
                 estimator_features_out: Optional[list[RatingEstimatorFeatures]] = None,
                 historical_features_out: Optional[list[RatingHistoricalFeatures]] = None,
                 estimator_features_pass_through: Optional[list[RatingEstimatorFeatures]] = None,
                 distinct_positions: Optional[list[str]] = None,
                 ):

        """

        :param match_rating_generator:
            The class contains the logic for generating and updating team ratings and contains many parameters that can be tuned.
        :param features_names_created:
            If called by match_predictor, feature_names_created determines which features will be used for prediction.
            If other features such as player_rating_difference is used, it must be added to this list.
        """
        super().__init__(column_names=column_names)
        self.match_rating_generator = match_rating_generator or MatchRatingGenerator()
        self.distinct_positions = distinct_positions
        self._estimator_features_pass_through = estimator_features_pass_through or []

        self._estimator_features_out = estimator_features_out if estimator_features_out is not None else [
            RatingEstimatorFeatures.RATING_MEAN_PROJECTED] if isinstance(self.match_rating_generator.performance_predictor,
                                                                         RatingMeanPerformancePredictor) else [
            RatingEstimatorFeatures.PLAYER_RATING] if isinstance(self.match_rating_generator.performance_predictor,
                                                                 RatingNonOpponentPerformancePredictor) else [
            RatingEstimatorFeatures.RATING_DIFFERENCE_PROJECTED]

        if self.distinct_positions:
            self._estimator_features_out += [RatingEstimatorFeatures.RATING_DIFFERENCE_POSITION + "_" + p for p in
                                             self.distinct_positions]

        self._historical_features_out = historical_features_out or []


        # If projected participation weight is not None, then the projected ratings will be used instead of the actual ratings (which first are known after game is finished)

        self.ratings_df = None

    def generate_historical(self, df: Optional[pd.DataFrame] = None, matches: Optional[list[Match]] = None) -> dict[Union[
                                                                                                                        RatingEstimatorFeatures, RatingHistoricalFeatures], list[float]]:

        """
        Generate ratings by iterating over each match, calculate predicted performance and update ratings after the match is finished.
        Default settin

        :param matches: list of matches. Each match must contain two teams.
        :param df: The dataframe from which the matches were generated. Only needed if you want to store the ratings in the class object in which case the column names must also be passed.
        :param column_names: The column names of the dataframe. Only needed if you want to store the ratings in the class object in which case the df must also be passed.
        :return: A dictionary containing historical match-rating values.
         These ratings can easily be added as new columns to the original dataframe for later model training or exploration
        """

        if self.column_names.participation_weight is not None and df is not None and self.column_names.participation_weight not in df.columns:
            raise ValueError(f"participation_weight {self.column_names.participation_weight} not in df columns")

        if matches is not None and len(matches) > 0 and not isinstance(matches[0], Match):
            raise ValueError("matches must be a list of Match objects")

        if matches is None and df is None:
            raise ValueError("If matches is not passed, df must be massed")

        if matches is None:
            matches = convert_df_to_matches(df=df, column_names=self.column_names)

        pre_match_player_rating_values = []
        pre_match_team_rating_values = []
        pre_match_opponent_projected_rating_values = []
        pre_match_opponent_rating_values = []
        team_opponent_leagues = []
        rating_update_match_ids = []
        rating_update_team_ids = []
        rating_update_team_ids_opponent = []
        player_rating_changes = []
        player_leagues = []
        player_predicted_performances = []
        projected_participation_weights = []
        pre_match_team_projected_rating_values = []
        position_rating_difference_values = {}
        performances = []
        player_ids = []
        team_leagues = []

        team_rating_changes = []

        for match_idx, match in enumerate(matches):
            self._validate_match(match)

            pre_match_rating = PreMatchRating(
                id=match.id,
                teams=self._get_pre_match_team_ratings(match=match),
                day_number=match.day_number
            )
            match_team_rating_changes = self._create_match_team_rating_changes(match=match,
                                                                               pre_match_rating=pre_match_rating)
            team_rating_changes += match_team_rating_changes

            if match_idx == len(matches) - 1 or matches[match_idx + 1].update_id != match.update_id:
                self._update_ratings(team_rating_changes=team_rating_changes)
                team_rating_changes = []

            match_position_ratings = []

            for team_idx, team_rating_change in enumerate(match_team_rating_changes):
                match_position_ratings.append({})
                opponent_team = match_team_rating_changes[-team_idx + 1]
                for player_idx, player_rating_change in enumerate(team_rating_change.players):

                    position = match.teams[team_idx].players[player_idx].position
                    if position:
                        match_position_ratings[team_idx][position] = player_rating_change.pre_match_rating_value

                    pre_match_team_projected_rating_values.append(
                        pre_match_rating.teams[team_idx].projected_rating_value)
                    pre_match_opponent_projected_rating_values.append(
                        pre_match_rating.teams[-team_idx + 1].projected_rating_value)

                    pre_match_player_rating_values.append(player_rating_change.pre_match_rating_value)
                    pre_match_team_rating_values.append(pre_match_rating.teams[team_idx].rating_value)
                    pre_match_opponent_rating_values.append(pre_match_rating.teams[-team_idx + 1].rating_value)
                    player_leagues.append(player_rating_change.league)
                    team_opponent_leagues.append(opponent_team.league)
                    team_leagues.append(team_rating_change.league)
                    rating_update_match_ids.append(match.update_id)
                    rating_update_team_ids.append(match.teams[team_idx].update_id)
                    rating_update_team_ids_opponent.append(match.teams[-team_idx + 1].update_id)
                    projected_participation_weights.append(
                        match.teams[team_idx].players[player_idx].performance.projected_participation_weight)

                    performances.append(player_rating_change.performance)
                    player_predicted_performances.append(player_rating_change.predicted_performance)
                    player_rating_changes.append(player_rating_change.rating_change_value)
                    player_ids.append(player_rating_change.id)

            if self.distinct_positions:
                for team_idx in range(len(match_team_rating_changes)):
                    player_per_team_count = len(match_team_rating_changes[team_idx].players)

                    for position in self.distinct_positions:
                        if position not in position_rating_difference_values:
                            position_rating_difference_values[position] = []
                        if position in match_position_ratings[team_idx] and position in match_position_ratings[
                            -team_idx + 1]:
                            position_rating_difference_values[position] += [match_position_ratings[team_idx][position] -
                                                                            match_position_ratings[-team_idx + 1][
                                                                                position]] * player_per_team_count
                        else:
                            position_rating_difference_values[position] += [0] * player_per_team_count

        potential_feature_values = self._get_shared_rating_values(
            position_rating_difference_values=position_rating_difference_values,
            pre_match_team_projected_rating_values=pre_match_team_projected_rating_values,
            pre_match_opponent_projected_rating_values=pre_match_opponent_projected_rating_values,
            pre_match_player_rating_values=pre_match_player_rating_values,
            player_leagues=player_leagues,
            team_opponent_leagues=team_opponent_leagues,
            projected_participation_weights=projected_participation_weights,
            match_ids=rating_update_match_ids,
            team_ids=rating_update_team_ids,
            team_id_opponents=rating_update_team_ids_opponent,
            player_ids=player_ids,
            team_leagues=team_leagues,
        )
        potential_feature_values[RatingHistoricalFeatures.PLAYER_RATING_DIFFERENCE] = np.array(
            pre_match_player_rating_values) - np.array(
            pre_match_opponent_rating_values)
        potential_feature_values[RatingHistoricalFeatures.RATING_DIFFERENCE] = np.array(
            pre_match_team_rating_values) - np.array(
            pre_match_opponent_rating_values)
        potential_feature_values[RatingEstimatorFeatures.PLAYER_RATING] = pre_match_player_rating_values
        potential_feature_values[RatingHistoricalFeatures.OPPONENT_RATING] = pre_match_opponent_rating_values
        potential_feature_values[RatingHistoricalFeatures.TEAM_RATING] = pre_match_team_rating_values
        potential_feature_values[RatingHistoricalFeatures.RATING_MEAN] = np.array(
            pre_match_team_rating_values) * 0.5 + 0.5 * np.array(pre_match_opponent_rating_values)

        potential_feature_values[RatingHistoricalFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM] = np.array(
            pre_match_player_rating_values) - np.array(pre_match_team_rating_values)
        potential_feature_values[RatingHistoricalFeatures.PERFORMANCE] = performances

        potential_feature_values[RatingHistoricalFeatures.PLAYER_RATING_CHANGE] = player_rating_changes
        potential_feature_values[
            RatingHistoricalFeatures.PLAYER_PREDICTED_PERFORMANCE] = player_predicted_performances

        if df is not None and self.column_names:
            self.ratings_df = df[
                [self.column_names.team_id, self.column_names.player_id, self.column_names.match_id]].assign(
                **potential_feature_values)

        return {f: potential_feature_values[f] for f in self.estimator_features_return + self._historical_features_out}

    def generate_future(self, df: Optional[pd.DataFrame] = None, matches: Optional[list[Match]] = None) -> dict[Union[
                                                                                                                    RatingEstimatorFeatures, RatingHistoricalFeatures], list[float]]:

        if matches is not None and len(matches) > 0 and not isinstance(matches[0], Match):
            raise ValueError("matches must be a list of Match objects")

        if matches is None and df is None:
            raise ValueError("If matches is not passed, df must be massed")

        if matches is None:
            matches = convert_df_to_matches(df=df, column_names=self.column_names)

        pre_match_player_rating_values = []
        pre_match_opponent_projected_rating_values = []
        team_opponent_leagues = []
        match_ids = []
        player_leagues = []
        projected_participation_weights = []
        rating_update_team_ids = []
        rating_update_team_ids_opponent = []
        player_ids = []
        team_leagues = []
        position_rating_difference_values = {}

        pre_match_team_projected_rating_values = []

        for match_idx, match in enumerate(matches):
            match_position_ratings = []
            self._validate_match(match)
            pre_match_rating = PreMatchRating(
                id=match.id,
                teams=self._get_pre_match_team_ratings(match=match),
                day_number=match.day_number
            )

            for team_idx, pre_match_team in enumerate(pre_match_rating.teams):
                match_position_ratings.append({})
                opponent_team = pre_match_rating.teams[-team_idx + 1]
                for player_idx, pre_match_player in enumerate(pre_match_team.players):
                    position = match.teams[team_idx].players[player_idx].position
                    if position:
                        match_position_ratings[team_idx][position] = pre_match_player.rating_value
                    pre_match_team_projected_rating_values.append(pre_match_team.projected_rating_value)
                    pre_match_player_rating_values.append(pre_match_player.rating_value)
                    pre_match_opponent_projected_rating_values.append(opponent_team.projected_rating_value)
                    team_opponent_leagues.append(opponent_team.league)
                    team_leagues.append(pre_match_team.league)
                    player_leagues.append(pre_match_player.league)
                    match_ids.append(match.id)
                    player_ids.append(pre_match_player.id)
                    rating_update_team_ids.append(match.teams[team_idx].update_id)
                    rating_update_team_ids_opponent.append(match.teams[-team_idx + 1].update_id)
                    projected_participation_weights.append(
                        match.teams[team_idx].players[player_idx].performance.projected_participation_weight)

            if self.distinct_positions:
                for team_idx in range(len(pre_match_rating.teams)):
                    player_per_team_count = len(pre_match_rating.teams[team_idx].players)

                    for position in self.distinct_positions:
                        if position not in position_rating_difference_values:
                            position_rating_difference_values[position] = []
                        if position in match_position_ratings[team_idx] and position in match_position_ratings[
                            -team_idx + 1]:
                            position_rating_difference_values[position] += [match_position_ratings[team_idx][position] -
                                                                            match_position_ratings[-team_idx + 1][
                                                                                position]] * player_per_team_count
                        else:
                            position_rating_difference_values[position] += [0] * player_per_team_count

        potential_feature_values = self._get_shared_rating_values(
            position_rating_difference_values=position_rating_difference_values,
            pre_match_team_projected_rating_values=pre_match_team_projected_rating_values,
            pre_match_opponent_projected_rating_values=pre_match_opponent_projected_rating_values,
            pre_match_player_rating_values=pre_match_player_rating_values,
            player_leagues=player_leagues,
            team_opponent_leagues=team_opponent_leagues,
            match_ids=match_ids,
            projected_participation_weights=projected_participation_weights,
            team_ids=rating_update_team_ids,
            team_id_opponents=rating_update_team_ids_opponent,
            player_ids=player_ids,
            team_leagues=team_leagues,
        )

        return {f: potential_feature_values[f] for f in self.estimator_features_return + self._historical_features_out}

    def _get_shared_rating_values(self,
                                  position_rating_difference_values: dict[str, list[float]],
                                  pre_match_team_projected_rating_values: list[float],
                                  pre_match_opponent_projected_rating_values: list[float],
                                  pre_match_player_rating_values: list[float],
                                  player_leagues: list[str],
                                  team_opponent_leagues: list[str],
                                  match_ids: list[str],
                                  team_ids: list[str],
                                  team_id_opponents: list[str],
                                  player_ids: list[str],
                                  projected_participation_weights: list[float],
                                  team_leagues: list[str],
                                  ) -> dict[Union[RatingEstimatorFeatures, RatingHistoricalFeatures], Any]:

        if self.column_names.projected_participation_weight:
            df = pd.DataFrame({
                "match_id": match_ids,
                "team_id": team_ids,
                "team_id_opponent": team_id_opponents,
                RatingEstimatorFeatures.PLAYER_RATING: pre_match_player_rating_values,
                "projected_participation_weight": projected_participation_weights,
                "player_id": player_ids,
            })

            game_player = df.groupby(["match_id", "player_id", "team_id", "team_id_opponent"])[
                [RatingEstimatorFeatures.PLAYER_RATING, "projected_participation_weight"]].mean().reset_index()

            game_player["game_team_sum_projected_participation_weight"] = game_player.groupby(["match_id", "team_id"])[
                "projected_participation_weight"].transform('sum')

            game_player['weighted_pre_match_player_rating_value'] = game_player[RatingEstimatorFeatures.PLAYER_RATING] * \
                                                                    game_player["projected_participation_weight"]

            game_player[RatingEstimatorFeatures.TEAM_RATING_PROJECTED] = game_player.groupby(["match_id", "team_id"])[
                                                                       "weighted_pre_match_player_rating_value"].transform(
                'sum') / game_player['game_team_sum_projected_participation_weight']

            game_team = game_player.groupby(["match_id", "team_id", "team_id_opponent"])[
                RatingEstimatorFeatures.TEAM_RATING_PROJECTED].mean().reset_index()

            game_team = game_team.merge(
                game_team[["match_id", "team_id_opponent", RatingEstimatorFeatures.TEAM_RATING_PROJECTED]].rename(
                    columns={RatingEstimatorFeatures.TEAM_RATING_PROJECTED: RatingEstimatorFeatures.OPPONENT_RATING_PROJECTED}),
                left_on=["match_id", "team_id"], right_on=["match_id", "team_id_opponent"])

            game_player = game_player.merge(
                game_team[['match_id', 'team_id', RatingEstimatorFeatures.OPPONENT_RATING_PROJECTED]],
                on=["match_id", "team_id"])

            game_player[RatingEstimatorFeatures.RATING_MEAN_PROJECTED] = (game_player[
                                                                        RatingEstimatorFeatures.TEAM_RATING_PROJECTED] +
                                                                          game_player[
                                                                        RatingEstimatorFeatures.OPPONENT_RATING_PROJECTED]) / 2

            df = df[["match_id", "player_id"]].merge(
                game_player[["match_id", "player_id",
                             RatingEstimatorFeatures.TEAM_RATING_PROJECTED,
                             RatingEstimatorFeatures.OPPONENT_RATING_PROJECTED,
                             RatingEstimatorFeatures.RATING_MEAN_PROJECTED,
                             RatingEstimatorFeatures.PLAYER_RATING
                             ]], on=["match_id", "player_id"], how='left')

            rating_differences_projected = (df[RatingEstimatorFeatures.TEAM_RATING_PROJECTED] - df[
                RatingEstimatorFeatures.OPPONENT_RATING_PROJECTED]).tolist()
            player_rating_difference_from_team_projected = (
                    df[RatingEstimatorFeatures.PLAYER_RATING] - df[RatingEstimatorFeatures.TEAM_RATING_PROJECTED]).tolist()
            player_rating_differences_projected = (
                    df[RatingEstimatorFeatures.PLAYER_RATING] - df[RatingEstimatorFeatures.OPPONENT_RATING_PROJECTED]).tolist()
            rating_means_projected = df[RatingEstimatorFeatures.RATING_MEAN_PROJECTED].tolist()
            pre_match_opponent_projected_rating_values = df[RatingEstimatorFeatures.OPPONENT_RATING_PROJECTED].tolist()
            pre_match_team_projected_rating_values = df[RatingEstimatorFeatures.TEAM_RATING_PROJECTED].tolist()
            pre_match_player_rating_values = df[RatingEstimatorFeatures.PLAYER_RATING].tolist()

        else:
            rating_differences_projected = (np.array(pre_match_team_projected_rating_values) - np.array(
                pre_match_opponent_projected_rating_values)).tolist()
            player_rating_differences_projected = (np.array(pre_match_player_rating_values) - np.array(
                pre_match_opponent_projected_rating_values)).tolist()
            player_rating_difference_from_team_projected = (np.array(pre_match_player_rating_values) - np.array(
                pre_match_team_projected_rating_values)).tolist()
            rating_means_projected = (np.array(pre_match_team_projected_rating_values) * 0.5 + 0.5 * np.array(
                pre_match_opponent_projected_rating_values)).tolist()

        return_values = {
            RatingEstimatorFeatures.RATING_DIFFERENCE_PROJECTED: rating_differences_projected,
            RatingEstimatorFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED: player_rating_difference_from_team_projected,
            RatingEstimatorFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED: player_rating_differences_projected,
            RatingEstimatorFeatures.TEAM_RATING_PROJECTED: pre_match_team_projected_rating_values,
            RatingEstimatorFeatures.OPPONENT_RATING_PROJECTED: pre_match_opponent_projected_rating_values,
            RatingEstimatorFeatures.PLAYER_RATING: pre_match_player_rating_values,
            RatingEstimatorFeatures.PLAYER_LEAGUE: player_leagues,
            RatingEstimatorFeatures.OPPONENT_LEAGUE: team_opponent_leagues,
            RatingEstimatorFeatures.TEAM_LEAGUE: team_leagues,
            RatingEstimatorFeatures.RATING_MEAN_PROJECTED: rating_means_projected,
            RatingEstimatorFeatures.MATCH_ID: match_ids,
        }

        if self.distinct_positions:
            for position, rating_values in position_rating_difference_values.items():
                return_values[RatingEstimatorFeatures.RATING_DIFFERENCE_POSITION + "_" + position] = rating_values

        return return_values

    def _create_match_team_rating_changes(self, match: Match, pre_match_rating: PreMatchRating) -> list[
        TeamRatingChange]:

        team_rating_changes = []

        for team_idx, pre_match_team_rating in enumerate(pre_match_rating.teams):
            team_rating_change = self.match_rating_generator.generate_rating_change(day_number=match.day_number,
                                                                                    pre_match_team_rating=pre_match_team_rating,
                                                                                    pre_match_opponent_team_rating=
                                                                                   pre_match_rating.teams[
                                                                                       -team_idx + 1])
            team_rating_changes.append(team_rating_change)

        return team_rating_changes

    def _update_ratings(self, team_rating_changes: list[TeamRatingChange]):

        for idx, team_rating_change in enumerate(team_rating_changes):
            self.match_rating_generator.update_rating_by_team_rating_change(team_rating_change=team_rating_change,
                                                                            opponent_team_rating_change=
                                                                           team_rating_changes[-idx + 1])

    def _get_pre_match_team_ratings(self, match: Match) -> list[PreMatchTeamRating]:
        pre_match_team_ratings = []
        for match_team in match.teams:
            pre_match_team_ratings.append(self.match_rating_generator.generate_pre_match_team_rating(
                match_team=match_team, day_number=match.day_number))

        return pre_match_team_ratings

    def _validate_match(self, match: Match):
        if len(match.teams) < 2:
            raise ValueError(f"{match.id} only contains {len(match.teams)} teams")

    @property
    def player_ratings(self) -> dict[str, PlayerRating]:
        return dict(sorted(self.match_rating_generator.player_ratings.items(),
                           key=lambda item: item[1].rating_value, reverse=True))

    @property
    def team_ratings(self) -> list[TeamRating]:
        team_id_ratings: list[TeamRating] = []
        teams = self.match_rating_generator.teams
        player_ratings = self.player_ratings
        for id, team in teams.items():
            team_player_ratings = [player_ratings[p] for p in team.player_ids]
            team_rating_value = sum([p.rating_value for p in team_player_ratings]) / len(team_player_ratings)
            team_id_ratings.append(TeamRating(id=team.id, name=team.name, players=team_player_ratings,
                                              last_match_day_number=team.last_match_day_number,
                                              rating_value=team_rating_value))

        return list(sorted(team_id_ratings,
                           key=lambda team: team.rating_value, reverse=True))

    @property
    def estimator_features_out(self) -> list[RatingEstimatorFeatures]:
        return self._estimator_features_out

    @property
    def features_out(self) -> list[Union[RatingEstimatorFeatures, RatingHistoricalFeatures]]:
        return self._estimator_features_out + self._historical_features_out

    @property
    def estimator_features_return(self) -> list[RatingEstimatorFeatures]:
        if self._estimator_features_pass_through:
            return list(set(self._estimator_features_pass_through + self.estimator_features_out))
        return self.estimator_features_out