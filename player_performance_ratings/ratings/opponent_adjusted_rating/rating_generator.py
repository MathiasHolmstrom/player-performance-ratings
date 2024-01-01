import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from player_performance_ratings.ratings import convert_df_to_matches
from player_performance_ratings.ratings.opponent_adjusted_rating.team_rating_generator import TeamRatingGenerator
from player_performance_ratings.ratings.enums import RatingColumnNames

from player_performance_ratings.data_structures import Match, PreMatchRating, PreMatchTeamRating, PlayerRating, \
    TeamRating, ColumnNames, TeamRatingChange


class RatingGenerator(ABC):

    def __init__(self, column_names: ColumnNames):
        self.column_names = column_names

    @abstractmethod
    def generate(self, matches: Optional[list[Match]] = None, df: Optional[pd.DataFrame] = None) -> dict[
        RatingColumnNames, list[float]]:
        pass

    @property
    @abstractmethod
    def player_ratings(self) -> dict[str, PlayerRating]:
        pass

    @property
    @abstractmethod
    def team_ratings(self) -> list[TeamRating]:
        pass

    @property
    @abstractmethod
    def features_out(self) -> list[str]:
        pass


class OpponentAdjustedRatingGenerator(RatingGenerator):
    """
    Generates ratings for players and teams based on the match-performance of the player and the ratings of the players and teams.
    Ratings are updated after a match is finished
    """

    def __init__(self,
                 column_names: ColumnNames,
                 team_rating_generator: TeamRatingGenerator = TeamRatingGenerator(),
                 features_out: Optional[list[str]] = None,
                 ):

        """

        :param team_rating_generator:
            The class contains the logic for generating and updating team ratings and contains many parameters that can be tuned.
        :param features_names_created:
            If called by match_predictor, feature_names_created determines which features will be used for prediction.
            If other features such as player_rating_difference is used, it must be added to this list.
        """
        super().__init__(column_names=column_names)
        self.team_rating_generator = team_rating_generator
        if column_names.projected_participation_weight is not None and column_names.projected_participation_weight != column_names.participation_weight:
            self._features_out = [RatingColumnNames.RATING_DIFFERENCE_PROJECTED]
        self._features_out = features_out or [RatingColumnNames.RATING_DIFFERENCE]

        # If projected participation weight is not None, then the projected ratings will be used instead of the actual ratings (which first are known after game is finished)


        self.ratings_df = None

    def generate(self, matches: Optional[list[Match]] = None, df: Optional[pd.DataFrame] = None) -> dict[
        RatingColumnNames, list[float]]:

        """
        Generate ratings by iterating over each match, calculate predicted performance and update ratings after the match is finished.
        Default settin

        :param matches: list of matches. Each match must contain two teams.
        :param df: The dataframe from which the matches were generated. Only needed if you want to store the ratings in the class object in which case the column names must also be passed.
        :param column_names: The column names of the dataframe. Only needed if you want to store the ratings in the class object in which case the df must also be passed.
        :return: A dictionary containing historical match-rating values.
         These ratings can easily be added as new columns to the original dataframe for later model training or exploration
        """

        if matches is not None and len(matches) > 0 and not isinstance(matches[0], Match):
            raise ValueError("matches must be a list of Match objects")

        if matches is None and df is None:
            raise ValueError("If matches is not passed, df and column names must be massed")

        if matches is None:
            matches = convert_df_to_matches(df=df, column_names=self.column_names)

        if df is not None and self.column_names:
            logging.info(
                "both df and column names are passed, and match-ratings will therefore be stored in the class object")

        elif self.column_names and df is None:
            logging.warning(
                "Column names is passed but df not - this match-ratings will not be stored in the class object")

        pre_match_player_rating_values = []
        pre_match_team_rating_values = []
        pre_match_opponent_projected_rating_values = []
        pre_match_opponent_rating_values = []
        team_opponent_leagues = []
        match_ids = []
        player_rating_changes = []
        player_leagues = []
        player_predicted_performances = []
        pre_match_team_projected_rating_values = []
        performances = []

        team_rating_changes = []

        is_future = True if len(matches) > 0 and matches[0].teams[0].players[
            0].performance.performance_value is None else False
        logging.info("Creating ratings for past matches" if is_future else "Creating ratings for future matches")

        for match_idx, match in enumerate(matches):
            self._validate_match(match)
            pre_match_rating = PreMatchRating(
                id=match.id,
                teams=self._get_pre_match_team_ratings(match=match),
                day_number=match.day_number
            )

            if is_future:
                for team_idx, pre_match_team in enumerate(pre_match_rating.teams):
                    opponent_team = pre_match_rating.teams[-team_idx + 1]
                    for player_idx, pre_match_player in enumerate(pre_match_team.players):
                        pre_match_team_projected_rating_values.append(pre_match_team.projected_rating_value)
                        pre_match_player_rating_values.append(pre_match_player.rating_value)
                        pre_match_opponent_projected_rating_values.append(opponent_team.projected_rating_value)
                        team_opponent_leagues.append(opponent_team.league)
                        player_leagues.append(pre_match_player.league)

                        match_ids.append(match.id)


            else:
                match_team_rating_changes = self._create_match_team_rating_changes(match=match,
                                                                                   pre_match_rating=pre_match_rating)
                team_rating_changes += match_team_rating_changes

                if match_idx == len(matches) - 1 or matches[match_idx + 1].update_id != match.update_id:
                    self._update_ratings(team_rating_changes=team_rating_changes)
                    team_rating_changes = []

                for team_idx, team_rating_change in enumerate(match_team_rating_changes):
                    opponent_team = match_team_rating_changes[-team_idx + 1]
                    for player_idx, player_rating_change in enumerate(team_rating_change.players):
                        pre_match_team_projected_rating_values.append(
                            pre_match_rating.teams[team_idx].projected_rating_value)
                        pre_match_opponent_projected_rating_values.append(
                            pre_match_rating.teams[-team_idx + 1].projected_rating_value)

                        pre_match_player_rating_values.append(player_rating_change.pre_match_rating_value)
                        pre_match_team_rating_values.append(pre_match_rating.teams[team_idx].rating_value)
                        pre_match_opponent_rating_values.append(pre_match_rating.teams[-team_idx+1].rating_value)
                        player_leagues.append(player_rating_change.league)
                        team_opponent_leagues.append(opponent_team.league)
                        match_ids.append(match.id)

                        performances.append(player_rating_change.performance)
                        player_predicted_performances.append(player_rating_change.predicted_performance)
                        player_rating_changes.append(player_rating_change.rating_change_value)

        rating_differences_projected = np.array(pre_match_team_projected_rating_values) - (
            pre_match_opponent_projected_rating_values)
        player_rating_differences_projected = np.array(pre_match_team_projected_rating_values) - (
            pre_match_opponent_projected_rating_values)
        player_rating_difference_from_team_projected = np.array(pre_match_player_rating_values) - np.array(
            pre_match_team_projected_rating_values)
        rating_means_projected = np.array(pre_match_team_projected_rating_values) * 0.5 + 0.5 * np.array(
            pre_match_opponent_projected_rating_values)

        feature_values = {
            RatingColumnNames.RATING_DIFFERENCE_PROJECTED: rating_differences_projected,
            RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED: player_rating_difference_from_team_projected,
            RatingColumnNames.PLAYER_RATING_DIFFERENCE_PROJECTED: player_rating_differences_projected,
            RatingColumnNames.TEAM_RATING_PROJECTED: pre_match_team_projected_rating_values,
            RatingColumnNames.OPPONENT_RATING_PROJECTED: pre_match_opponent_projected_rating_values,
            RatingColumnNames.PLAYER_LEAGUE: player_leagues,
            RatingColumnNames.OPPONENT_LEAGUE: team_opponent_leagues,
            RatingColumnNames.TEAM_RATING: pre_match_team_rating_values,
            RatingColumnNames.RATING_MEAN_PROJECTED: rating_means_projected,
            RatingColumnNames.MATCH_ID: match_ids,
            RatingColumnNames.PLAYER_RATING: pre_match_player_rating_values,
        }

        if self.column_names.projected_participation_weight is None or self.column_names.projected_participation_weight == self.column_names.participation_weight:
            if RatingColumnNames.RATING_DIFFERENCE_PROJECTED not in self._features_out:
                feature_values[RatingColumnNames.RATING_DIFFERENCE] = feature_values.pop(
                    RatingColumnNames.RATING_DIFFERENCE_PROJECTED)

            if RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED not in self._features_out:
                feature_values[RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM] = feature_values.pop(
                    RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED)

            if RatingColumnNames.PLAYER_RATING_DIFFERENCE_PROJECTED not in self._features_out:
                feature_values[RatingColumnNames.PLAYER_RATING_DIFFERENCE] = feature_values.pop(
                    RatingColumnNames.PLAYER_RATING_DIFFERENCE_PROJECTED)

            if RatingColumnNames.TEAM_RATING_PROJECTED not in self._features_out:
                feature_values[RatingColumnNames.TEAM_RATING] = feature_values.pop(
                    RatingColumnNames.TEAM_RATING_PROJECTED)

            if RatingColumnNames.OPPONENT_RATING_PROJECTED not in self._features_out:
                feature_values[RatingColumnNames.OPPONENT_RATING] = feature_values.pop(
                    RatingColumnNames.OPPONENT_RATING_PROJECTED)

            if RatingColumnNames.RATING_MEAN_PROJECTED not in self._features_out:
                feature_values[RatingColumnNames.RATING_MEAN] = feature_values.pop(RatingColumnNames.RATING_MEAN_PROJECTED)

        if not is_future:
            if RatingColumnNames.RATING_DIFFERENCE not in feature_values:
                feature_values[RatingColumnNames.PLAYER_RATING_DIFFERENCE] = np.array(
                    pre_match_player_rating_values) - np.array(
                    pre_match_opponent_rating_values)
                feature_values[RatingColumnNames.RATING_DIFFERENCE] = np.array(pre_match_team_rating_values) - np.array(
                    pre_match_opponent_rating_values)
                feature_values[RatingColumnNames.PLAYER_RATING] = pre_match_player_rating_values
                feature_values[RatingColumnNames.OPPONENT_RATING] = pre_match_opponent_rating_values
                feature_values[RatingColumnNames.TEAM_RATING] = pre_match_team_rating_values
                feature_values[RatingColumnNames.RATING_MEAN] = np.array(
                    pre_match_team_rating_values) * 0.5 + 0.5 * np.array(pre_match_opponent_rating_values)

                feature_values[RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM] = np.array(
                    pre_match_player_rating_values) - np.array(pre_match_team_rating_values)
                feature_values[RatingColumnNames.PERFORMANCE] = performances

            feature_values[RatingColumnNames.PLAYER_RATING_CHANGE] = player_rating_changes
            feature_values[RatingColumnNames.PLAYER_PREDICTED_PERFORMANCE] = player_predicted_performances

        if df is not None and self.column_names and is_future:
            self.ratings_df = df[
                [self.column_names.team_id, self.column_names.player_id, self.column_names.match_id]].assign(
                **feature_values)

        return {f: feature_values[f] for f in self._features_out}

    def _create_match_team_rating_changes(self, match: Match, pre_match_rating: PreMatchRating) -> list[
        TeamRatingChange]:

        team_rating_changes = []

        for team_idx, pre_match_team_rating in enumerate(pre_match_rating.teams):
            team_rating_change = self.team_rating_generator.generate_rating_change(day_number=match.day_number,
                                                                                   pre_match_team_rating=pre_match_team_rating,
                                                                                   pre_match_opponent_team_rating=
                                                                                   pre_match_rating.teams[
                                                                                       -team_idx + 1])
            team_rating_changes.append(team_rating_change)

        return team_rating_changes

    def _update_ratings(self, team_rating_changes: list[TeamRatingChange]):

        for team_rating_change in team_rating_changes:
            self.team_rating_generator.update_rating_by_team_rating_change(team_rating_change=team_rating_change)

    def _get_pre_match_team_ratings(self, match: Match) -> list[PreMatchTeamRating]:
        pre_match_team_ratings = []
        for match_team in match.teams:
            pre_match_team_ratings.append(self.team_rating_generator.generate_pre_match_team_rating(
                match_team=match_team, day_number=match.day_number))

        return pre_match_team_ratings

    def _validate_match(self, match: Match):
        if len(match.teams) < 2:
            print(f"{match.id} only contains {len(match.teams)} teams")
            raise ValueError

    @property
    def player_ratings(self) -> dict[str, PlayerRating]:
        return dict(sorted(self.team_rating_generator.player_ratings.items(),
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

    @property
    def features_out(self) -> list[str]:
        return self._features_out
