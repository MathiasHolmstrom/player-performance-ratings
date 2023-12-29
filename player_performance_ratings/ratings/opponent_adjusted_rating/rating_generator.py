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

    @abstractmethod
    def generate(self, matches: Optional[list[Match]] = None, df: Optional[pd.DataFrame] = None,
                 column_names: Optional[ColumnNames] = None) -> dict[RatingColumnNames, list[float]]:
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
    def features_created(self) -> list[str]:
        pass


class OpponentAdjustedRatingGenerator(RatingGenerator):
    """
    Generates ratings for players and teams based on the match-performance of the player and the ratings of the players and teams.
    Ratings are updated after a match is finished
    """

    def __init__(self,
                 team_rating_generator: TeamRatingGenerator = TeamRatingGenerator(),
                 features_created: Optional[list[str]] = None,
                 ):

        """

        :param team_rating_generator:
            The class contains the logic for generating and updating team ratings and contains many parameters that can be tuned.
        :param features_names_created:
            If called by match_predictor, feature_names_created determines which features will be used for prediction.
            If other features such as player_rating_difference is used, it must be added to this list.
        """
        self.team_rating_generator = team_rating_generator
        self._features_created = features_created or [RatingColumnNames.RATING_DIFFERENCE]
        self.ratings_df = None

    def generate(self, matches: Optional[list[Match]] = None, df: Optional[pd.DataFrame] = None,
                 column_names: Optional[ColumnNames] = None) -> dict[RatingColumnNames, list[float]]:

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

        if matches is None and df is None or matches is None and column_names is None:
            raise ValueError("If matches is not passed, df and column names must be massed")

        if matches is None:
            matches = convert_df_to_matches(df=df, column_names=column_names)

        if df is not None and column_names:
            logging.info(
                "both df and column names are passed, and match-ratings will therefore be stored in the class object")

        elif column_names and df is None:
            logging.warning(
                "Column names is passed but df not - this match-ratings will not be stored in the class object")

        pre_match_player_rating_values = []
        pre_match_team_rating_values = []
        pre_match_opponent_rating_values = []
        team_opponent_leagues = []
        match_ids = []
        player_rating_changes = []
        player_leagues = []
        player_predicted_performances = []
        player_rating_differences = []
        performances = []

        team_rating_changes = []

        is_past = False if len(matches) > 0 and matches[0].teams[0].players[0].performance.performance_value is None else True
        logging.info("Creating ratings for past matches" if is_past else "Creating ratings for future matches")

        for match_idx, match in enumerate(matches):
            self._validate_match(match)
            pre_match_rating = PreMatchRating(
                id=match.id,
                teams=self._get_pre_match_team_ratings(match=match),
                day_number=match.day_number
            )

            if not is_past:
                for team_idx, pre_match_team in enumerate(pre_match_rating.teams):
                    opponent_team = pre_match_rating.teams[-team_idx + 1]
                    for player_idx, pre_match_player in enumerate(pre_match_team.players):
                        pre_match_player_rating_values.append(pre_match_player.rating_value)
                        pre_match_team_rating_values.append(pre_match_team.rating_value)
                        pre_match_opponent_rating_values.append(opponent_team.rating_value)
                        player_leagues.append(pre_match_player.league)
                        team_opponent_leagues.append(opponent_team.league)
                        player_rating_differences.append(
                            pre_match_player.rating_value - opponent_team.rating_value)
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
                        pre_match_player_rating_values.append(player_rating_change.pre_match_rating_value)
                        pre_match_team_rating_values.append(player_rating_change.pre_match_rating_value)
                        pre_match_opponent_rating_values.append(opponent_team.pre_match_rating_value)
                        player_leagues.append(player_rating_change.league)
                        team_opponent_leagues.append(opponent_team.league)
                        match_ids.append(match.id)

                        performances.append(player_rating_change.performance)
                        player_rating_differences.append(
                            player_rating_change.pre_match_rating_value - opponent_team.pre_match_rating_value)
                        player_predicted_performances.append(player_rating_change.predicted_performance)
                        player_rating_changes.append(player_rating_change.rating_change_value)

        rating_differences = np.array(pre_match_team_rating_values) - (
            pre_match_opponent_rating_values)
        rating_means = np.array(pre_match_team_rating_values) * 0.5 + 0.5 * np.array(
            pre_match_opponent_rating_values)

        if df is not None and column_names and is_past:
            self.ratings_df = df[
                [column_names.team_id, column_names.player_id, column_names.match_id]].assign(
                **{
                    RatingColumnNames.RATING_DIFFERENCE: rating_differences,
                    RatingColumnNames.PLAYER_LEAGUE: player_leagues,
                    RatingColumnNames.OPPONENT_LEAGUE: team_opponent_leagues,
                    RatingColumnNames.PLAYER_RATING: pre_match_player_rating_values,
                    RatingColumnNames.PLAYER_RATING_CHANGE: player_rating_changes,
                    RatingColumnNames.TEAM_RATING: pre_match_team_rating_values,
                    RatingColumnNames.OPPONENT_RATING: pre_match_opponent_rating_values,
                    RatingColumnNames.PERFORMANCE: performances,
                    RatingColumnNames.RATING_MEAN: rating_means,
                    RatingColumnNames.PLAYER_PREDICTED_PERFORMANCE: player_predicted_performances
                })

        return {
            RatingColumnNames.PLAYER_RATING_DIFFERENCE: player_rating_differences,
            RatingColumnNames.RATING_DIFFERENCE: rating_differences,
            RatingColumnNames.PLAYER_LEAGUE: player_leagues,
            RatingColumnNames.OPPONENT_LEAGUE: team_opponent_leagues,
            RatingColumnNames.PLAYER_RATING: pre_match_player_rating_values,
            RatingColumnNames.PLAYER_RATING_CHANGE: player_rating_changes,
            RatingColumnNames.MATCH_ID: match_ids,
            RatingColumnNames.TEAM_RATING: pre_match_team_rating_values,
            RatingColumnNames.OPPONENT_RATING: pre_match_opponent_rating_values,
            RatingColumnNames.RATING_MEAN: rating_means,
            RatingColumnNames.PLAYER_PREDICTED_PERFORMANCE: player_predicted_performances
        }

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
    def features_created(self) -> list[str]:
        return self._features_created
