from typing import Optional, Any

import numpy as np
import pandas as pd

from player_performance_ratings.ratings import convert_df_to_matches
from player_performance_ratings.ratings.opponent_adjusted_rating import RatingMeanPerformancePredictor
from player_performance_ratings.ratings.opponent_adjusted_rating.team_rating_generator import TeamRatingGenerator
from player_performance_ratings.ratings.enums import RatingColumnNames, HistoricalRatingColumnNames

from player_performance_ratings.data_structures import Match, PreMatchRating, PreMatchTeamRating, PlayerRating, \
    TeamRating, ColumnNames, TeamRatingChange
from player_performance_ratings.ratings.rating_generator import RatingGenerator


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

        self._features_out = features_out if features_out is not None else [
            RatingColumnNames.RATING_MEAN_PROJECTED] if isinstance(team_rating_generator.performance_predictor, RatingMeanPerformancePredictor) else [
            RatingColumnNames.RATING_DIFFERENCE_PROJECTED]

        # If projected participation weight is not None, then the projected ratings will be used instead of the actual ratings (which first are known after game is finished)

        self.ratings_df = None

    def generate_historical(self, matches: Optional[list[Match]] = None, df: Optional[pd.DataFrame] = None) -> dict[
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
        match_ids = []
        player_rating_changes = []
        player_leagues = []
        player_predicted_performances = []
        pre_match_team_projected_rating_values = []
        performances = []

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

            for team_idx, team_rating_change in enumerate(match_team_rating_changes):
                opponent_team = match_team_rating_changes[-team_idx + 1]
                for player_idx, player_rating_change in enumerate(team_rating_change.players):
                    pre_match_team_projected_rating_values.append(
                        pre_match_rating.teams[team_idx].projected_rating_value)
                    pre_match_opponent_projected_rating_values.append(
                        pre_match_rating.teams[-team_idx + 1].projected_rating_value)

                    pre_match_player_rating_values.append(player_rating_change.pre_match_rating_value)
                    pre_match_team_rating_values.append(pre_match_rating.teams[team_idx].rating_value)
                    pre_match_opponent_rating_values.append(pre_match_rating.teams[-team_idx + 1].rating_value)
                    player_leagues.append(player_rating_change.league)
                    team_opponent_leagues.append(opponent_team.league)
                    match_ids.append(match.id)

                    performances.append(player_rating_change.performance)
                    player_predicted_performances.append(player_rating_change.predicted_performance)
                    player_rating_changes.append(player_rating_change.rating_change_value)

        potential_feature_values = self._get_shared_rating_values(
            pre_match_team_projected_rating_values=pre_match_team_projected_rating_values,
            pre_match_opponent_projected_rating_values=pre_match_opponent_projected_rating_values,
            pre_match_player_rating_values=pre_match_player_rating_values,
            player_leagues=player_leagues,
            team_opponent_leagues=team_opponent_leagues,
            match_ids=match_ids
        )
        potential_feature_values[HistoricalRatingColumnNames.PLAYER_RATING_DIFFERENCE] = np.array(
            pre_match_player_rating_values) - np.array(
            pre_match_opponent_rating_values)
        potential_feature_values[HistoricalRatingColumnNames.RATING_DIFFERENCE] = np.array(
            pre_match_team_rating_values) - np.array(
            pre_match_opponent_rating_values)
        potential_feature_values[RatingColumnNames.PLAYER_RATING] = pre_match_player_rating_values
        potential_feature_values[HistoricalRatingColumnNames.OPPONENT_RATING] = pre_match_opponent_rating_values
        potential_feature_values[HistoricalRatingColumnNames.TEAM_RATING] = pre_match_team_rating_values
        potential_feature_values[HistoricalRatingColumnNames.RATING_MEAN] = np.array(
            pre_match_team_rating_values) * 0.5 + 0.5 * np.array(pre_match_opponent_rating_values)

        potential_feature_values[HistoricalRatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM] = np.array(
            pre_match_player_rating_values) - np.array(pre_match_team_rating_values)
        potential_feature_values[HistoricalRatingColumnNames.PERFORMANCE] = performances

        potential_feature_values[HistoricalRatingColumnNames.PLAYER_RATING_CHANGE] = player_rating_changes
        potential_feature_values[
            HistoricalRatingColumnNames.PLAYER_PREDICTED_PERFORMANCE] = player_predicted_performances

        if df is not None and self.column_names:
            self.ratings_df = df[
                [self.column_names.team_id, self.column_names.player_id, self.column_names.match_id]].assign(
                **potential_feature_values)

        return {f: potential_feature_values[f] for f in self._features_out}

    def generate_future(self, matches: Optional[list[Match]] = None, df: Optional[pd.DataFrame] = None) -> dict[
        RatingColumnNames, list[float]]:

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

        pre_match_team_projected_rating_values = []

        for match_idx, match in enumerate(matches):
            self._validate_match(match)
            pre_match_rating = PreMatchRating(
                id=match.id,
                teams=self._get_pre_match_team_ratings(match=match),
                day_number=match.day_number
            )

            for team_idx, pre_match_team in enumerate(pre_match_rating.teams):
                opponent_team = pre_match_rating.teams[-team_idx + 1]
                for player_idx, pre_match_player in enumerate(pre_match_team.players):
                    pre_match_team_projected_rating_values.append(pre_match_team.projected_rating_value)
                    pre_match_player_rating_values.append(pre_match_player.rating_value)
                    pre_match_opponent_projected_rating_values.append(opponent_team.projected_rating_value)
                    team_opponent_leagues.append(opponent_team.league)
                    player_leagues.append(pre_match_player.league)
                    match_ids.append(match.id)

        potential_feature_values = self._get_shared_rating_values(
            pre_match_team_projected_rating_values=pre_match_team_projected_rating_values,
            pre_match_opponent_projected_rating_values=pre_match_opponent_projected_rating_values,
            pre_match_player_rating_values=pre_match_player_rating_values,
            player_leagues=player_leagues,
            team_opponent_leagues=team_opponent_leagues,
            match_ids=match_ids
        )

        return {f: potential_feature_values[f] for f in self._features_out}

    def _get_shared_rating_values(self,
                                  pre_match_team_projected_rating_values: list[float],
                                  pre_match_opponent_projected_rating_values: list[float],
                                  pre_match_player_rating_values: list[float],
                                  player_leagues: list[str],
                                  team_opponent_leagues: list[str],
                                  match_ids: list[str]
                                  ) -> dict[RatingColumnNames, Any]:
        rating_differences_projected = (np.array(pre_match_team_projected_rating_values) - np.array(
            pre_match_opponent_projected_rating_values)).tolist()
        player_rating_differences_projected = (np.array(pre_match_player_rating_values) - np.array(
            pre_match_opponent_projected_rating_values)).tolist()
        player_rating_difference_from_team_projected = (np.array(pre_match_player_rating_values) - np.array(
            pre_match_team_projected_rating_values)).tolist()
        rating_means_projected = (np.array(pre_match_team_projected_rating_values) * 0.5 + 0.5 * np.array(
            pre_match_opponent_projected_rating_values)).tolist()

        return {
            RatingColumnNames.RATING_DIFFERENCE_PROJECTED: rating_differences_projected,
            RatingColumnNames.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED: player_rating_difference_from_team_projected,
            RatingColumnNames.PLAYER_RATING_DIFFERENCE_PROJECTED: player_rating_differences_projected,
            RatingColumnNames.TEAM_RATING_PROJECTED: pre_match_team_projected_rating_values,
            RatingColumnNames.OPPONENT_RATING_PROJECTED: pre_match_opponent_projected_rating_values,
            RatingColumnNames.PLAYER_LEAGUE: player_leagues,
            RatingColumnNames.OPPONENT_LEAGUE: team_opponent_leagues,
            RatingColumnNames.RATING_MEAN_PROJECTED: rating_means_projected,
            RatingColumnNames.MATCH_ID: match_ids,
            RatingColumnNames.PLAYER_RATING: pre_match_player_rating_values,
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
    def features_out(self) -> list[RatingColumnNames]:
        return self._features_out
