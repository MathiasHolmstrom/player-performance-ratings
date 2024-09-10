from typing import Optional, Any, Union

import numpy as np
import pandas as pd

from player_performance_ratings.ratings import convert_df_to_matches
from player_performance_ratings.ratings.rating_calculators import (
    RatingMeanPerformancePredictor,
)
from player_performance_ratings.ratings.rating_calculators.performance_predictor import (
    RatingNonOpponentPerformancePredictor,
)
from player_performance_ratings.ratings.rating_calculators.match_rating_generator import (
    MatchRatingGenerator,
)
from player_performance_ratings.ratings.enums import (
    RatingKnownFeatures,
    RatingHistoricalFeatures,
)

from player_performance_ratings.data_structures import (
    Match,
    PreMatchRating,
    PreMatchTeamRating,
    ColumnNames,
    TeamRatingChange,
)
from player_performance_ratings.ratings.rating_generator import RatingGenerator


class UpdateRatingGenerator(RatingGenerator):
    """
    Generates ratings for players and teams based on the match-performance of the player and the ratings of the players and teams.
    Ratings are updated after a match is finished
    """

    def __init__(
        self,
        performance_column: str = "performance",
        match_rating_generator: Optional[MatchRatingGenerator] = None,
        known_features_out: Optional[list[RatingKnownFeatures]] = None,
        historical_features_out: Optional[list[RatingHistoricalFeatures]] = None,
        non_estimator_known_features_out: Optional[list[RatingKnownFeatures]] = None,
        distinct_positions: Optional[list[str]] = None,
        seperate_player_by_position: bool = False,
        prefix: str = "",
    ):
        """
        :param performance_column: The ratings will be updated by on the value of the column
        :param match_rating_generator: Passing in the MatchRatingGenerator allows for customisation of the classes and parameters used within it.
        :param known_features_out: A list of features where the information is available before the match is started.
            The pre-game ratings are an example of this whereas player-rating-change is a historical feature as it's first known after the match is finished.
            If none, default logic is generated based on the performance predictor used in the match_rating_generator
        :param historical_features_out: The types of historical rating-features the rating-generator should return.
            The historical_features cannot be used as estimator features as they contain leakage, however can be interesting for exploration
        :param non_estimator_known_features_out: Rating Features to return but which are not intended to be used as estimator_features for the predictor
        :param distinct_positions: If true, the rating_difference for each player relative to the opponent player by the same position will be generated and returned.
        :param seperate_player_by_position: Creates a unique identifier for each player based on the player_id and position.
            Set to true if a players skill-level is dependent on the position they play.
        """

        super().__init__(
            non_estimator_known_features_out=non_estimator_known_features_out,
            historical_features_out=historical_features_out,
            performance_column=performance_column,
            seperate_player_by_position=seperate_player_by_position,
            match_rating_generator=match_rating_generator or MatchRatingGenerator(),
            prefix=prefix,
        )
        self.distinct_positions = distinct_positions
        self._non_estimator_rating_features_out = non_estimator_known_features_out or []

        self._known_features_out = (
            [self.prefix + f for f in known_features_out]
            if known_features_out is not None
            else (
                [self.prefix + RatingKnownFeatures.RATING_MEAN_PROJECTED]
                if isinstance(
                    self.match_rating_generator.performance_predictor,
                    RatingMeanPerformancePredictor,
                )
                else (
                    [self.prefix + RatingKnownFeatures.PLAYER_RATING]
                    if isinstance(
                        self.match_rating_generator.performance_predictor,
                        RatingNonOpponentPerformancePredictor,
                    )
                    else [self.prefix + RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
                )
            )
        )

        if self.distinct_positions:
            self._known_features_out += [
                self.prefix + RatingKnownFeatures.RATING_DIFFERENCE_POSITION + "_" + p
                for p in self.distinct_positions
            ]

    def reset_ratings(self):
        self._calculated_match_ids = []
        self.match_rating_generator.player_ratings = {}
        self.match_rating_generator._teams = {}
        self.match_rating_generator.start_rating_generator.reset()
        self.match_rating_generator.performance_predictor.reset()

    def generate_historical_by_matches(
        self,
        matches: list[Match],
        column_names: ColumnNames,
        historical_features_out: Optional[list[RatingHistoricalFeatures]] = None,
        known_features_out: Optional[list[RatingKnownFeatures]] = None,
    ) -> dict[Union[RatingKnownFeatures, RatingHistoricalFeatures], list[float]]:
        """
        Generate ratings by iterating over each match, calculate predicted performance and update ratings after the match is finished.

        :param matches: A list of Matches containing the data required to calculate ratings
        :param column_names: The column names of the dataframe. Only needed if you want to store the ratings in the class object in which case the df must also be passed.
        :param historical_features_out: The types of historical rating-features the rating-generator should return.

        :return: A dictionary containing a list of match-rating
        """
        self.column_names = column_names
        historical_features_out = (
            historical_features_out or self._historical_features_out
        )
        known_features_out = known_features_out or self._known_features_out
        potential_feature_values = self._generate_potential_feature_values(
            matches=matches
        )
        return {
            f: potential_feature_values[f]
            for f in list(set(known_features_out + historical_features_out + self._non_estimator_rating_features_out))
        }

    def generate_historical(
        self,
        df: pd.DataFrame,
        column_names: ColumnNames,
        historical_features_out: Optional[list[RatingHistoricalFeatures]] = None,
        known_features_out: Optional[list[RatingKnownFeatures]] = None,
    ) -> pd.DataFrame:
        """
        Generate ratings by iterating over the dataframe, calculate predicted performance and update ratings after the match is finished.

        :param df: The dataframe from which the matches were generated. Only needed if you want to store the ratings in the class object in which case the column names must also be passed.
        :param column_names: The column names of the dataframe. Only needed if you want to store the ratings in the class object in which case the df must also be passed.

        :return: A dataframe with the original columns + estimator_features_out, historical_features_out and non_estimator_rating_features_out
        """
        input_cols = df.columns.tolist()
        self.column_names = column_names
        if (
            self.column_names.participation_weight is not None
            and self.column_names.participation_weight not in df.columns
        ):
            raise ValueError(
                f"participation_weight {self.column_names.participation_weight} not in df columns"
            )

        matches = convert_df_to_matches(
            df=df,
            column_names=self.column_names,
            performance_column_name=self.performance_column,
            separate_player_by_position=self.seperate_player_by_position,
        )

        potential_feature_values = self._generate_potential_feature_values(
            matches=matches
        )

        df = df.assign(**potential_feature_values)
        known_features_out = known_features_out or self.known_features_return
        historical_features_out = (
            historical_features_out or self._historical_features_out
        )
        self._calculated_match_ids = df[self.column_names.match_id].unique().tolist()
        return df[list(set(input_cols + known_features_out + historical_features_out + self._non_estimator_rating_features_out))]

    def _generate_potential_feature_values(self, matches: list[Match]):
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
                day_number=match.day_number,
            )
            match_team_rating_changes = self._create_match_team_rating_changes(
                match=match, pre_match_rating=pre_match_rating
            )
            team_rating_changes += match_team_rating_changes

            if (
                match_idx == len(matches) - 1
                or matches[match_idx + 1].update_id != match.update_id
            ):
                self._update_ratings(team_rating_changes=team_rating_changes)
                team_rating_changes = []

            match_position_ratings = []

            for team_idx, team_rating_change in enumerate(match_team_rating_changes):
                match_position_ratings.append({})
                opponent_team = match_team_rating_changes[-team_idx + 1]
                for player_idx, player_rating_change in enumerate(
                    team_rating_change.players
                ):

                    position = match.teams[team_idx].players[player_idx].position
                    if position:
                        match_position_ratings[team_idx][
                            position
                        ] = player_rating_change.pre_match_rating_value

                    pre_match_team_projected_rating_values.append(
                        pre_match_rating.teams[team_idx].projected_rating_value
                    )
                    pre_match_opponent_projected_rating_values.append(
                        pre_match_rating.teams[-team_idx + 1].projected_rating_value
                    )

                    pre_match_player_rating_values.append(
                        player_rating_change.pre_match_rating_value
                    )
                    pre_match_team_rating_values.append(
                        pre_match_rating.teams[team_idx].rating_value
                    )
                    pre_match_opponent_rating_values.append(
                        pre_match_rating.teams[-team_idx + 1].rating_value
                    )
                    player_leagues.append(player_rating_change.league)
                    team_opponent_leagues.append(opponent_team.league)
                    team_leagues.append(team_rating_change.league)
                    rating_update_match_ids.append(match.update_id)
                    rating_update_team_ids.append(match.teams[team_idx].update_id)
                    rating_update_team_ids_opponent.append(
                        match.teams[-team_idx + 1].update_id
                    )
                    projected_participation_weights.append(
                        match.teams[team_idx]
                        .players[player_idx]
                        .performance.projected_participation_weight
                    )

                    performances.append(player_rating_change.performance)
                    player_predicted_performances.append(
                        player_rating_change.predicted_performance
                    )
                    player_rating_changes.append(
                        player_rating_change.rating_change_value
                    )
                    player_ids.append(player_rating_change.id)

            if self.distinct_positions:
                for team_idx in range(len(match_team_rating_changes)):
                    player_per_team_count = len(
                        match_team_rating_changes[team_idx].players
                    )

                    for position in self.distinct_positions:
                        if position not in position_rating_difference_values:
                            position_rating_difference_values[position] = []
                        if (
                            position in match_position_ratings[team_idx]
                            and position in match_position_ratings[-team_idx + 1]
                        ):
                            position_rating_difference_values[position] += [
                                match_position_ratings[team_idx][position]
                                - match_position_ratings[-team_idx + 1][position]
                            ] * player_per_team_count
                        else:
                            position_rating_difference_values[position] += [
                                0
                            ] * player_per_team_count

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
        potential_feature_values[
            self.prefix + RatingHistoricalFeatures.PLAYER_RATING_DIFFERENCE
        ] = np.array(pre_match_player_rating_values) - np.array(
            pre_match_opponent_rating_values
        )
        potential_feature_values[
            self.prefix + RatingHistoricalFeatures.RATING_DIFFERENCE
        ] = np.array(pre_match_team_rating_values) - np.array(
            pre_match_opponent_rating_values
        )
        potential_feature_values[RatingKnownFeatures.PLAYER_RATING] = (
            pre_match_player_rating_values
        )
        potential_feature_values[
            self.prefix + RatingHistoricalFeatures.OPPONENT_RATING
        ] = pre_match_opponent_rating_values
        potential_feature_values[self.prefix + RatingHistoricalFeatures.TEAM_RATING] = (
            pre_match_team_rating_values
        )
        potential_feature_values[self.prefix + RatingHistoricalFeatures.RATING_MEAN] = (
            np.array(pre_match_team_rating_values) * 0.5
            + 0.5 * np.array(pre_match_opponent_rating_values)
        )

        potential_feature_values[
            self.prefix + RatingHistoricalFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM
        ] = np.array(pre_match_player_rating_values) - np.array(
            pre_match_team_rating_values
        )
        potential_feature_values[self.prefix + RatingHistoricalFeatures.PERFORMANCE] = (
            performances
        )

        potential_feature_values[
            self.prefix + RatingHistoricalFeatures.PLAYER_RATING_CHANGE
        ] = player_rating_changes
        potential_feature_values[
            self.prefix + RatingHistoricalFeatures.PLAYER_PREDICTED_PERFORMANCE
        ] = player_predicted_performances

        return potential_feature_values

    def generate_future(
        self,
        df: Optional[pd.DataFrame],
        matches: Optional[list[Match]] = None,
        historical_features_out: Optional[list[RatingHistoricalFeatures]] = None,
        known_features_out: Optional[list[RatingKnownFeatures]] = None,
    ) -> pd.DataFrame:

        input_cols = df.columns.tolist()

        if (
            matches is not None
            and len(matches) > 0
            and not isinstance(matches[0], Match)
        ):
            raise ValueError("matches must be a list of Match objects")

        if matches is None and df is None:
            raise ValueError("If matches is not passed, df must be massed")

        if matches is None:
            matches = convert_df_to_matches(
                df=df,
                column_names=self.column_names,
                performance_column_name=self.performance_column,
                separate_player_by_position=self.seperate_player_by_position,
            )

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
                day_number=match.day_number,
            )

            for team_idx, pre_match_team in enumerate(pre_match_rating.teams):
                match_position_ratings.append({})
                opponent_team = pre_match_rating.teams[-team_idx + 1]
                for player_idx, pre_match_player in enumerate(pre_match_team.players):
                    position = match.teams[team_idx].players[player_idx].position
                    if position:
                        match_position_ratings[team_idx][
                            position
                        ] = pre_match_player.rating_value
                    pre_match_team_projected_rating_values.append(
                        pre_match_team.projected_rating_value
                    )
                    pre_match_player_rating_values.append(pre_match_player.rating_value)
                    pre_match_opponent_projected_rating_values.append(
                        opponent_team.projected_rating_value
                    )
                    team_opponent_leagues.append(opponent_team.league)
                    team_leagues.append(pre_match_team.league)
                    player_leagues.append(pre_match_player.league)
                    match_ids.append(match.id)
                    player_ids.append(pre_match_player.id)
                    rating_update_team_ids.append(match.teams[team_idx].update_id)
                    rating_update_team_ids_opponent.append(
                        match.teams[-team_idx + 1].update_id
                    )
                    projected_participation_weights.append(
                        match.teams[team_idx]
                        .players[player_idx]
                        .performance.projected_participation_weight
                    )

            if self.distinct_positions:
                for team_idx in range(len(pre_match_rating.teams)):
                    player_per_team_count = len(
                        pre_match_rating.teams[team_idx].players
                    )

                    for position in self.distinct_positions:
                        if position not in position_rating_difference_values:
                            position_rating_difference_values[position] = []
                        if (
                            position in match_position_ratings[team_idx]
                            and position in match_position_ratings[-team_idx + 1]
                        ):
                            position_rating_difference_values[position] += [
                                match_position_ratings[team_idx][position]
                                - match_position_ratings[-team_idx + 1][position]
                            ] * player_per_team_count
                        else:
                            position_rating_difference_values[position] += [
                                0
                            ] * player_per_team_count

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

        for f in self._historical_features_out:
            potential_feature_values[f] = [np.nan] * len(pre_match_player_rating_values)

        df = df.assign(**potential_feature_values)

        known_features_return = known_features_out or self.known_features_return
        historical_features_out = (
            historical_features_out or self._historical_features_out
        )
        return df[list(set(input_cols + known_features_return + historical_features_out + self._non_estimator_rating_features_out))]

    def _get_shared_rating_values(
        self,
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
    ) -> dict[Union[RatingKnownFeatures, RatingHistoricalFeatures], Any]:

        if self.column_names.projected_participation_weight:
            df = pd.DataFrame(
                {
                    "match_id": match_ids,
                    "team_id": team_ids,
                    "team_id_opponent": team_id_opponents,
                    RatingKnownFeatures.PLAYER_RATING: pre_match_player_rating_values,
                    "projected_participation_weight": projected_participation_weights,
                    "player_id": player_ids,
                }
            )

            game_player = (
                df.groupby(["match_id", "player_id", "team_id", "team_id_opponent"])[
                    [
                        RatingKnownFeatures.PLAYER_RATING,
                        "projected_participation_weight",
                    ]
                ]
                .mean()
                .reset_index()
            )

            game_player["game_team_sum_projected_participation_weight"] = (
                game_player.groupby(["match_id", "team_id"])[
                    "projected_participation_weight"
                ].transform("sum")
            )

            game_player["weighted_pre_match_player_rating_value"] = (
                game_player[self.prefix + RatingKnownFeatures.PLAYER_RATING]
                * game_player["projected_participation_weight"]
            )

            game_player[self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED] = (
                game_player.groupby(["match_id", "team_id"])[
                    "weighted_pre_match_player_rating_value"
                ].transform("sum")
                / game_player["game_team_sum_projected_participation_weight"]
            )

            game_team = (
                game_player.groupby(["match_id", "team_id", "team_id_opponent"])[
                    self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED
                ]
                .mean()
                .reset_index()
            )

            game_team = game_team.merge(
                game_team[
                    [
                        "match_id",
                        "team_id_opponent",
                        self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED,
                    ]
                ].rename(
                    columns={
                        self.prefix
                        + RatingKnownFeatures.TEAM_RATING_PROJECTED: self.prefix
                        + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                    }
                ),
                left_on=["match_id", "team_id"],
                right_on=["match_id", "team_id_opponent"],
            )

            game_player = game_player.merge(
                game_team[
                    [
                        "match_id",
                        "team_id",
                        self.prefix + RatingKnownFeatures.OPPONENT_RATING_PROJECTED,
                    ]
                ],
                on=["match_id", "team_id"],
            )

            game_player[self.prefix + RatingKnownFeatures.RATING_MEAN_PROJECTED] = (
                game_player[self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED]
                + game_player[
                    self.prefix + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                ]
            ) / 2

            df = df[["match_id", "player_id"]].merge(
                game_player[
                    [
                        "match_id",
                        "player_id",
                        self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED,
                        self.prefix + RatingKnownFeatures.OPPONENT_RATING_PROJECTED,
                        self.prefix + RatingKnownFeatures.RATING_MEAN_PROJECTED,
                        self.prefix + RatingKnownFeatures.PLAYER_RATING,
                    ]
                ],
                on=["match_id", "player_id"],
                how="left",
            )

            rating_differences_projected = (
                df[self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED]
                - df[self.prefix + RatingKnownFeatures.OPPONENT_RATING_PROJECTED]
            ).tolist()
            player_rating_difference_from_team_projected = (
                df[self.prefix + RatingKnownFeatures.PLAYER_RATING]
                - df[self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED]
            ).tolist()
            player_rating_differences_projected = (
                df[self.prefix + RatingKnownFeatures.PLAYER_RATING]
                - df[self.prefix + RatingKnownFeatures.OPPONENT_RATING_PROJECTED]
            ).tolist()
            rating_means_projected = df[
                self.prefix + RatingKnownFeatures.RATING_MEAN_PROJECTED
            ].tolist()
            pre_match_opponent_projected_rating_values = df[
                self.prefix + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
            ].tolist()
            pre_match_team_projected_rating_values = df[
                self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED
            ].tolist()
            pre_match_player_rating_values = df[
                self.prefix + RatingKnownFeatures.PLAYER_RATING
            ].tolist()

        else:
            rating_differences_projected = (
                np.array(pre_match_team_projected_rating_values)
                - np.array(pre_match_opponent_projected_rating_values)
            ).tolist()
            player_rating_differences_projected = (
                np.array(pre_match_player_rating_values)
                - np.array(pre_match_opponent_projected_rating_values)
            ).tolist()
            player_rating_difference_from_team_projected = (
                np.array(pre_match_player_rating_values)
                - np.array(pre_match_team_projected_rating_values)
            ).tolist()
            rating_means_projected = (
                np.array(pre_match_team_projected_rating_values) * 0.5
                + 0.5 * np.array(pre_match_opponent_projected_rating_values)
            ).tolist()

        return_values = {
            self.prefix
            + RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED: rating_differences_projected,
            self.prefix
            + RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED: player_rating_difference_from_team_projected,
            self.prefix
            + RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED: player_rating_differences_projected,
            self.prefix
            + RatingKnownFeatures.TEAM_RATING_PROJECTED: pre_match_team_projected_rating_values,
            self.prefix
            + RatingKnownFeatures.OPPONENT_RATING_PROJECTED: pre_match_opponent_projected_rating_values,
            self.prefix
            + RatingKnownFeatures.PLAYER_RATING: pre_match_player_rating_values,
            self.prefix + RatingKnownFeatures.PLAYER_LEAGUE: player_leagues,
            self.prefix + RatingKnownFeatures.OPPONENT_LEAGUE: team_opponent_leagues,
            self.prefix + RatingKnownFeatures.TEAM_LEAGUE: team_leagues,
            self.prefix
            + RatingKnownFeatures.RATING_MEAN_PROJECTED: rating_means_projected,
            self.prefix + RatingKnownFeatures.MATCH_ID: match_ids,
        }

        if self.distinct_positions:
            for position, rating_values in position_rating_difference_values.items():
                return_values[
                    self.prefix
                    + RatingKnownFeatures.RATING_DIFFERENCE_POSITION
                    + "_"
                    + position
                ] = rating_values

        return return_values

    def _create_match_team_rating_changes(
        self, match: Match, pre_match_rating: PreMatchRating
    ) -> list[TeamRatingChange]:

        team_rating_changes = []

        for team_idx, pre_match_team_rating in enumerate(pre_match_rating.teams):
            team_rating_change = self.match_rating_generator.generate_rating_change(
                day_number=match.day_number,
                pre_match_team_rating=pre_match_team_rating,
                pre_match_opponent_team_rating=pre_match_rating.teams[-team_idx + 1],
            )
            team_rating_changes.append(team_rating_change)

        return team_rating_changes

    def _update_ratings(self, team_rating_changes: list[TeamRatingChange]):

        for idx, team_rating_change in enumerate(team_rating_changes):
            self.match_rating_generator.update_rating_by_team_rating_change(
                team_rating_change=team_rating_change,
                opponent_team_rating_change=team_rating_changes[-idx + 1],
            )

    def _get_pre_match_team_ratings(self, match: Match) -> list[PreMatchTeamRating]:
        pre_match_team_ratings = []
        for match_team in match.teams:
            pre_match_team_ratings.append(
                self.match_rating_generator.generate_pre_match_team_rating(
                    match_team=match_team, day_number=match.day_number
                )
            )

        return pre_match_team_ratings
