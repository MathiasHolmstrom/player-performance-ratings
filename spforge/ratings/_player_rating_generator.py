import logging
from typing import Optional, Any, Union

import numpy as np
import polars as pl
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from spforge.ratings import convert_df_to_matches

from spforge.ratings.rating_calculators import (
    RatingMeanPerformancePredictor,
)

from spforge.ratings.rating_calculators.match_rating_generator import (
    MatchRatingGenerator,
)
from spforge.ratings.enums import (
    RatingKnownFeatures,
    RatingUnknownFeatures,
)

from spforge.data_structures import (
    Match,
    PreMatchRating,
    PreMatchTeamRating,
    ColumnNames,
    TeamRatingChange,
)
from spforge.ratings.rating_generator import RatingGenerator
from spforge.transformers.fit_transformers import PerformanceWeightsManager
from spforge.transformers.fit_transformers._performance_manager import (
    ColumnWeight,
    PerformanceManager,
)
from spforge.transformers.lag_transformers._utils import transformation_validator


class PlayerRatingGenerator(RatingGenerator):
    """
    Generates ratings for players and teams based on the match-performance of the player and the ratings of the players and teams.
    Ratings are updated after a match is finished
    """

    def __init__(
        self,
        performance_column: str = "performance",
        performance_weights: Optional[
            list[Union[ColumnWeight, dict[str, float]]]
        ] = None,
        auto_scale_performance: bool = False,
        performances_generator: Optional[PerformanceWeightsManager] = None,
        match_rating_generator: Optional[MatchRatingGenerator] = None,
        features_out: Optional[list[RatingKnownFeatures]] = None,
        unknown_features_out: Optional[list[RatingUnknownFeatures]] = None,
        non_predictor_known_features_out: Optional[list[RatingKnownFeatures]] = None,
        distinct_positions: Optional[list[str]] = None,
        seperate_player_by_position: bool = False,
        column_names: Optional[ColumnNames] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        :param performance_column: The ratings will be updated by on the value of the column
        :param match_rating_generator: Passing in the MatchRatingGenerator allows for customisation of the classes and parameters used within it.
        :param features_out: A list of features where the information is available before the match is started.
            The pre-game ratings are an example of this whereas player-rating-change is a historical feature as it's first known after the match is finished.
            If none, default logic is generated based on the performance predictor used in the match_rating_generator
        :param unknown_features_out: The types of historical rating-features the rating-generator should return.
            The historical_features cannot be used as estimator features as they contain leakage, however can be interesting for exploration
        :param non_predictor_known_features_out: Rating Features to return but which are not intended to be used as estimator_features for the predictor
        :param distinct_positions: If true, the rating_difference for each player relative to the opponent player by the same position will be generated and returned.
        :param seperate_player_by_position: Creates a unique identifier for each player based on the player_id and position.
            Set to true if a players skill-level is dependent on the position they play.
        """
        match_rating_generator = match_rating_generator or MatchRatingGenerator()
        self.auto_scale_performance = auto_scale_performance
        if not features_out:
            if (
                match_rating_generator.performance_predictor.__class__
                == RatingMeanPerformancePredictor
            ):
                features_out = [RatingKnownFeatures.RATING_MEAN_PROJECTED]
            else:
                features_out = [RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]

        if performance_weights and not performances_generator:
            if isinstance(performance_weights[0], dict):
                performance_weights = [
                    ColumnWeight(**weight) for weight in performance_weights
                ]

            self.performances_generator = PerformanceWeightsManager(
                weights=performance_weights,
            )
        else:
            self.performances_generator = performances_generator

        if self.auto_scale_performance and self.performances_generator:
            self.performances_generator.auto_scale_performance = True

        if self.auto_scale_performance and not self.performances_generator:
            assert (
                performance_column
            ), "performance_column must be set if auto_scale_performance is True"
            if not performance_weights:
                self.performances_generator = PerformanceManager(
                    features=[performance_column],
                )
            else:
                self.performances_generator = PerformanceWeightsManager(
                    weights=performance_weights,
                )
            logging.info(
                f"Renamed performance column to performance_{performance_column}"
            )

        super().__init__(
            column_names=column_names,
            non_estimator_known_features_out=non_predictor_known_features_out,
            unknown_features_out=unknown_features_out,
            performance_column=performance_column,
            seperate_player_by_position=seperate_player_by_position,
            match_rating_generator=match_rating_generator,
            prefix=prefix,
            features_out=features_out,
            suffix=suffix,
        )
        if self.performances_generator:
            self.performance_column = self.performances_generator.performance_column
        self.auto_scale_performance = auto_scale_performance
        self.distinct_positions = distinct_positions

    def reset_ratings(self):
        self._calculated_match_ids = []
        self.match_rating_generator.player_ratings = {}
        self.match_rating_generator._teams = {}
        self.match_rating_generator.start_rating_generator.reset()
        self.match_rating_generator.performance_predictor.reset()

    @transformation_validator
    @nw.narwhalify
    def fit_transform(
        self,
        df: FrameT,
        column_names: Optional[ColumnNames] = None,
    ) -> IntoFrameT:
        """
        If performances_generator is defined. It will fit the performances_generator and transform the dataframe  before calculating ratings.
        Generate ratings by iterating over the dataframe, calculate predicted performance and update ratings after the match is finished.

        :param df: The dataframe from which the matches were generated. Only needed if you want to store the ratings in the class object in which case the column names must also be passed.
        :param column_names: The column names of the dataframe. Only needed if you want to store the ratings in the class object in which case the df must also be passed.

        :return: A dataframe with the original columns + estimator_features_out, historical_features_out and non_estimator_rating_features_out
        """
        self.column_names = column_names or self.column_names
        if not self.column_names:
            raise ValueError(
                "column_names must be passed into as method arguments or during RatingGenerator initialisation"
            )
        if (
            self.column_names.participation_weight is not None
            and self.column_names.participation_weight not in df.columns
        ):
            raise ValueError(
                f"participation_weight {self.column_names.participation_weight} not in df columns"
            )

        if self.performances_generator:
            df = nw.from_native(self.performances_generator.fit_transform(df))

        return self._transform_historical(df, column_names=column_names)

    @transformation_validator
    @nw.narwhalify
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        """
        Generate ratings by iterating over the dataframe, calculate predicted performance and update ratings after the match is finished.

        :param df: The dataframe from which the matches were generated. Only needed if you want to store the ratings in the class object in which case the column names must also be passed.
        :param column_names: The column names of the dataframe. Only needed if you want to store the ratings in the class object in which case the df must also be passed.

        :return: A dataframe with the original columns + estimator_features_out, historical_features_out and non_estimator_rating_features_out
        """
        if self.performances_generator:
            df = nw.from_native(self.performances_generator.transform(df))
        return self._transform_historical(df, column_names=column_names)

    def _transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> FrameT:

        self.column_names = column_names or self.column_names
        if not self.column_names:
            raise ValueError(
                "column_names must be passed into as method arguments or during RatingGenerator initialisation"
            )
        df = df.sort(
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
        )
        input_cols = df.columns
        ori_game_ids = df[self.column_names.match_id].unique().to_list()

        if self.historical_df is not None:
            df_with_new_matches = df.filter(
                ~nw.col(self.column_names.match_id).is_in(
                    nw.from_native(self.historical_df)[self.column_names.match_id]
                    .unique()
                    .to_list()
                )
            )
        else:
            df_with_new_matches = df
        if len(df_with_new_matches) > 0:
            matches = convert_df_to_matches(
                df=df_with_new_matches,
                column_names=self.column_names,
                performance_column_name=self.performance_column,
                separate_player_by_position=self.seperate_player_by_position,
            )

            potential_feature_values = self._generate_potential_feature_values(
                matches=matches, ori_df=pl.DataFrame()
            )
            exclude_cols = [
                k
                for k in potential_feature_values.keys()
                if k
                not in [
                    self.column_names.match_id,
                    self.column_names.player_id,
                    self.column_names.team_id,
                ]
            ]
            df_with_new_matches = df_with_new_matches.select(
                [c for c in df_with_new_matches.columns if c not in exclude_cols]
            ).join(
                nw.from_dict(
                    potential_feature_values,
                    native_namespace=nw.get_native_namespace(df_with_new_matches),
                ),
                on=[
                    self.column_names.match_id,
                    self.column_names.player_id,
                    self.column_names.team_id,
                ],
                how="left",
            )
            if self.suffix or self.prefix:
                if (
                    self.prefix + self.performance_column + self.suffix
                    in df_with_new_matches.columns
                ):
                    df_with_new_matches = df_with_new_matches.drop(
                        self.prefix + self.performance_column + self.suffix
                    )
                df_with_new_matches = df_with_new_matches.rename(
                    {
                        self.performance_column: self.prefix
                        + self.performance_column
                        + self.suffix
                    }
                )
            self._store_df(df_with_new_matches)
            self._calculated_match_ids = (
                nw.from_native(self.historical_df)[self.column_names.match_id]
                .unique()
                .to_list()
            )

        df = (
            df.drop([f for f in self.all_rating_features_out if f in df.columns])
            .join(
                nw.from_native(self.historical_df).select(
                    [
                        self.column_names.match_id,
                        self.column_names.player_id,
                        self.column_names.team_id,
                        *self.all_rating_features_out,
                    ]
                ),
                on=[
                    self.column_names.match_id,
                    self.column_names.team_id,
                    self.column_names.player_id,
                ],
                how="left",
            )
            .unique(
                subset=[
                    self.column_names.match_id,
                    self.column_names.team_id,
                    self.column_names.player_id,
                ],
                keep="last",
            )
        )

        df = df.sort(
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
        )
        feats_out = list(
            set(
                (
                    input_cols
                    + self.all_rating_features_out
                    + [
                        self.prefix
                        + self.performances_generator.performance_column
                        + self.suffix
                    ]
                    if self.performances_generator
                    else input_cols + self.all_rating_features_out
                )
            )
        )
        df = df.with_columns(
            nw.col(self.performance_column).alias(
                self.prefix + self.performance_column + self.suffix
            )
        )
        return df.filter(nw.col(self.column_names.match_id).is_in(ori_game_ids)).select(
            list(set(feats_out))
        )

    def _generate_potential_feature_values(self, matches: list[Match], ori_df: FrameT):
        pre_match_player_rating_values = []
        pre_match_team_rating_values = []
        pre_match_opponent_projected_rating_values = []
        pre_match_opponent_rating_values = []
        team_opponent_leagues = []
        rating_match_ids = []
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
                        match_position_ratings[team_idx][position] = float(
                            player_rating_change.pre_match_rating_value
                        )

                    pre_match_team_projected_rating_values.append(
                        float(pre_match_rating.teams[team_idx].projected_rating_value)
                    )
                    pre_match_opponent_projected_rating_values.append(
                        float(
                            pre_match_rating.teams[-team_idx + 1].projected_rating_value
                        )
                    )

                    pre_match_player_rating_values.append(
                        float(player_rating_change.pre_match_rating_value)
                    )
                    pre_match_team_rating_values.append(
                        float(pre_match_rating.teams[team_idx].rating_value)
                    )
                    pre_match_opponent_rating_values.append(
                        float(pre_match_rating.teams[-team_idx + 1].rating_value)
                    )
                    player_leagues.append(player_rating_change.league)
                    team_opponent_leagues.append(opponent_team.league)
                    team_leagues.append(team_rating_change.league)
                    rating_match_ids.append(match.id)
                    rating_update_team_ids.append(match.teams[team_idx].update_id)
                    rating_update_team_ids_opponent.append(
                        match.teams[-team_idx + 1].update_id
                    )
                    projected_participation_weights.append(
                        float(
                            match.teams[team_idx]
                            .players[player_idx]
                            .performance.projected_participation_weight
                        )
                    )

                    performances.append(player_rating_change.performance)
                    player_predicted_performances.append(
                        float(player_rating_change.predicted_performance)
                    )
                    player_rating_changes.append(
                        float(player_rating_change.rating_change_value)
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
            match_ids=rating_match_ids,
            team_ids=rating_update_team_ids,
            team_id_opponents=rating_update_team_ids_opponent,
            player_ids=player_ids,
            team_leagues=team_leagues,
            ori_df=ori_df,
        )
        potential_feature_values[
            self.prefix + RatingUnknownFeatures.PLAYER_RATING_DIFFERENCE + self.suffix
        ] = np.array(pre_match_player_rating_values) - np.array(
            pre_match_opponent_rating_values
        )
        potential_feature_values[
            self.prefix + RatingUnknownFeatures.RATING_DIFFERENCE + self.suffix
        ] = np.array(pre_match_team_rating_values) - np.array(
            pre_match_opponent_rating_values
        )
        potential_feature_values[
            self.prefix + RatingKnownFeatures.PLAYER_RATING + self.suffix
        ] = pre_match_player_rating_values
        potential_feature_values[
            self.prefix + RatingUnknownFeatures.OPPONENT_RATING + self.suffix
        ] = pre_match_opponent_rating_values
        potential_feature_values[
            self.prefix + RatingUnknownFeatures.TEAM_RATING + self.suffix
        ] = pre_match_team_rating_values
        potential_feature_values[
            self.prefix + RatingUnknownFeatures.RATING_MEAN + self.suffix
        ] = np.array(pre_match_team_rating_values) * 0.5 + 0.5 * np.array(
            pre_match_opponent_rating_values
        )

        potential_feature_values[
            self.prefix
            + RatingUnknownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM
            + self.suffix
        ] = np.array(pre_match_player_rating_values) - np.array(
            pre_match_team_rating_values
        )
        potential_feature_values[
            self.prefix + RatingUnknownFeatures.PERFORMANCE + self.suffix
        ] = performances

        potential_feature_values[
            self.prefix + RatingUnknownFeatures.PLAYER_RATING_CHANGE + self.suffix
        ] = player_rating_changes
        potential_feature_values[
            self.prefix
            + RatingUnknownFeatures.PLAYER_PREDICTED_PERFORMANCE
            + self.suffix
        ] = player_predicted_performances

        return potential_feature_values

    @transformation_validator
    @nw.narwhalify
    def transform_future(
        self,
        df: FrameT,
        matches: Optional[list[Match]] = None,
    ) -> IntoFrameT:

        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")
        df = df.sort(
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
        )

        input_cols = df.columns

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
                        match_position_ratings[team_idx][position] = float(
                            pre_match_player.rating_value
                        )
                    pre_match_team_projected_rating_values.append(
                        float(pre_match_team.projected_rating_value)
                    )
                    pre_match_player_rating_values.append(
                        float(pre_match_player.rating_value)
                    )
                    pre_match_opponent_projected_rating_values.append(
                        float(opponent_team.projected_rating_value)
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
                        float(
                            match.teams[team_idx]
                            .players[player_idx]
                            .performance.projected_participation_weight
                        )
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
            ori_df=df,
        )

        for f in self.unknown_features_out:
            potential_feature_values[f] = [np.nan] * len(pre_match_player_rating_values)

        df = df.join(
            nw.from_dict(
                potential_feature_values, native_namespace=nw.get_native_namespace(df)
            ),
            on=[
                self.column_names.match_id,
                self.column_names.player_id,
                self.column_names.team_id,
            ],
            how="left",
        )

        out_df = df.select(list(set(input_cols + self.all_rating_features_out)))

        return out_df.sort("__row_index").drop("__row_index")

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
        ori_df: FrameT,
    ) -> dict[Union[RatingKnownFeatures, RatingUnknownFeatures], Any]:

        if self.column_names.projected_participation_weight:
            data = {
                self.column_names.match_id: match_ids,
                self.column_names.team_id: team_ids,
                "team_id_opponent": team_id_opponents,
                self.prefix
                + RatingKnownFeatures.PLAYER_RATING
                + self.suffix: pre_match_player_rating_values,
                "projected_participation_weight": projected_participation_weights,
                self.column_names.player_id: player_ids,
            }

            if len(ori_df) == 0:
                df = nw.from_native(pl.DataFrame(data))
            else:
                df = nw.from_dict(
                    data,
                    native_namespace=nw.get_native_namespace(ori_df),
                )

            game_player = df.group_by(
                [
                    self.column_names.match_id,
                    self.column_names.player_id,
                    self.column_names.team_id,
                    "team_id_opponent",
                ]
            ).agg(
                [
                    nw.col(
                        self.prefix + RatingKnownFeatures.PLAYER_RATING + self.suffix
                    ).mean(),
                    nw.col("projected_participation_weight").mean(),
                ]
            )

            game_player = game_player.with_columns(
                nw.col("projected_participation_weight")
                .sum()
                .over([self.column_names.match_id, self.column_names.team_id])
                .alias("game_team_sum_projected_participation_weight")
            )

            game_player = game_player.with_columns(
                (
                    nw.col(
                        self.prefix + RatingKnownFeatures.PLAYER_RATING + self.suffix
                    )
                    * nw.col("projected_participation_weight")
                ).alias("weighted_pre_match_player_rating_value")
            )

            game_player = game_player.with_columns(
                (
                    nw.col("weighted_pre_match_player_rating_value")
                    .sum()
                    .over([self.column_names.match_id, self.column_names.team_id])
                    / nw.col("game_team_sum_projected_participation_weight")
                ).alias(
                    self.prefix
                    + RatingKnownFeatures.TEAM_RATING_PROJECTED
                    + self.suffix
                )
            )

            game_team = game_player.group_by(
                [
                    self.column_names.match_id,
                    self.column_names.team_id,
                    "team_id_opponent",
                ]
            ).agg(
                nw.col(
                    self.prefix
                    + RatingKnownFeatures.TEAM_RATING_PROJECTED
                    + self.suffix
                ).mean()
            )

            game_team_opp = game_team.select(
                [
                    self.column_names.match_id,
                    "team_id_opponent",
                    self.prefix
                    + RatingKnownFeatures.TEAM_RATING_PROJECTED
                    + self.suffix,
                ]
            ).rename(
                {
                    self.prefix
                    + RatingKnownFeatures.TEAM_RATING_PROJECTED
                    + self.suffix: self.prefix
                    + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                    + self.suffix
                }
            )

            game_team = game_team.join(
                game_team_opp,
                left_on=[self.column_names.match_id, self.column_names.team_id],
                right_on=[self.column_names.match_id, "team_id_opponent"],
                how="left",
            )

            game_player = game_player.join(
                game_team.select(
                    [
                        self.column_names.match_id,
                        self.column_names.team_id,
                        self.prefix
                        + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                        + self.suffix,
                    ]
                ),
                on=[self.column_names.match_id, self.column_names.team_id],
                how="left",
            )

            game_player = game_player.with_columns(
                (
                    nw.col(
                        self.prefix
                        + RatingKnownFeatures.TEAM_RATING_PROJECTED
                        + self.suffix
                    )
                    + nw.col(
                        self.prefix
                        + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                        + self.suffix
                    )
                ).alias(
                    self.prefix
                    + RatingKnownFeatures.RATING_MEAN_PROJECTED
                    + self.suffix
                )
                / 2
            )

            df = df.select(
                [
                    self.column_names.match_id,
                    self.column_names.player_id,
                    self.column_names.team_id,
                ]
            ).join(
                game_player.select(
                    [
                        self.column_names.match_id,
                        self.column_names.player_id,
                        self.column_names.team_id,
                        self.prefix
                        + RatingKnownFeatures.TEAM_RATING_PROJECTED
                        + self.suffix,
                        self.prefix
                        + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                        + self.suffix,
                        self.prefix
                        + RatingKnownFeatures.RATING_MEAN_PROJECTED
                        + self.suffix,
                        self.prefix + RatingKnownFeatures.PLAYER_RATING + self.suffix,
                    ]
                ),
                on=[
                    self.column_names.match_id,
                    self.column_names.player_id,
                    self.column_names.team_id,
                ],
                how="left",
            )

            rating_differences_projected = (
                df[
                    self.prefix
                    + RatingKnownFeatures.TEAM_RATING_PROJECTED
                    + self.suffix
                ]
                - df[
                    self.prefix
                    + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                    + self.suffix
                ]
            ).to_list()

            player_rating_difference_from_team_projected = (
                df[self.prefix + RatingKnownFeatures.PLAYER_RATING + self.suffix]
                - df[
                    self.prefix
                    + RatingKnownFeatures.TEAM_RATING_PROJECTED
                    + self.suffix
                ]
            ).to_list()

            player_rating_differences_projected = (
                df[self.prefix + RatingKnownFeatures.PLAYER_RATING + self.suffix]
                - df[
                    self.prefix
                    + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                    + self.suffix
                ]
            ).to_list()

            rating_means_projected = df[
                self.prefix + RatingKnownFeatures.RATING_MEAN_PROJECTED + self.suffix
            ].to_list()

            pre_match_opponent_projected_rating_values = df[
                self.prefix
                + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
                + self.suffix
            ].to_list()
            pre_match_team_projected_rating_values = df[
                self.prefix + RatingKnownFeatures.TEAM_RATING_PROJECTED + self.suffix
            ].to_list()
            pre_match_player_rating_values = df[
                self.prefix + RatingKnownFeatures.PLAYER_RATING + self.suffix
            ].to_list()

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
            + RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED
            + self.suffix: rating_differences_projected,
            self.prefix
            + RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED
            + self.suffix: player_rating_difference_from_team_projected,
            self.prefix
            + RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED
            + self.suffix: player_rating_differences_projected,
            self.prefix
            + RatingKnownFeatures.TEAM_RATING_PROJECTED
            + self.suffix: pre_match_team_projected_rating_values,
            self.prefix
            + RatingKnownFeatures.OPPONENT_RATING_PROJECTED
            + self.suffix: pre_match_opponent_projected_rating_values,
            self.prefix
            + RatingKnownFeatures.PLAYER_RATING
            + self.suffix: pre_match_player_rating_values,
            self.prefix
            + RatingKnownFeatures.PLAYER_LEAGUE
            + self.suffix: player_leagues,
            self.prefix
            + RatingKnownFeatures.OPPONENT_LEAGUE
            + self.suffix: team_opponent_leagues,
            self.prefix + RatingKnownFeatures.TEAM_LEAGUE + self.suffix: team_leagues,
            self.prefix
            + RatingKnownFeatures.RATING_MEAN_PROJECTED
            + self.suffix: rating_means_projected,
            self.column_names.match_id: match_ids,
            self.column_names.player_id: player_ids,
            self.column_names.team_id: team_ids,
        }

        if self.distinct_positions:
            for position, rating_values in position_rating_difference_values.items():
                return_values[
                    self.prefix
                    + RatingKnownFeatures.RATING_DIFFERENCE_POSITION
                    + self.suffix
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

    def _store_df(
        self, df: nw.DataFrame, additional_cols_to_use: Optional[list[str]] = None
    ):

        cols = list(
            {
                *self.all_rating_features_out,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
                self.column_names.parent_team_id,
                self.column_names.update_match_id,
                self.column_names.start_date,
            }
        )
        if self.column_names.player_id not in df.columns:
            cols.remove(self.column_names.player_id)
        if self.column_names.participation_weight in df.columns:
            cols += [self.column_names.participation_weight]
        if self.column_names.projected_participation_weight in df.columns:
            cols += [self.column_names.projected_participation_weight]

        if additional_cols_to_use:
            cols += additional_cols_to_use

        if self._df is None:
            self._df = df.select(cols)
        else:
            self._df = nw.concat([nw.from_native(self._df), df.select(cols)])

        sort_cols = (
            [
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id in self._df.columns
            else [
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )

        self._df = (
            self._df.sort(
                sort_cols
                #   descending=True
            )
            .unique(
                subset=sort_cols,
                maintain_order=True,
            )
            .to_native()
        )
