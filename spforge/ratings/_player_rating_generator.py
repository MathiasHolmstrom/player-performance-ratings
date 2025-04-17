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
        features_out: Optional[list[RatingKnownFeatures]] = None,
        unknown_features_out: Optional[list[RatingUnknownFeatures]] = None,
        non_predictor_known_features_out: Optional[list[RatingKnownFeatures]] = None,
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
        :param seperate_player_by_position: Creates a unique identifier for each player based on the player_id and position.
            Set to true if a players skill-level is dependent on the position they play.
        """
        self.auto_scale_performance = auto_scale_performance
        if not features_out:
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
            self.performances_generator.scale_performance = True

        if self.auto_scale_performance and not self.performances_generator:
            assert (
                performance_column
            ), "performance_column must be set if auto_scale_performance is True"
            if not performance_weights:
                self.performances_generator = PerformanceManager(
                    features=[performance_column],
                    scale_performance=True,
                )
            else:
                self.performances_generator = PerformanceWeightsManager(
                    weights=performance_weights,
                    scale_performance=True,
                )
            logging.info(
                f"Renamed performance column to performance_{performance_column}"
            )

        super().__init__(
            column_names=column_names,
            non_estimator_known_features_out=non_predictor_known_features_out,
            unknown_features_out=unknown_features_out,
            performance_column=performance_column,
            prefix=prefix,
            features_out=features_out,
            suffix=suffix,
        )
        if self.performances_generator:
            self.performance_column = self.performances_generator.performance_column
        self.auto_scale_performance = auto_scale_performance

    def reset_ratings(self):
        self._calculated_match_ids = []
        self._player_ratings = {}
        self._teams = {}
        self.start_rating_generator.reset()
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
            df.group_by()




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
