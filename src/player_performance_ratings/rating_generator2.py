from __future__ import annotations

import logging

from typing import List, Dict, Optional

import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator

from src.player_performance_ratings.data_structures import PlayerRating
from src.player_performance_ratings.match_rating.match_rating_calculator import MatchGenerator
from src.player_performance_ratings.data_structures import MatchOutValues, PredictedRatingMethod, RatingColumnNames, \
    MatchPerformanceRating, \
    RatingType, MatchPlayer, Match, ConfigurationColumnNames

HOUR_NUMBER_COLUMN_NAME = "hour_number"


class RatingTransformer(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            config_column_names: ConfigurationColumnNames,
            predicted_performance_method: PredictedRatingMethod = PredictedRatingMethod.DEFAULT,

            league_rating_regularizer: float = 0,

            max_days_ago: int = 90,
            min_count_using_percentiles: int = 150,

            min_match_ratings_for_team: int = 20,
            max_days_ago_league_entities: int = 120,
            rating_diff_coef: float = 0.005757,
            rating_diff_team_from_entity_coef: float = 0.0,
            team_rating_diff_coef: float = 0.0,
            return_values: Optional[List[RatingColumnNames]] = None,
            passthrough: bool = False,
            add_return_feature_names: List = None,
            average_teams_together_for_game: bool = False,
            out_column_name_prefix=''
    ):

        self.league_rating_regularizer = league_rating_regularizer


        self.return_values = return_values


        self.add_return_feature_names = add_return_feature_names or []

        self.rating_diff_team_from_entity_coef = rating_diff_team_from_entity_coef
        self.team_rating_diff_coef = team_rating_diff_coef
        self.max_days_ago_league_entities = max_days_ago_league_entities
        self.min_match_ratings_for_team = min_match_ratings_for_team
        self.min_count_using_percentiles = min_count_using_percentiles
        self.max_days_ago = max_days_ago


        self.config_column_names = config_column_names
        self.predicted_performance_method = predicted_performance_method


        self.out_column_name_prefix = out_column_name_prefix
        self.average_teams_together_for_game = average_teams_together_for_game
        self.passthrough = passthrough
        self.match_generator: MatchGenerator

        self.calculated_match_ids: List[str] = []
        self._instantiate_new()

    @property
    def entity_ratings(self) -> Dict[str, PlayerRating]:
        return self.match_generator.match_rating_calculator.player_ratings



    def fit(self, X: pd.DataFrame, y: pd.Series) -> RatingTransformer:
        self._instantiate_new()

        self.feature_names_in = X.columns.tolist()
        self._generate_out_dataframe_values(X, y)
        self.feature_names_out_ = []
        if self.passthrough:
            self.feature_names_out_ = X.columns.tolist()
        else:
            self.feature_names_out_ = self.add_return_feature_names
        if self.predicted_performance_method == PredictedRatingMethod.DEFAULT:
            if self.out_team_rating_difference:
                if self.config_column_names.projected_participation_weight is not None:
                    self.feature_names_out_.append(
                        self.out_column_name_prefix + RatingColumnNames.rating_difference_projected.value)
                else:
                    self.feature_names_out_.append(
                        self.out_column_name_prefix + RatingColumnNames.rating_difference.value)

        elif self.predicted_performance_method == PredictedRatingMethod.MEAN_RATING:
            if self.out_team_rating_difference:
                if self.config_column_names.projected_participation_weight is not None:
                    self.feature_names_out_.append(
                        self.out_column_name_prefix + RatingColumnNames.rating_mean_projected.value)
                else:
                    self.feature_names_out_.append(
                        self.out_column_name_prefix + RatingColumnNames.rating_mean.value)

        return self

    def transform(self,
                  X: pd.DataFrame,
                  y=None,
                  passthrough: bool = None,
                  ) -> pd.DataFrame:

        X_copy = X.copy()

        if passthrough is None:
            passthrough = self.passthrough

        self._generate_out_dataframe_values(X_copy, y)
        match_ids = X_copy[self.config_column_names.match_id].unique().tolist()
        combined_column_out_values = {}

        rating_out_column_names = []

        count = 0
        for match_id in match_ids:
            if match_id not in self.match_id_to_out_df_column_values:
                print(match_id)
            match_out_values = self.match_id_to_out_df_column_values[match_id]

            rating_column_names = [c for c in match_out_values.RatingValues]

            if self.out_entity_rating_difference_from_team:
                rating_column_names.append(RatingColumnNames.entity_rating_difference_from_team)
            if self.out_entity_rating_difference_from_team_projected:
                rating_column_names.append(RatingColumnNames.entity_rating_difference_from_team_projected)

            for rating_column_name in rating_column_names:
                if rating_column_name in match_out_values.RatingValues:
                    column_out_values = match_out_values.RatingValues[rating_column_name]
                elif rating_column_name == RatingColumnNames.entity_rating_difference_from_team:
                    column_out_values = []
                    for idx, entity_rating in enumerate(match_out_values.RatingValues[RatingColumnNames.entity_rating]):
                        team_rating = match_out_values.RatingValues[RatingColumnNames.team_rating][idx]
                        column_out_values.append(entity_rating - team_rating)
                else:
                    column_out_values = []
                    for idx, entity_rating in enumerate(
                            match_out_values.RatingValues[RatingColumnNames.entity_rating]):
                        team_rating_projected = match_out_values.RatingValues[RatingColumnNames.team_rating_projected][
                            idx]
                        column_out_values.append(entity_rating - team_rating_projected)
                count += len(column_out_values)

                if rating_column_name == RatingColumnNames.rating_mean_projected:
                    if out_rating_mean_projected is False:
                        continue

                if rating_column_name == RatingColumnNames.rating_mean:
                    if out_rating_mean is False:
                        continue

                out_column_name = self.out_column_name_prefix + rating_column_name.value
                if out_column_name not in rating_out_column_names:
                    rating_out_column_names.append(out_column_name)
                if out_column_name not in combined_column_out_values:
                    combined_column_out_values[out_column_name] = []
                combined_column_out_values[out_column_name] += column_out_values

        for out_column_name, out_values in combined_column_out_values.items():
            X_copy[out_column_name] = out_values

        if passthrough is False:
            feature_names_out_ = rating_out_column_names + [c for c in self.add_return_feature_names if
                                                            c not in rating_out_column_names]
        else:
            feature_names_out_ = self.feature_names_in + [c for c in rating_out_column_names]
            for c in feature_names_out_:
                if c not in X_copy.columns:
                    feature_names_out_.remove(c)
                    logging.info(f"removed returning feature {c} as it was not in input dataframe")

        return X_copy[feature_names_out_]

    def _generate_out_dataframe_values(self, X: pd.DataFrame, y: pd.Series):
        sorted_correctly = self._validate_sorting(X)
        if not sorted_correctly:
            raise ValueError("X needs to be sorted by date, game_id, team_id in ascending order")

        df = X.copy()
        df['__TARGET'] = y
        col_names = self.config_column_names
        df[col_names.start_date_time] = pd.to_datetime(df[col_names.start_date_time], format='%Y-%m-%d %H:%M:%S')
        try:
            date_time = df[col_names.start_date_time].dt.tz_convert('UTC')
        except TypeError:
            date_time = df[col_names.start_date_time].dt.tz_localize('UTC')
        df[HOUR_NUMBER_COLUMN_NAME] = (date_time - pd.Timestamp("1970-01-01").tz_localize('UTC')) // pd.Timedelta(
            '1h')

        league_in_df = False
        if self.config_column_names.league in df.columns.tolist():
            league_in_df = True

        input_entity_rating_in_df = False
        if RatingColumnNames.entity_rating in df.columns.tolist():
            input_entity_rating_in_df = True

        input_opponent_rating_in_df = False
        if RatingColumnNames.opponent_rating in df.columns.tolist():
            input_opponent_rating_in_df = True

        participation_weight_in_df = False
        if self.config_column_names.participation_weight in df.columns.tolist():
            participation_weight_in_df = True

        projected_participation_weight_in_df = False
        if self.config_column_names.projected_participation_weight in df.columns.tolist():
            projected_participation_weight_in_df = True

        team_players_percentage_playing_time_in_df = False
        if self.config_column_names.team_players_percentage_playing_time in df.columns.tolist():
            team_players_percentage_playing_time_in_df = True

        player_id_in_df = False
        if self.config_column_names.player_id in df.columns.tolist():
            player_id_in_df = True

        use_parent_match_id = False
        if self.config_column_names.parent_match_id is not None and col_names.parent_match_id in df.columns:
            use_parent_match_id = True

        prev_match_id = None
        prev_parent_match_id = None
        match = None

        data_dict = df.to_dict('records')
        self.ot = 0
        parent_matches = []
        for row in data_dict:
            match_id = row[col_names.match_id]

            parent_match_id = None
            if use_parent_match_id:
                parent_match_id = row[col_names.parent_match_id]

            if match_id in self.match_id_to_out_df_column_values:
                continue

            if match_id != prev_match_id:

                if prev_match_id is not None:
                    parent_matches.append(match)

                match = Match(
                    match_id=row[col_names.match_id],
                    entities=[],
                    team_ids=[],
                    day_number=int(row[HOUR_NUMBER_COLUMN_NAME] / 24),
                )

            if use_parent_match_id and parent_match_id != prev_parent_match_id and prev_parent_match_id is not None or \
                    use_parent_match_id is False and match_id != prev_match_id and prev_match_id is not None:
                self._update_matches(parent_matches, projected_participation_weight_in_df)
                parent_matches = []

            input_entity_rating = None
            if input_entity_rating_in_df:
                input_entity_rating = row[RatingColumnNames.entity_rating]

            input_opponent_rating = None
            if input_opponent_rating_in_df:
                input_opponent_rating = row[RatingColumnNames.opponent_rating]

            participation_weight = 1.0
            if self.config_column_names.participation_weight is not None and participation_weight_in_df:
                participation_weight = row[self.config_column_names.participation_weight]

            projected_participation_weight = None
            if self.config_column_names.projected_participation_weight is not None and projected_participation_weight_in_df:
                projected_participation_weight = row[self.config_column_names.projected_participation_weight]

            team_players_percentage_playing_time: Dict[str, float] = {}
            if self.config_column_names.team_players_percentage_playing_time is not None \
                    and isinstance(row[self.config_column_names.team_players_percentage_playing_time],
                                   Dict) and team_players_percentage_playing_time_in_df:
                team_players_percentage_playing_time: Dict[str, float] = row[
                    self.config_column_names.team_players_percentage_playing_time]

            entity_id = row[col_names.team_id]
            if self.config_column_names.player_id is not None and player_id_in_df:
                entity_id = row[self.config_column_names.player_id]

            """
            Build feature to allow ratings to be used from X dataframes
            """

            league = None
            if league_in_df:
                league = row[col_names.league]

            match_entity = MatchPlayer(
                entity_id=entity_id,
                team_id=row[col_names.team_id],
                league=league,
                match_ratings={},
            )

            if self.config_column_names.league in row:
                match.league = row[self.config_column_names.league]

            if row[col_names.team_id] not in match.team_ids:
                match.team_ids.append(row[col_names.team_id])

            if self.predicted_performance_method in (PredictedRatingMethod.DEFAULT, PredictedRatingMethod.MEAN_RATING):
                default_performance = row[col_names.default_performance]

                match_entity.match_performance_rating[RatingType.DEFAULT] = MatchPerformanceRating(
                    match_performance=default_performance,
                    participation_weight=participation_weight,
                    projected_participation_weight=projected_participation_weight,
                    team_players_ratio_playing_time=team_players_percentage_playing_time,
                )

            elif self.predicted_performance_method == PredictedRatingMethod.OFFENSE_VS_DEFENSE:
                offense_performance = row[col_names.offense_performance]
                defense_performance = row[col_names.defense_performance]
                match_entity.match_performance_rating[RatingType.OFFENSE] = MatchPerformanceRating(
                    match_performance=offense_performance,
                    participation_weight=participation_weight,
                    projected_participation_weight=projected_participation_weight,
                    team_players_ratio_playing_time=team_players_percentage_playing_time,
                )
                match_entity.match_performance_rating[RatingType.DEFENSE] = MatchPerformanceRating(
                    match_performance=defense_performance,
                    participation_weight=participation_weight,
                    projected_participation_weight=projected_participation_weight,
                    team_players_ratio_playing_time=team_players_percentage_playing_time,
                )

            match.entities.append(match_entity)
            prev_match_id = match_id
            prev_parent_match_id = parent_match_id

        if match is not None:
            parent_matches.append(match)
            self._update_matches(parent_matches, projected_participation_weight_in_df)

    def _update_matches(self, matches: List[Match], calculate_proj_participation_weight: bool):

        match_id_to_bugged = {}
        for match_index, match in enumerate(matches):
            match_id = match.match_id
            match_id_to_bugged[match_id] = False
            try:
                matches[match_index] = self.match_generator.generate(match, calculate_proj_participation_weight)
            except ValueError:
                match_id_to_bugged[match_id] = True

            self.match_id_to_out_df_column_values[match_id] = MatchOutValues(
                match_id=match_id,
                RatingValues={

                },
                indexes=[]
            )

        try:
            self.match_generator.update_ratings_for_matches(matches)
        except ValueError:
            for match_id in matches:
                match_id_to_bugged[match_id] = True

        for match in matches:
            match_id = match.match_id
            for match_entity in match.entities:

                if self.predicted_performance_method in (
                        PredictedRatingMethod.DEFAULT, PredictedRatingMethod.MEAN_RATING):

                    self.update_out_with_entity_rating(
                        out_column_name=RatingColumnNames.entity_rating,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFAULT
                    )

                    self.update_out_with_projected_entity_rating_difference(
                        out_column_name=RatingColumnNames.entity_rating_difference_projected,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFAULT
                    )

                    self.update_out_with_entity_rating_difference(
                        out_column_name=RatingColumnNames.entity_rating_difference,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFAULT
                    )

                    if self.predicted_performance_method == PredictedRatingMethod.MEAN_RATING:
                        self.update_out_with_rating_mean(
                            out_column_name=RatingColumnNames.rating_mean,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFAULT
                        )
                    else:
                        self.update_out_with_rating_difference(
                            out_column_name=RatingColumnNames.rating_difference,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFAULT
                        )

                    if self.config_column_names.projected_participation_weight is not None:
                        self.update_out_with_team_rating(
                            out_column_name=RatingColumnNames.team_rating,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFAULT
                        )

                        self.update_out_with_projected_team_rating(
                            out_column_name=RatingColumnNames.team_rating_projected,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFAULT
                        )

                        self.update_out_with_projected_opponent_rating(
                            out_column_name=RatingColumnNames.opponent_rating_projected,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFAULT
                        )

                        if self.predicted_performance_method == PredictedRatingMethod.MEAN_RATING:
                            self.update_out_with_projected_rating_mean(
                                out_column_name=RatingColumnNames.rating_mean_projected,
                                match_id=match_id,
                                match_entity=match_entity,
                                match_data_is_bugged=match_id_to_bugged[match_id],
                                rating_type=RatingType.DEFAULT
                            )
                        else:
                            self.update_out_with_projected_rating_difference(
                                out_column_name=RatingColumnNames.rating_difference_projected,
                                match_id=match_id,
                                match_entity=match_entity,
                                match_data_is_bugged=match_id_to_bugged[match_id],
                                rating_type=RatingType.DEFAULT
                            )

                    self.update_out_with_opponent_rating(
                        out_column_name=RatingColumnNames.opponent_rating,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFAULT
                    )


                elif self.predicted_performance_method == PredictedRatingMethod.OFFENSE_VS_DEFENSE:

                    if self.average_teams_together_for_game:
                        self.update_out_with_rating_difference_averaged_for_game(
                            out_column_name=RatingColumnNames.offense_game_rating,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.OFFENSE
                        )
                        self.update_out_with_rating_difference_averaged_for_game(
                            out_column_name=RatingColumnNames.defense_game_rating,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFENSE
                        )

                    self.update_out_with_entity_rating(
                        out_column_name=RatingColumnNames.offense_entity_rating,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.OFFENSE
                    )

                    self.update_out_with_entity_rating(
                        out_column_name=RatingColumnNames.defense_entity_rating,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFENSE
                    )

                    self.update_out_with_projected_entity_rating_difference(
                        out_column_name=RatingColumnNames.offense_entity_rating_difference_projected,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.OFFENSE
                    )

                    self.update_out_with_projected_entity_rating_difference(
                        out_column_name=RatingColumnNames.defense_entity_rating_difference_projected,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFENSE
                    )

                    self.update_out_with_entity_rating_difference(
                        out_column_name=RatingColumnNames.offense_entity_rating_difference,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.OFFENSE
                    )

                    self.update_out_with_entity_rating_difference(
                        out_column_name=RatingColumnNames.defense_entity_rating_difference,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFENSE
                    )

                    self.update_out_with_rating_difference(
                        out_column_name=RatingColumnNames.offense_rating_difference,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.OFFENSE
                    )

                    self.update_out_with_rating_difference(
                        out_column_name=RatingColumnNames.defense_rating_difference,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFENSE
                    )

                    if self.config_column_names.projected_participation_weight is not None:

                        if self.average_teams_together_for_game:
                            self.update_out_with_projected_rating_difference_averaged_for_game(
                                out_column_name=RatingColumnNames.offense_game_rating_projected,
                                match_id=match_id,
                                match_entity=match_entity,
                                match_data_is_bugged=match_id_to_bugged[match_id],
                                rating_type=RatingType.OFFENSE
                            )
                            self.update_out_with_projected_rating_difference_averaged_for_game(
                                out_column_name=RatingColumnNames.defense_game_rating_projected,
                                match_id=match_id,
                                match_entity=match_entity,
                                match_data_is_bugged=match_id_to_bugged[match_id],
                                rating_type=RatingType.DEFENSE
                            )

                        self.update_out_with_team_rating(
                            out_column_name=RatingColumnNames.offense_team_rating,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.OFFENSE
                        )

                        self.update_out_with_team_rating(
                            out_column_name=RatingColumnNames.defense_team_rating,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFENSE
                        )

                        self.update_out_with_projected_team_rating(
                            out_column_name=RatingColumnNames.offense_team_rating_projected,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.OFFENSE
                        )

                        self.update_out_with_projected_opponent_rating(
                            out_column_name=RatingColumnNames.offense_opponent_rating_projected,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.OFFENSE
                        )

                        self.update_out_with_projected_team_rating(
                            out_column_name=RatingColumnNames.defense_team_rating_projected,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFENSE
                        )

                        self.update_out_with_projected_opponent_rating(
                            out_column_name=RatingColumnNames.defense_opponent_rating_projected,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFENSE
                        )

                        self.update_out_with_projected_rating_difference(
                            out_column_name=RatingColumnNames.offense_rating_difference_projected,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.OFFENSE
                        )

                        self.update_out_with_projected_rating_difference(
                            out_column_name=RatingColumnNames.defense_rating_difference_projected,
                            match_id=match_id,
                            match_entity=match_entity,
                            match_data_is_bugged=match_id_to_bugged[match_id],
                            rating_type=RatingType.DEFENSE
                        )

                    self.update_out_with_opponent_rating(
                        out_column_name=RatingColumnNames.offense_opponent_rating,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.OFFENSE
                    )

                    self.update_out_with_opponent_rating(
                        out_column_name=RatingColumnNames.defense_opponent_rating,
                        match_id=match_id,
                        match_entity=match_entity,
                        match_data_is_bugged=match_id_to_bugged[match_id],
                        rating_type=RatingType.DEFENSE
                    )

    def update_out_with_projected_entity_rating_difference(self,
                                                           out_column_name: RatingColumnNames,
                                                           match_id: str,
                                                           match_entity: MatchPlayer,
                                                           match_data_is_bugged: bool,
                                                           rating_type: RatingType
                                                           ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return
        if match_entity.match_performance_rating[rating_type].rating.pre_match_projected_opponent_rating is None:
            return

        rating_difference = match_entity.match_performance_rating[rating_type].rating.pre_match_entity_rating - \
                            match_entity.match_performance_rating[
                                rating_type].rating.pre_match_projected_opponent_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating_difference)

    def update_out_with_entity_rating_difference(self,
                                                 out_column_name: RatingColumnNames,
                                                 match_id: str,
                                                 match_entity: MatchPlayer,
                                                 match_data_is_bugged: bool,
                                                 rating_type: RatingType
                                                 ):
        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        if match_entity.match_performance_rating[rating_type].rating.pre_match_opponent_rating is None:
            return

        rating_difference = match_entity.match_performance_rating[rating_type].rating.pre_match_entity_rating - \
                            match_entity.match_performance_rating[rating_type].rating.pre_match_opponent_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating_difference)

    def update_out_with_rating_difference(self,
                                          out_column_name: RatingColumnNames,
                                          match_id: str,
                                          match_entity: MatchPlayer,
                                          match_data_is_bugged: bool,
                                          rating_type: RatingType
                                          ):
        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        if match_entity.match_performance_rating[rating_type].rating.pre_match_team_rating is None:
            return

        rating_difference = match_entity.match_performance_rating[rating_type].rating.pre_match_team_rating - \
                            match_entity.match_performance_rating[rating_type].rating.pre_match_opponent_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating_difference)

    def update_out_with_rating_mean(self,
                                    out_column_name: RatingColumnNames,
                                    match_id: str,
                                    match_entity: MatchPlayer,
                                    match_data_is_bugged: bool,
                                    rating_type: RatingType
                                    ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        if match_entity.match_performance_rating[rating_type].rating.pre_match_team_rating is None:
            return

        rating_mean = match_entity.match_performance_rating[rating_type].rating.pre_match_team_rating * 0.5 + \
                      match_entity.match_performance_rating[rating_type].rating.pre_match_opponent_rating * 0.5

        rating_mean = rating_mean - self.match_generator.match_rating_calculator.average_rating + 1000

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating_mean)

    def update_out_with_projected_rating_mean(self,
                                              out_column_name: RatingColumnNames,
                                              match_id: str,
                                              match_entity: MatchPlayer,
                                              match_data_is_bugged: bool,
                                              rating_type: RatingType
                                              ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        value = match_entity.match_performance_rating[rating_type].rating.pre_match_projected_team_rating * 0.5 + \
                match_entity.match_performance_rating[rating_type].rating.pre_match_projected_opponent_rating * 0.5

        if self.target_rating is not None and self.match_generator.match_rating_calculator.average_rating is not None:
            value = value - self.match_generator.match_rating_calculator.average_rating + self.target_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(value)

    def update_out_with_projected_opponent_rating(self,
                                                  out_column_name: RatingColumnNames,
                                                  match_id: str,
                                                  match_entity: MatchPlayer,
                                                  match_data_is_bugged: bool,
                                                  rating_type: RatingType
                                                  ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        value = match_entity.match_performance_rating[rating_type].rating.pre_match_projected_opponent_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(value)

    def update_out_with_projected_team_rating(self,
                                              out_column_name: RatingColumnNames,
                                              match_id: str,
                                              match_entity: MatchPlayer,
                                              match_data_is_bugged: bool,
                                              rating_type: RatingType
                                              ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        value = match_entity.match_performance_rating[rating_type].rating.pre_match_projected_team_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(value)

    def update_out_with_rating_difference_averaged_for_game(self,
                                                            out_column_name: RatingColumnNames,
                                                            match_id: str,
                                                            match_entity: MatchPlayer,
                                                            match_data_is_bugged: bool,
                                                            rating_type: RatingType
                                                            ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        rating_same_type = match_entity.match_performance_rating[rating_type].rating.pre_match_team_rating * 0.5 + \
                           match_entity.match_performance_rating[
                               rating_type].rating.pre_match_opponent_rating_same_type * 0.5

        other_rating_type = match_entity.match_performance_rating[
                                rating_type].rating.pre_match_opponent_rating * 0.5 + 0.5 * \
                            match_entity.match_performance_rating[rating_type].rating.pre_match_team_other_rating_type

        rating_difference = rating_same_type - other_rating_type

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating_difference)

    def update_out_with_projected_rating_difference_averaged_for_game(self,
                                                                      out_column_name: RatingColumnNames,
                                                                      match_id: str,
                                                                      match_entity: MatchPlayer,
                                                                      match_data_is_bugged: bool,
                                                                      rating_type: RatingType
                                                                      ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        rating_same_type = match_entity.match_performance_rating[
                               rating_type].rating.pre_match_projected_team_rating * 0.5 + \
                           match_entity.match_performance_rating[
                               rating_type].rating.pre_match_projected_opponent_rating_same_rating_type * 0.5

        other_rating_type = match_entity.match_performance_rating[
                                rating_type].rating.pre_match_projected_opponent_rating * 0.5 + 0.5 * \
                            match_entity.match_performance_rating[
                                rating_type].rating.pre_match_projected_team_other_rating_type

        rating_difference = rating_same_type - other_rating_type

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating_difference)

    def update_out_with_projected_rating_difference(self,
                                                    out_column_name: RatingColumnNames,
                                                    match_id: str,
                                                    match_entity: MatchPlayer,
                                                    match_data_is_bugged: bool,
                                                    rating_type: RatingType
                                                    ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        rating_difference = match_entity.match_performance_rating[rating_type].rating.pre_match_projected_team_rating - \
                            match_entity.match_performance_rating[
                                rating_type].rating.pre_match_projected_opponent_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating_difference)

    def update_out_with_opponent_rating(self,
                                        out_column_name: RatingColumnNames,
                                        match_id: str,
                                        match_entity: MatchPlayer,
                                        match_data_is_bugged: bool,
                                        rating_type: RatingType
                                        ):
        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        if match_entity.match_performance_rating[rating_type].rating.pre_match_opponent_rating is None:
            return

        opponent_rating = match_entity.match_performance_rating[rating_type].rating.pre_match_opponent_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(opponent_rating)

    def update_out_with_team_rating(self,
                                    out_column_name: RatingColumnNames,
                                    match_id: str,
                                    match_entity: MatchPlayer,
                                    match_data_is_bugged: bool,
                                    rating_type: RatingType
                                    ):
        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        if match_entity.match_performance_rating[rating_type].rating.pre_match_team_rating is None:
            return

        rating = match_entity.match_performance_rating[rating_type].rating.pre_match_team_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating)

    def update_out_with_entity_rating(self,
                                      out_column_name: RatingColumnNames,
                                      match_id: str,
                                      match_entity: MatchPlayer,
                                      match_data_is_bugged: bool,
                                      rating_type: RatingType
                                      ):

        if out_column_name not in self.match_id_to_out_df_column_values[
            match_id].RatingValues:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name] = []

        if match_data_is_bugged:
            self.match_id_to_out_df_column_values[match_id].RatingValues[
                out_column_name].append(None)
            return

        rating = match_entity.match_performance_rating[rating_type].rating.pre_match_entity_rating

        self.match_id_to_out_df_column_values[match_id].RatingValues[
            out_column_name].append(rating)

    def _validate_sorting(self, X: pd.DataFrame) -> bool:
        max_game_id_checks = 10
        prev_row_date = pd.to_datetime("1970-01-01 00:00:00")

        prev_team_id = ""
        prev_match_id = ""
        match_id_to_team_ids = {}
        X[self.config_column_names.start_date_time] = pd.to_datetime(X[self.config_column_names.start_date_time],
                                                                     format='%Y-%m-%d %H:%M:%S')
        for index, row in X.iterrows():
            try:
                if row[self.config_column_names.start_date_time] < prev_row_date:
                    return False
            except TypeError:

                prev_row_date = prev_row_date.tz_localize('CET')
                if row[self.config_column_names.start_date_time] < prev_row_date:
                    return False

            match_id = row[self.config_column_names.match_id]

            if match_id != prev_match_id and match_id in match_id_to_team_ids:
                return False

            if match_id not in match_id_to_team_ids:
                match_id_to_team_ids[match_id] = []

            team_id = row[self.config_column_names.team_id]
            if team_id != prev_team_id and team_id in match_id_to_team_ids[match_id]:
                return False

            if team_id not in match_id_to_team_ids[match_id]:
                match_id_to_team_ids[match_id].append(team_id)

            if len(match_id_to_team_ids) == max_game_id_checks:
                return True

            prev_row_date = row[self.config_column_names.start_date_time]
            prev_match_id = match_id
            prev_team_id = team_id

        return True
