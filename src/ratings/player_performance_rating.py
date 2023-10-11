from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

from src.ratings.data_structures import ColumnNames, Match
from src.ratings.enums import RatingColumnNames, InputColumnNames
from src.ratings.performance.performance_generator import BasePerformanceGenerator
from src.ratings.rating_generator import RatingGenerator

PERFORMANCE = "performance"


@dataclass
class Config:
    league: bool
    hardcoded_entity_rating: bool


def _get_config(df: pd.DataFrame, column_names: ColumnNames) -> Config:
    league_in_df = False
    if column_names.league in df.columns.tolist():
        league_in_df = True

    input_entity_rating_in_df = False
    if RatingColumnNames.entity_rating in df.columns.tolist():
        input_entity_rating_in_df = True

    input_opponent_rating_in_df = False
    if RatingColumnNames.opponent_rating in df.columns.tolist():
        input_opponent_rating_in_df = True

    participation_weight_in_df = False
    if column_names.participation_weight in df.columns.tolist():
        participation_weight_in_df = True

    projected_participation_weight_in_df = False
    if column_names.projected_participation_weight in df.columns.tolist():
        projected_participation_weight_in_df = True

    team_players_percentage_playing_time_in_df = False
    if column_names.team_players_percentage_playing_time in df.columns.tolist():
        team_players_percentage_playing_time_in_df = True

    player_id_in_df = False
    if column_names.player_id in df.columns.tolist():
        player_id_in_df = True

    use_parent_match_id = False
    if column_names.parent_match_id is not None and col_names.parent_match_id in df.columns:
        use_parent_match_id = True


def generate_matches(df: pd.DataFrame, column_names: ColumnNames) -> list[Match]:
    sorted_correctly = _validate_sorting(df)
    if not sorted_correctly:
        raise ValueError("X needs to be sorted by date, game_id, team_id in ascending order")

    df[column_names.start_date_time] = pd.to_datetime(df[column_names.start_date_time], format='%Y-%m-%d %H:%M:%S')
    try:
        date_time = df[column_names.start_date_time].dt.tz_convert('UTC')
    except TypeError:
        date_time = df[column_names.start_date_time].dt.tz_localize('UTC')
    df[InputColumnNames.hour] = (date_time - pd.Timestamp("1970-01-01").tz_localize('UTC')) // pd.Timedelta(
        '1h')

    config = _get_config(df=df, column_names=column_names)

    prev_match_id = None
    prev_parent_match_id = None
    match = None

    data_dict = df.to_dict('records')

    parent_matches = []
    for row in data_dict:
        match_id = row[col_names.id]

        parent_match_id = None
        if use_parent_match_id:
            parent_match_id = row[col_names.parent_match_id]

        if match_id in self.match_id_to_out_df_column_values:
            continue

        if match_id != prev_match_id:

            if prev_match_id is not None:
                parent_matches.append(match)

            match = Match(
                match_id=row[col_names.id],
                teams=[],
                team_ids=[],
                day_number=int(row[InputColumnNames.hour] / 24),
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
        if self.column_names.participation_weight is not None and participation_weight_in_df:
            participation_weight = row[self.column_names.participation_weight]

        projected_participation_weight = None
        if self.column_names.projected_participation_weight is not None and projected_participation_weight_in_df:
            projected_participation_weight = row[self.column_names.projected_participation_weight]

        team_players_percentage_playing_time: Dict[str, float] = {}
        if self.column_names.team_players_percentage_playing_time is not None \
                and isinstance(row[self.column_names.team_players_percentage_playing_time],
                               Dict) and team_players_percentage_playing_time_in_df:
            team_players_percentage_playing_time: Dict[str, float] = row[
                self.column_names.team_players_percentage_playing_time]

        entity_id = row[col_names.team_id]
        if self.column_names.player_id is not None and player_id_in_df:
            entity_id = row[self.column_names.player_id]

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

        if self.column_names.league in row:
            match.league = row[self.column_names.league]

        if row[col_names.team_id] not in match.team_ids:
            match.team_ids.append(row[col_names.team_id])

        if self.predicted_performance_method in (PredictedRatingMethod.DEFAULT, PredictedRatingMethod.MEAN_RATING):
            default_performance = row[col_names.default_performance]

            match_entity.match_player_performance[RatingType.DEFAULT] = MatchPerformanceRating(
                match_performance=default_performance,
                participation_weight=participation_weight,
                projected_participation_weight=projected_participation_weight,
                team_players_ratio_playing_time=team_players_percentage_playing_time,
            )

        elif self.predicted_performance_method == PredictedRatingMethod.OFFENSE_VS_DEFENSE:
            offense_performance = row[col_names.offense_performance]
            defense_performance = row[col_names.defense_performance]
            match_entity.match_player_performance[RatingType.OFFENSE] = MatchPerformanceRating(
                match_performance=offense_performance,
                participation_weight=participation_weight,
                projected_participation_weight=projected_participation_weight,
                team_players_ratio_playing_time=team_players_percentage_playing_time,
            )
            match_entity.match_player_performance[RatingType.DEFENSE] = MatchPerformanceRating(
                match_performance=defense_performance,
                participation_weight=participation_weight,
                projected_participation_weight=projected_participation_weight,
                team_players_ratio_playing_time=team_players_percentage_playing_time,
            )

        match.entities.append(match_entity)
        prev_match_id = match_id
        prev_parent_match_id = parent_match_id


class PlayerRatingGenerator():

    def __init__(self,
                 column_names: ColumnNames,
                 performance_generator: Optional[BasePerformanceGenerator],
                 rating_generator: RatingGenerator
                 ):
        self.column_names = column_names
        self.performance_generator = performance_generator
        self.rating_generator = rating_generator

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.performance_generator:
            df = self.performance_generator.generate(df)

        if PERFORMANCE not in df.columns:
            raise ValueError("performance column not created")

        matches = generate_matches(column_names=self.column_names, df=df)

        match_ratings = self.rating_generator.generate(matches)
        df[RatingColumnNames.entity_rating] = match_ratings.pre_match_player_rating_values
        df[RatingColumnNames.team_rating] = match_ratings.pre_match_team_rating_values
        df[RatingColumnNames.opponent_rating] = match_ratings.pre_match_opponent_rating_values
        return df

    @property
    def player_ratings(self):
        return

    @property
    def team_ratings(self):
        return
