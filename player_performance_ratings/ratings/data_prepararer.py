from __future__ import annotations

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from typing import Dict, Optional

import pandas as pd

from player_performance_ratings.data_structures import MatchTeam, ColumnNames, MatchPerformance, \
    MatchPlayer, Match

from player_performance_ratings.ratings.league_identifier import LeagueIdentifier

HOUR_NUMBER_COLUMN_NAME = "hour_number"


class MatchGenerator():

    def __init__(self, column_names: ColumnNames, league_identifier: Optional[LeagueIdentifier] = None):
        self.column_names = column_names
        self.league_identifier = league_identifier or LeagueIdentifier()

    def generate(self, df: pd.DataFrame) -> list[Match]:
        sorted_correctly = self._validate_sorting(df)
        if not sorted_correctly:
            raise ValueError("X needs to be sorted by date, game_id, team_id in ascending order")

        col_names = self.column_names
        df[col_names.start_date] = pd.to_datetime(df[col_names.start_date], format='%Y-%m-%d %H:%M:%S')
        try:
            date_time = df[col_names.start_date].dt.tz_convert('UTC')
        except TypeError:
            date_time = df[col_names.start_date].dt.tz_localize('UTC')
        df[HOUR_NUMBER_COLUMN_NAME] = (date_time - pd.Timestamp("1970-01-01").tz_localize('UTC')) // pd.Timedelta(
            '1h')

        league_in_df = False
        if col_names.league in df.columns.tolist():
            league_in_df = True

        participation_weight_in_df = False
        if col_names.participation_weight in df.columns.tolist():
            participation_weight_in_df = True

        projected_participation_weight_in_df = False
        if col_names.projected_participation_weight in df.columns.tolist():
            projected_participation_weight_in_df = True

        team_players_percentage_playing_time_in_df = False
        if col_names.team_players_percentage_playing_time in df.columns.tolist():
            team_players_percentage_playing_time_in_df = True

        prev_match_id = None

        data_dict = df.to_dict('records')

        matches = []

        prev_team_id = None
        prev_row: Optional[None, pd.Series] = None
        match_teams = []
        match_team_players = []
        team_league_counts = {}

        for row in data_dict:
            match_id = row[col_names.match_id]
            team_id = row[col_names.team_id]
            if team_id != prev_team_id and prev_team_id != None or prev_match_id != match_id and prev_match_id != None:
                match_team = self._create_match_team(team_league_counts=team_league_counts, team_id=prev_team_id,
                                                     match_team_players=match_team_players)
                match_teams.append(match_team)
                match_team_players = []
                team_league_counts = {}

            if match_id != prev_match_id and prev_match_id != None:
                match = self._create_match(league_in_df=league_in_df, row=prev_row, match_teams=match_teams)
                matches.append(match)
                match_teams = []

            participation_weight = 1.0
            if col_names.participation_weight is not None and participation_weight_in_df:
                participation_weight = row[col_names.participation_weight]

            projected_participation_weight = 1
            if col_names.projected_participation_weight is not None and projected_participation_weight_in_df:
                projected_participation_weight = row[col_names.projected_participation_weight]

            team_players_percentage_playing_time: Dict[str, float] = {}
            if col_names.team_players_percentage_playing_time is not None \
                    and isinstance(row[col_names.team_players_percentage_playing_time],
                                   Dict) and team_players_percentage_playing_time_in_df:
                team_players_percentage_playing_time: Dict[str, float] = row[
                    col_names.team_players_percentage_playing_time]

            player_id = row[col_names.team_id]
            if col_names.player_id is not None:
                player_id = row[col_names.player_id]

            if league_in_df:
                match_league = row[self.column_names.league]
                if team_id not in team_league_counts:
                    team_league_counts[team_id] = {}
                player_league = self.league_identifier.identify(player_id=player_id,
                                                                league_match=match_league)
                if player_league not in team_league_counts[team_id]:
                    team_league_counts[team_id][player_league] = 0

                team_league_counts[team_id][player_league] += 1
            else:
                player_league = None

            performance = MatchPerformance(
                performance_value=row[col_names.performance],
                participation_weight=participation_weight,
                projected_participation_weight=projected_participation_weight,
                ratio=team_players_percentage_playing_time,
            )

            match_player = MatchPlayer(
                id=player_id,
                league=player_league,
                performance=performance,
            )
            match_team_players.append(match_player)

            prev_match_id = match_id
            prev_team_id = team_id
            prev_row = row

        match_team = self._create_match_team(team_league_counts=team_league_counts, team_id=prev_team_id,
                                             match_team_players=match_team_players)
        match_teams.append(match_team)
        match = self._create_match(league_in_df=league_in_df, row=df.iloc[len(df) - 1],
                                   match_teams=match_teams)
        matches.append(match)

        return matches

    def _create_match(self, league_in_df, row: pd.Series, match_teams: list[MatchTeam]) -> Match:
        match_id = row[self.column_names.match_id]
        if league_in_df:
            match_league = row[self.column_names.league]
        else:
            match_league = None

        return Match(
            id=match_id,
            teams=match_teams,
            day_number=int(row[HOUR_NUMBER_COLUMN_NAME] / 24),
            league=match_league,
            update_id=row[self.column_names.rating_update_id]
        )

    def _create_match_team(self, team_league_counts: dict, team_id: str,
                           match_team_players: list[MatchPlayer]) -> MatchTeam:
        if team_league_counts:
            team_league = max(team_league_counts[team_id], key=team_league_counts[team_id].get)
        else:
            team_league = None
        return MatchTeam(
            id=team_id,
            players=match_team_players,
            league=team_league
        )

    def _validate_sorting(self, X: pd.DataFrame) -> bool:
        df_sorted = X.sort_values(
            by=[self.column_names.start_date,  self.column_names.match_id,
                self.column_names.team_id, self.column_names.player_id])

        return X.equals(df_sorted)
