from __future__ import annotations

from typing import Dict

import pandas as pd

from src.ratings.data_structures import MatchTeam, ColumnNames

from src.ratings.data_structures import MatchPerformance, \
    MatchPlayer, Match

HOUR_NUMBER_COLUMN_NAME = "hour_number"


def _validate_sorting(X: pd.DataFrame, config_column_names) -> bool:
    max_game_id_checks = 10
    prev_row_date = pd.to_datetime("1970-01-01 00:00:00")

    prev_team_id = ""
    prev_match_id = ""
    match_id_to_team_ids = {}
    X[config_column_names.start_date] = pd.to_datetime(X[config_column_names.start_date],
                                                       format='%Y-%m-%d %H:%M:%S')
    for index, row in X.iterrows():
        try:
            if row[config_column_names.start_date] < prev_row_date:
                return False
        except TypeError:

            prev_row_date = prev_row_date.tz_localize('CET')
            if row[config_column_names.start_date] < prev_row_date:
                return False

        match_id = row[config_column_names.match_id]

        if match_id != prev_match_id and match_id in match_id_to_team_ids:
            return False

        if match_id not in match_id_to_team_ids:
            match_id_to_team_ids[match_id] = []

        team_id = row[config_column_names.team_id]
        if team_id != prev_team_id and team_id in match_id_to_team_ids[match_id]:
            return False

        if team_id not in match_id_to_team_ids[match_id]:
            match_id_to_team_ids[match_id].append(team_id)

        if len(match_id_to_team_ids) == max_game_id_checks:
            return True

        prev_row_date = row[config_column_names.start_date]
        prev_match_id = match_id
        prev_team_id = team_id

    return True


def get_matches_from_df(df: pd.DataFrame, column_names: ColumnNames) -> list[Match]:
    sorted_correctly = _validate_sorting(df, column_names)
    if not sorted_correctly:
        raise ValueError("X needs to be sorted by date, game_id, team_id in ascending order")

    col_names = column_names
    df[col_names.start_date] = pd.to_datetime(df[col_names.start_date], format='%Y-%m-%d %H:%M:%S')
    try:
        date_time = df[col_names.start_date].dt.tz_convert('UTC')
    except TypeError:
        date_time = df[col_names.start_date].dt.tz_localize('UTC')
    df[HOUR_NUMBER_COLUMN_NAME] = (date_time - pd.Timestamp("1970-01-01").tz_localize('UTC')) // pd.Timedelta(
        '1h')

    league_in_df = False
    if column_names.league in df.columns.tolist():
        league_in_df = True

    participation_weight_in_df = False
    if column_names.participation_weight in df.columns.tolist():
        participation_weight_in_df = True

    projected_participation_weight_in_df = False
    if column_names.projected_participation_weight in df.columns.tolist():
        projected_participation_weight_in_df = True

    team_players_percentage_playing_time_in_df = False
    if column_names.team_players_percentage_playing_time in df.columns.tolist():
        team_players_percentage_playing_time_in_df = True

    prev_match_id = None
    match = None

    data_dict = df.to_dict('records')

    matches = []
    teams = []
    for row in data_dict:
        match_id = row[col_names.match_id]

        parent_match_id = None

        if match_id != prev_match_id:
            teams = []
            if prev_match_id is not None:
                matches.append(match)

            match = Match(
                id=row[col_names.match_id],
                teams=teams,
                day_number=int(row[HOUR_NUMBER_COLUMN_NAME] / 24),
            )

        participation_weight = 1.0
        if column_names.participation_weight is not None and participation_weight_in_df:
            participation_weight = row[column_names.participation_weight]

        projected_participation_weight = 1
        if column_names.projected_participation_weight is not None and projected_participation_weight_in_df:
            projected_participation_weight = row[column_names.projected_participation_weight]

        team_players_percentage_playing_time: Dict[str, float] = {}
        if column_names.team_players_percentage_playing_time is not None \
                and isinstance(row[column_names.team_players_percentage_playing_time],
                               Dict) and team_players_percentage_playing_time_in_df:
            team_players_percentage_playing_time: Dict[str, float] = row[
                column_names.team_players_percentage_playing_time]

        team_id = row[col_names.team_id]

        if team_id not in [t.id for t in teams]:
            match_team = MatchTeam(
                id=team_id,
                players=[]
            )
            teams.append(match_team)

        player_id = row[col_names.team_id]
        if column_names.player_id is not None:
            player_id = row[column_names.player_id]

        league = None
        if league_in_df:
            league = row[col_names.league]

        performance = MatchPerformance(
            performance_value=row[column_names.performance],
            participation_weight=participation_weight,
            projected_participation_weight=projected_participation_weight,
            ratio=team_players_percentage_playing_time,
        )

        match_player = MatchPlayer(
            id=player_id,
            league=league,
            performance=performance,
        )

        if column_names.league in row:
            match.league = row[column_names.league]

        teams[len(teams) - 1].players.append(match_player)

        prev_match_id = match_id

    if match is not None:
        matches.append(match)

    return matches
