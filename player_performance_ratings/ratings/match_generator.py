from __future__ import annotations

import json
import logging
import warnings

from player_performance_ratings.utils import validate_sorting

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Optional
import pandas as pd

from player_performance_ratings.data_structures import (
    MatchTeam,
    ColumnNames,
    MatchPerformance,
    MatchPlayer,
    Match,
)
from player_performance_ratings.ratings.league_identifier import LeagueIdentifier

HOUR_NUMBER_COLUMN_NAME = "hour_number"


def convert_df_to_matches(
    df: pd.DataFrame,
    column_names: ColumnNames,
    performance_column_name: str,
    separate_player_by_position: bool = False,
    league_identifier: Optional[LeagueIdentifier] = LeagueIdentifier(),
) -> list[Match]:
    """
    Converts a dataframe to a list of matches.
    Each dataframe row needs to be a unique combination of match_id and player_id.
    The dataframe needs to contain the following columns:
        * player_id
        * team_id
        * match_id
        * start_date
        * performance


    Dataframe needs to be sorted by date, game_id, team_id in ascending order.

    Optionally a column for participation_weight and league can be passed.
    The participation_weight indicates the percentage (as a ratio) of the time played by the player in the match.
    The league column indicates the league of the match.
    If the league_identifier is passed it will identify the league of the match by the players past matches played.
    If not the  league of the match will be equal to the league of the current match
    """

    df = df.copy()

    if (
        column_names.participation_weight is None
        and column_names.projected_participation_weight is not None
    ):
        raise ValueError(
            "projected_participation_weight column passed but not participation_weight column"
        )
    if performance_column_name in df.columns:
        if (
            max(df[performance_column_name]) > 1.001
            or min(df[performance_column_name]) < -0.001
        ):
            raise ValueError("performance column must be between 0 and 1")

        mean_performance = df[performance_column_name].mean()

        if abs(mean_performance - 0.5) > 0.05:
            logging.warning(
                f"mean performance is {mean_performance} which is far from 0.5. It is recommended to do further pre_transformations of the performance column"
            )

        if df[performance_column_name].isnull().any():
            logging.error(
                f"df[{performance_column_name}] contains nan values. Make sure all column_names used in column_weights are imputed beforehand"
            )
            raise ValueError("performance contains nan values")

    validate_sorting(df=df, column_names=column_names)

    col_names = column_names
    try:
        df[col_names.start_date] = pd.to_datetime(
            df[col_names.start_date], format="%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        df[col_names.start_date] = pd.to_datetime(
            df[col_names.start_date], format="%Y-%m-%d"
        )
    try:
        date_time = df[col_names.start_date].dt.tz_convert("UTC")
    except TypeError:
        date_time = df[col_names.start_date].dt.tz_localize("UTC")
    df[HOUR_NUMBER_COLUMN_NAME] = (
        date_time - pd.Timestamp("1970-01-01").tz_localize("UTC")
    ) // pd.Timedelta("1h")

    if col_names.league and col_names.league not in df.columns:
        raise ValueError("league column passed but not in dataframe.")

    league_in_df = False
    if col_names.league is not None:
        league_in_df = True

    if col_names.projected_participation_weight:
        if col_names.projected_participation_weight not in df.columns:
            logging.warning(
                "projected_participation_weight column passed but not in dataframe."
                " Will use participation_weight as projected_participation_weight"
            )

    if col_names.team_players_playing_time:
        if col_names.team_players_playing_time not in df.columns:
            raise ValueError(
                "team_players_playing_time column passed but not in dataframe."
            )

        is_team_players_playing_time = True
    else:
        is_team_players_playing_time = False

    if col_names.opponent_players_playing_time:
        if col_names.opponent_players_playing_time not in df.columns:
            raise ValueError(
                "opponent_players_playing_time column passed but not in dataframe."
            )

        is_opponent_players_playing_time = True
    else:
        is_opponent_players_playing_time = False

    prev_match_id = None
    prev_update_team_id = None

    if len(df.columns) > 60:
        logging.warning(
            f"Dataframe column count {len(df.columns)} is high. Consider removing unneeded columns to speed up processing itme"
        )
    data_dict = df.to_dict("records")

    matches = []

    prev_team_id = None
    prev_row: Optional[None, pd.Series] = None
    match_teams = []
    match_team_players = []
    team_league_counts = {}

    for row in data_dict:
        match_id = row[col_names.match_id]
        team_id = row[col_names.team_id]
        update_team_id = row[col_names.parent_team_id]
        if (
            team_id != prev_team_id
            and prev_team_id != None
            or prev_match_id != match_id
            and prev_match_id != None
        ):
            match_team = _create_match_team(
                team_league_counts=team_league_counts,
                team_id=prev_team_id,
                match_team_players=match_team_players,
                update_team_id=prev_update_team_id,
            )
            match_teams.append(match_team)
            match_team_players = []
            team_league_counts = {}

        if match_id != prev_match_id and prev_match_id != None:
            match = _create_match(
                league_in_df=league_in_df,
                row=prev_row,
                match_teams=match_teams,
                column_names=column_names,
            )
            matches.append(match)
            match_teams = []

        participation_weight = 1.0
        if col_names.participation_weight and col_names.participation_weight in row:
            participation_weight = row[col_names.participation_weight]

        player_id = row[col_names.team_id]
        if col_names.player_id is not None:
            player_id = row[col_names.player_id]

        if col_names.position is not None:
            position = row[col_names.position]
            if separate_player_by_position:
                player_id = f"{player_id}_{position}"

        else:
            position = None

        if league_in_df:
            match_league = row[column_names.league]
            if team_id not in team_league_counts:
                team_league_counts[team_id] = {}
            player_league = league_identifier.identify(
                player_id=player_id, league_match=match_league
            )
            if player_league not in team_league_counts[team_id]:
                team_league_counts[team_id][player_league] = 0

            team_league_counts[team_id][player_league] += 1
        else:
            player_league = None

        if is_team_players_playing_time:
            team_players_playing_time = row[col_names.team_players_playing_time]
            try:
                team_players_playing_time = json.loads(team_players_playing_time)
            except Exception:
                pass
        else:
            team_players_playing_time = {}

        if is_opponent_players_playing_time:
            opponent_players_playing_time = row[col_names.opponent_players_playing_time]
            try:
                opponent_players_playing_time = json.loads(
                    opponent_players_playing_time
                )
            except Exception:
                pass
        else:
            opponent_players_playing_time = {}

        if (
            col_names.projected_participation_weight
            and col_names.participation_weight in row
        ):
            projected_participation_weight = row[
                col_names.projected_participation_weight
            ]
        elif col_names.participation_weight not in row:
            projected_participation_weight = 1
        else:
            projected_participation_weight = participation_weight

        if performance_column_name in row:
            performance = MatchPerformance(
                team_players_playing_time=team_players_playing_time,
                opponent_players_playing_time=opponent_players_playing_time,
                performance_value=row[performance_column_name],
                participation_weight=participation_weight,
                projected_participation_weight=projected_participation_weight,
            )

        else:
            performance = MatchPerformance(
                performance_value=None,
                participation_weight=None,
                projected_participation_weight=participation_weight,
            )

        others = {}
        if col_names.other_values:
            for other_col in col_names.other_values:
                others[other_col] = row[other_col]

        match_player = MatchPlayer(
            id=player_id,
            league=player_league,
            performance=performance,
            position=position,
            others=others,
        )
        match_team_players.append(match_player)

        prev_match_id = match_id
        prev_team_id = team_id
        prev_update_team_id = update_team_id
        prev_row = row

    match_team = _create_match_team(
        team_league_counts=team_league_counts,
        team_id=prev_team_id,
        update_team_id=prev_update_team_id,
        match_team_players=match_team_players,
    )
    match_teams.append(match_team)
    match = _create_match(
        league_in_df=league_in_df,
        row=df.iloc[len(df) - 1],
        match_teams=match_teams,
        column_names=column_names,
    )
    matches.append(match)

    return matches


def _create_match(
    league_in_df,
    row: pd.Series,
    match_teams: list[MatchTeam],
    column_names: ColumnNames,
) -> Match:
    match_id = row[column_names.match_id]
    if league_in_df:
        match_league = row[column_names.league]
    else:
        match_league = None

    return Match(
        id=match_id,
        teams=match_teams,
        day_number=int(row[HOUR_NUMBER_COLUMN_NAME] / 24),
        league=match_league,
        update_id=row[column_names.update_match_id],
    )


def _create_match_team(
    team_league_counts: dict,
    team_id: str,
    update_team_id: str,
    match_team_players: list[MatchPlayer],
) -> MatchTeam:
    if team_league_counts:
        team_league = max(
            team_league_counts[team_id], key=team_league_counts[team_id].get
        )
    else:
        team_league = None
    return MatchTeam(
        id=team_id,
        update_id=update_team_id,
        players=match_team_players,
        league=team_league,
    )
