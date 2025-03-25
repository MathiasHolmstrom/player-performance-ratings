import pandas as pd

from spforge import ColumnNames
from spforge.data_structures import (
    Match,
    MatchTeam,
    MatchPlayer,
    MatchPerformance,
)
from spforge.ratings.league_identifier import LeagueIdentifier
from spforge.ratings.match_generator import convert_df_to_matches


def test_convert_df_to_matches():
    """
    When LeagueGenerator is passed, league is added to MatchPlayer and MatchTeam and Match.
    For MatchPlayer and MatchTeam, League should be determined based on previous match league.
    """

    df = pd.DataFrame(
        {
            "game_id": ["1", "1", "1", "1", "2", "2", "2", "2"],
            "team_id": ["1", "1", "2", "2", "1", "1", "2", "2"],
            "league": [
                "league1",
                "league1",
                "league1",
                "league1",
                "league2",
                "league2",
                "league2",
                "league2",
            ],
            "player_id": ["3", "4", "5", "6", "3", "4", "5", "6"],
            "won": [1, 1, 0, 0, 1, 1, 0, 0],
            "start_date": [
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-01"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
                pd.to_datetime("2021-01-02"),
            ],
        }
    )

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        league="league",
    )

    matches = convert_df_to_matches(
        df=df,
        column_names=column_names,
        league_identifier=LeagueIdentifier(),
        performance_column_name="won",
    )

    expected_matches = [
        Match(
            update_id="1",
            id="1",
            league="league1",
            teams=[
                MatchTeam(
                    id="1",
                    league="league1",
                    players=[
                        MatchPlayer(
                            id="3",
                            league="league1",
                            others={},
                            performance=MatchPerformance(
                                participation_weight=1,
                                performance_value=1,
                                projected_participation_weight=1,
                                opponent_players_playing_time={},
                                team_players_playing_time={},
                            ),
                        ),
                        MatchPlayer(
                            id="4",
                            league="league1",
                            others={},
                            performance=MatchPerformance(
                                participation_weight=1,
                                performance_value=1,
                                projected_participation_weight=1,
                                opponent_players_playing_time={},
                                team_players_playing_time={},
                            ),
                        ),
                    ],
                ),
                MatchTeam(
                    id="2",
                    league="league1",
                    players=[
                        MatchPlayer(
                            league="league1",
                            id="5",
                            others={},
                            performance=MatchPerformance(
                                participation_weight=1,
                                performance_value=0,
                                projected_participation_weight=1,
                                opponent_players_playing_time={},
                                team_players_playing_time={},
                            ),
                        ),
                        MatchPlayer(
                            league="league1",
                            id="6",
                            others={},
                            performance=MatchPerformance(
                                participation_weight=1,
                                performance_value=0,
                                projected_participation_weight=1,
                                opponent_players_playing_time={},
                                team_players_playing_time={},
                            ),
                        ),
                    ],
                ),
            ],
            day_number=18628,
        ),
        Match(
            update_id="2",
            id="2",
            league="league2",
            teams=[
                MatchTeam(
                    id="1",
                    league="league1",
                    players=[
                        MatchPlayer(
                            id="3",
                            league="league1",
                            others={},
                            performance=MatchPerformance(
                                participation_weight=1,
                                performance_value=1,
                                projected_participation_weight=1,
                                opponent_players_playing_time={},
                                team_players_playing_time={},
                            ),
                        ),
                        MatchPlayer(
                            id="4",
                            league="league1",
                            others={},
                            performance=MatchPerformance(
                                participation_weight=1,
                                performance_value=1,
                                projected_participation_weight=1,
                                opponent_players_playing_time={},
                                team_players_playing_time={},
                            ),
                        ),
                    ],
                ),
                MatchTeam(
                    id="2",
                    league="league1",
                    players=[
                        MatchPlayer(
                            id="5",
                            league="league1",
                            others={},
                            performance=MatchPerformance(
                                participation_weight=1,
                                performance_value=0,
                                projected_participation_weight=1,
                                opponent_players_playing_time={},
                                team_players_playing_time={},
                            ),
                        ),
                        MatchPlayer(
                            id="6",
                            league="league1",
                            others={},
                            performance=MatchPerformance(
                                participation_weight=1,
                                performance_value=0,
                                projected_participation_weight=1,
                                opponent_players_playing_time={},
                                team_players_playing_time={},
                            ),
                        ),
                    ],
                ),
            ],
            day_number=18629,
        ),
    ]

    assert matches == expected_matches
