"""
Unit tests for team_ratings and player_ratings properties.
"""

import polars as pl
import pytest

from spforge import ColumnNames
from spforge.data_structures import PlayerRatingsResult, TeamRatingsResult
from spforge.ratings import PlayerRatingGenerator, TeamRatingGenerator


@pytest.fixture
def team_column_names():
    return ColumnNames(
        match_id="match_id",
        team_id="team_id",
        start_date="start_date",
        update_match_id="match_id",
    )


@pytest.fixture
def player_column_names():
    return ColumnNames(
        player_id="player_id",
        team_id="team_id",
        match_id="match_id",
        start_date="start_date",
        update_match_id="match_id",
        participation_weight="participation_weight",
    )


@pytest.fixture
def team_df():
    """Simple 2-match dataset for team ratings."""
    return pl.DataFrame(
        {
            "match_id": ["M1", "M1", "M2", "M2"],
            "team_id": ["T1", "T2", "T1", "T2"],
            "start_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "won": [0.6, 0.4, 0.7, 0.3],
        }
    )


@pytest.fixture
def player_df():
    """Simple 1-match dataset for player ratings."""
    return pl.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4"],
            "team_id": ["T1", "T1", "T2", "T2"],
            "match_id": ["M1", "M1", "M1", "M1"],
            "start_date": ["2024-01-01"] * 4,
            "perf": [0.6, 0.4, 0.7, 0.3],
            "participation_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )


class TestTeamRatingsProperty:
    def test_team_ratings_returns_dict_of_team_ratings_result(self, team_column_names, team_df):
        """When fit_transform is called, team_ratings should return a dict of TeamRatingsResult."""
        gen = TeamRatingGenerator(
            performance_column="won",
            column_names=team_column_names,
            confidence_weight=0.0,
        )
        gen.fit_transform(team_df)

        ratings = gen.team_ratings

        assert isinstance(ratings, dict)
        assert "T1" in ratings
        assert "T2" in ratings
        assert isinstance(ratings["T1"], TeamRatingsResult)
        assert isinstance(ratings["T2"], TeamRatingsResult)

    def test_team_ratings_contains_expected_fields(self, team_column_names, team_df):
        """When team_ratings is accessed, each result should have all expected fields."""
        gen = TeamRatingGenerator(
            performance_column="won",
            column_names=team_column_names,
            confidence_weight=0.0,
        )
        gen.fit_transform(team_df)

        rating = gen.team_ratings["T1"]

        assert rating.id == "T1"
        assert isinstance(rating.offense_rating, float)
        assert isinstance(rating.defense_rating, float)
        assert isinstance(rating.offense_games_played, float)
        assert isinstance(rating.defense_games_played, float)
        assert isinstance(rating.offense_confidence_sum, float)
        assert isinstance(rating.defense_confidence_sum, float)

    def test_team_ratings_games_played_matches_match_count(self, team_column_names, team_df):
        """After 2 matches, games_played should be 2.0 for each team."""
        gen = TeamRatingGenerator(
            performance_column="won",
            column_names=team_column_names,
            confidence_weight=0.0,
        )
        gen.fit_transform(team_df)

        assert gen.team_ratings["T1"].offense_games_played == 2.0
        assert gen.team_ratings["T1"].defense_games_played == 2.0
        assert gen.team_ratings["T2"].offense_games_played == 2.0
        assert gen.team_ratings["T2"].defense_games_played == 2.0

    def test_team_ratings_empty_before_fit(self, team_column_names):
        """Before fit_transform is called, team_ratings should be empty."""
        gen = TeamRatingGenerator(
            performance_column="won",
            column_names=team_column_names,
        )

        assert gen.team_ratings == {}


class TestPlayerRatingsProperty:
    def test_player_ratings_returns_dict_of_player_ratings_result(
        self, player_column_names, player_df
    ):
        """When fit_transform is called, player_ratings should return a dict of PlayerRatingsResult."""
        gen = PlayerRatingGenerator(
            performance_column="perf",
            column_names=player_column_names,
            auto_scale_performance=True,
        )
        gen.fit_transform(player_df)

        ratings = gen.player_ratings

        assert isinstance(ratings, dict)
        assert "P1" in ratings
        assert "P2" in ratings
        assert "P3" in ratings
        assert "P4" in ratings
        assert isinstance(ratings["P1"], PlayerRatingsResult)

    def test_player_ratings_contains_expected_fields(self, player_column_names, player_df):
        """When player_ratings is accessed, each result should have all expected fields."""
        gen = PlayerRatingGenerator(
            performance_column="perf",
            column_names=player_column_names,
            auto_scale_performance=True,
        )
        gen.fit_transform(player_df)

        rating = gen.player_ratings["P1"]

        assert rating.id == "P1"
        assert isinstance(rating.offense_rating, float)
        assert isinstance(rating.defense_rating, float)
        assert isinstance(rating.offense_games_played, float)
        assert isinstance(rating.defense_games_played, float)
        assert isinstance(rating.offense_confidence_sum, float)
        assert isinstance(rating.defense_confidence_sum, float)
        # most_recent_team_id can be str or None
        assert rating.most_recent_team_id is None or isinstance(rating.most_recent_team_id, str)

    def test_player_ratings_games_played_matches_match_count(self, player_column_names, player_df):
        """After 1 match, games_played should be 1.0 for each player."""
        gen = PlayerRatingGenerator(
            performance_column="perf",
            column_names=player_column_names,
            auto_scale_performance=True,
        )
        gen.fit_transform(player_df)

        assert gen.player_ratings["P1"].offense_games_played == 1.0
        assert gen.player_ratings["P1"].defense_games_played == 1.0

    def test_player_ratings_empty_before_fit(self, player_column_names):
        """Before fit_transform is called, player_ratings should be empty."""
        gen = PlayerRatingGenerator(
            performance_column="perf",
            column_names=player_column_names,
        )

        assert gen.player_ratings == {}
