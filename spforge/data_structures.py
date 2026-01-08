from dataclasses import dataclass
from typing import Any


@dataclass
class ColumnNames:
    team_id: str
    match_id: str
    start_date: str
    player_id: str | None = None
    league: str | None = None
    position: str | None = None
    participation_weight: str | None = None
    projected_participation_weight: str | None = None
    update_match_id: str | None = None
    parent_team_id: str | None = None
    team_players_playing_time: str | None = None
    opponent_players_playing_time: str | None = None
    other_values: list[str] | None = None

    def __post_init__(self):
        if self.update_match_id is None:
            self.update_match_id = self.match_id

        if self.parent_team_id is None:
            self.parent_team_id = self.team_id

        if self.update_match_id != self.match_id and self.parent_team_id is None:
            raise ValueError(
                "rating_update_team_id must be passed if rating_update_match_id is passed"
            )

        if self.parent_team_id != self.team_id and self.update_match_id is None:
            raise ValueError(
                "rating_update_match_id must be passed if rating_update_team_id is passed"
            )


@dataclass
class MatchPerformance:
    performance_value: float | None
    participation_weight: float | None
    projected_participation_weight: float
    team_players_playing_time: dict[str, float] | None = None
    opponent_players_playing_time: dict[str, float] | None = None


@dataclass
class RatingState:
    """Generic rating state (works for players or teams)."""

    id: str
    rating_value: float
    confidence_sum: float = 0.0
    games_played: float = 0.0
    last_match_day_number: int | None = None
    most_recent_group_id: str | None = None  # e.g. team_id for players, league, etc.
    prev_rating_changes: list[float] = None


@dataclass
class Team:
    id: str
    player_ids: list[str]
    last_match_day_number: int
    name: str | None = None


@dataclass
class PlayerRating(RatingState):
    most_recent_team_id: str | None = None


@dataclass
class TeamRating:
    id: str
    name: str
    rating_value: float
    players: list[PlayerRating]
    games_played: int = 0
    last_match_day_number: int = None


@dataclass
class PreMatchPlayerRating:
    id: str
    rating_value: float
    games_played: int
    league: str | None
    position: str | None
    match_performance: MatchPerformance
    other: dict[str, Any] | None = None


@dataclass
class PreMatchTeamRating:
    id: str
    players: list[PreMatchPlayerRating]
    rating_value: float


@dataclass
class PreMatchRating:
    id: str
    teams: list[PreMatchTeamRating]
    day_number: int


@dataclass
class PlayerRatingChange:
    id: str
    day_number: int
    league: str | None
    participation_weight: float
    predicted_performance: float
    performance: float
    pre_match_rating_value: float
    rating_change_value: float


@dataclass
class TeamRatingChange:
    id: str
    players: list[PlayerRatingChange]
    predicted_performance: float
    predicted_player_performances: list[float]
    performance: float
    pre_match_projected_rating_value: float
    rating_change_value: float
    league: str | None


@dataclass
class PostMatchTeamRatingChange:
    id: str
    players: list[PlayerRatingChange]
    rating_value: float
    predicted_performance: float


@dataclass
class PostMatchRatingChange:
    id: str
    teams: list[PostMatchTeamRatingChange]


@dataclass
class MatchRating:
    id: str
    pre_match_rating: PreMatchRating
    post_match_rating: PostMatchRatingChange


@dataclass
class MatchRatings:
    pre_match_team_rating_projected_values: list[float]
    pre_match_team_rating_values: list[float]
    pre_match_player_rating_values: list[float]
    pre_match_opponent_rating_values: list[float]
    player_rating_changes: list[float]
    player_leagues: list[str]
    team_opponent_leagues: list[str]
    match_ids: list[str]


@dataclass
class MatchPlayer:
    id: str
    performance: MatchPerformance | None
    league: str | None = None
    position: str | None = None
    team_players_participation_weight: dict[str, float] | None = None
    opponent_players_participation_weight: dict[str, float] | None = None
    others: dict[str, Any] | None = None


@dataclass
class PreMatchPlayersCollection:
    pre_match_player_ratings: list[PreMatchPlayerRating]
    player_rating_values: list[float]
    new_players: list[MatchPlayer]
    player_ids: list[str]
    projected_particiation_weights: list[float]


@dataclass
class NewPlayerPreMatchPlayersCollection:
    pre_match_player_ratings: list[PreMatchPlayerRating]
    player_rating_values: list[float]


@dataclass
class MatchTeam:
    id: str
    players: list[MatchPlayer]
    league: str = None
    update_id: str | None = None

    def __post_init__(self):

        if self.update_id is None:
            self.update_id = self.id


@dataclass
class Match:
    id: str
    teams: list[MatchTeam]
    day_number: int
    update_id: str | None = None
    league: str = None

    def __post_init__(self):
        if self.update_id is None:
            self.update_id = self.id
