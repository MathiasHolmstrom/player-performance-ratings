from dataclasses import dataclass
from typing import List, Optional, Union, Any


@dataclass
class ColumnNames:
    team_id: str
    match_id: str
    start_date: str
    player_id: str
    league: Optional[str] = None
    position: Optional[str] = None
    participation_weight: Optional[str] = None
    projected_participation_weight: Optional[str] = None
    update_match_id: Optional[str] = None
    parent_team_id: Optional[str] = None
    team_players_playing_time: Optional[str] = None
    opponent_players_playing_time: Optional[str] = None
    other_values: Optional[list[str]] = None

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
    performance_value: Optional[float]
    participation_weight: Optional[float]
    projected_participation_weight: float
    team_players_playing_time: Optional[dict[str, float]] = None
    opponent_players_playing_time: Optional[dict[str, float]] = None


@dataclass
class PlayerRating:
    id: str
    rating_value: float
    name: Optional[str] = None
    games_played: Union[float, int] = 0
    last_match_day_number: int = None
    confidence_sum: float = 0
    prev_rating_changes: List[float] = None
    most_recent_team_id: Optional[str] = None


@dataclass
class Team:
    id: str
    player_ids: list[str]
    last_match_day_number: int
    name: Optional[str] = None


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
    league: Optional[str]
    position: Optional[str]
    match_performance: MatchPerformance
    other: Optional[dict[str, Any]] = None


@dataclass
class PreMatchTeamRating:
    id: str
    players: list[PreMatchPlayerRating]
    rating_value: Optional[float]
    projected_rating_value: float
    league: Optional[str]


@dataclass
class PreMatchRating:
    id: str
    teams: list[PreMatchTeamRating]
    day_number: int


@dataclass
class PlayerRatingChange:
    id: str
    day_number: int
    league: Optional[str]
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
    performance: float
    pre_match_projected_rating_value: float
    rating_change_value: float
    league: Optional[str]


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
    performance: Optional[MatchPerformance]
    league: Optional[str] = None
    position: Optional[str] = None
    team_players_participation_weight: Optional[dict[str, float]] = None
    opponent_players_participation_weight: Optional[dict[str, float]] = None
    others: Optional[dict[str, Any]] = None


@dataclass
class MatchTeam:
    id: str
    players: list[MatchPlayer]
    league: str = None
    update_id: Optional[str] = None

    def __post_init__(self):

        if self.update_id is None:
            self.update_id = self.id


@dataclass
class Match:
    id: str
    teams: List[MatchTeam]
    day_number: int
    update_id: Optional[str] = None
    league: str = None

    def __post_init__(self):
        if self.update_id is None:
            self.update_id = self.id
