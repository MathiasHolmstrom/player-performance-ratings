from dataclasses import dataclass
from typing import List, Optional, Union
from dataclasses_json import DataClassJsonMixin


@dataclass
class ColumnNames:
    team_id: str
    match_id: str
    start_date: str
    player_id: str
    performance: Optional[str]
    league: Optional[str] = None
    position: Optional[str] = None
    participation_weight: Optional[str] = None
    projected_participation_weight: Optional[str] = None
    team_players_percentage_playing_time: Optional[str] = None
    rating_update_id: Optional[str] = None

    def __post_init__(self):
        if self.rating_update_id is None:
            self.rating_update_id = self.match_id


@dataclass
class MatchPerformance:
    performance_value: Optional[float]
    participation_weight: Optional[float]
    projected_participation_weight: float



@dataclass
class PlayerRating(DataClassJsonMixin):
    id: str
    rating_value: float
    name: Optional[str] = None
    games_played: Union[float, int] = 0
    last_match_day_number: int = None
    confidence_sum: float = 0
    prev_rating_changes: List[float] = None


@dataclass
class Team:
    id: str
    player_ids: list[str]
    last_match_day_number: int
    name: Optional[str] = None


@dataclass
class TeamRating(DataClassJsonMixin):
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


@dataclass
class MatchTeam:
    id: str
    players: list[MatchPlayer]
    league: str = None


@dataclass
class Match:
    id: str
    update_id: str
    teams: List[MatchTeam]
    day_number: int
    league: str = None
