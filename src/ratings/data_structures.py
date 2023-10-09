from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


@dataclass
class ColumnNames:
    team_id: str
    match_id: str
    start_date_time: str
    default_performance: str = None
    offense_performance: str = None
    defense_performance: str = None
    player_id: str = None
    league: str = None
    participation_weight: str = None
    projected_participation_weight: str = None
    team_players_percentage_playing_time: str = None
    parent_match_id: str = None


@dataclass
class StartRatingParameters:
    start_league_ratings: Optional[dict[str, float]] = None
    league_quantile: float = 0.2
    team_rating_subtract: float = 80
    team_weight: float = 0.2


@dataclass
class RatingUpdateParameters:
    certain_weight: float = 0.9,
    certain_days_ago_multiplier: float = 0.06,
    max_days_ago: int = 90
    max_certain_sum: float = 60,
    min_rating_change_for_league: float = 4,
    certain_value_denom: float = 35,
    min_rating_change_multiplier_ratio: float = 0.1,
    reference_certain_sum_value: float = 3
    rating_change_multiplier: float = 50


@dataclass
class PerformancePredictorParameters:
    rating_diff_coef: float
    rating_diff_team_from_entity_coef: float
    team_rating_diff_coef: float
    max_predict_value: float = 1,


@dataclass
class TeamRatingParameters:
    min_match_count: int


@dataclass
class MatchPerformance:
    performance: float
    participation_weight: float
    projected_participation_weight: float
    ratio: Dict[str, float]


@dataclass
class PlayerRating:
    id: str
    rating: float
    games_played: int = 0
    last_match_day_number: int = None
    certain_ratio: float = 0
    certain_sum: float = 0
    prev_rating_changes: List[float] = None


@dataclass
class PreMatchPlayerRating:
    id: str
    rating: float
    projected_rating: float
    match_performance: MatchPerformance


@dataclass
class PreMatchTeamRating:
    id: str
    players: Dict[str, PreMatchPlayerRating]
    rating: float


@dataclass
class PreMatchRating:
    id: str
    teams: dict[str, PreMatchTeamRating]


@dataclass
class PostMatchPlayerRating:
    id: str
    rating: float


@dataclass
class PostMatchTeamRating:
    id: str
    players: Dict[str, PlayerRating]
    rating: float


@dataclass
class PostMatchRating:
    id: str
    teams: dict[str, PostMatchTeamRating]


@dataclass
class MatchRating:
    id: str
    pre_match_rating: PreMatchRating
    post_match_rating: PostMatchRating
    pre_match_player_ratings: list[float]
    pre_match_team_ratings: list[float]
    pre_match_opponent_ratings: list[float]


@dataclass
class MatchRatings:
    pre_match_team_ratings: list[float]
    pre_match_player_ratings: list[float]
    pre_match_opponent_ratings: list[float]


@dataclass
class MatchPlayer:
    id: str
    match_player_performance: MatchPerformance
    league: str = None


@dataclass
class MatchTeam:
    id: str
    players: list[MatchPlayer]
    opponent_league: str = None


@dataclass
class Match:
    id: str
    teams: List[MatchTeam]
    team_ids: List[str]
    day_number: int
    league: str = None
