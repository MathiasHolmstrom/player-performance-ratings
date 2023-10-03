from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


@dataclass
class ConfigurationColumnNames:
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
    rating_change_momentum_games_count: int = 6,
    rating_change_momentum_multiplier: float = 0,
    certain_weight: float = 0.9,
    certain_days_ago_multiplier: float = 0.06,
    max_certain_sum: float = 60,
    min_rating_change_for_league: float = 4,
    certain_value_denom: float = 35,
    min_rating_change_multiplier_ratio: float = 0.1,
    reference_certain_sum_value: float = 3
    rating_change_multiplier: float = 50





@dataclass
class MatchOutValues:
    match_id: str
    RatingValues: Dict[RatingColumnNames, List[float]]
    indexes: List[int]




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
class MatchRating:
    pre_match_entity_rating: float
    pre_match_team_rating: float
    pre_match_opponent_rating: float
    pre_match_team_other_rating_type: float = None
    pre_match_opponent_rating_same_type: float = None
    post_match_entity_rating: float = None
    pre_match_projected_team_rating: float = None
    pre_match_projected_opponent_rating: float = None
    pre_match_projected_team_other_rating_type: float = None
    pre_match_projected_opponent_rating_same_rating_type: float = None
    # team_rating_change: float = None
    entity_rating_change: float = None


@dataclass
class MatchPerformanceRating:
    match_performance: float
    participation_weight: float
    projected_participation_weight: float
    ratio: Dict[str, float]
    rating: MatchRating = None


@dataclass
class MatchPlayer:
    id: str
    match_performance_rating: MatchPerformanceRating
    league: str = None


@dataclass
class MatchTeam:
    id: str
    entites: list[MatchPlayer]
    opponent_league: str = None


@dataclass
class Match:
    match_id: str
    teams: List[MatchTeam]
    team_ids: List[str]
    day_number: int
    league: str = None


