from dataclasses import dataclass
from typing import Optional, Any

import polars as pl
from abc import abstractmethod, ABC

from narwhals import DataFrame
from narwhals.stable.v1.typing import IntoFrameT
from spforge import ColumnNames
from spforge.ratings.league_identifier import LeagueIdentifer2

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
EXPECTED_MEAN_CONFIDENCE_SUM = 30

@dataclass
class RatingState:
    """Generic rating state (works for players or teams)."""
    id: str
    rating_value: float
    confidence_sum: float = 0.0
    games_played: float = 0.0
    last_match_day_number: Optional[int] = None
    most_recent_group_id: Optional[str] = None  # e.g. team_id for players, league, etc.


class RatingGenerator(ABC):
    """
    Template-method base class for rating generators.

    Shared:
      - narwhals -> native polars conversion
      - optional league enrichment
      - day_number creation
      - confidence + applied multiplier logic

    Subclasses implement:
      - how to build match_df (the "work unit" frame that gets iterated)
      - how to compute ratings outputs + join back
      - feature augmentation / column dropping rules
    """

    def __init__(
        self,
        performance_column: str,
        column_names: ColumnNames,
        rating_change_multiplier: float = 50,
        confidence_days_ago_multiplier: float = 0.06,
        confidence_max_days: int = 90,
        confidence_value_denom: float = 140,
        confidence_max_sum: float = 150,
        confidence_weight: float = 0.9,
        min_rating_change_multiplier_ratio: float = 0.1,
        team_id_change_confidence_sum_decrease: float = 0.0,  # useful for players, usually 0 for teams
        **kwargs: Any,
    ):
        self.performance_column = performance_column
        self.column_names = column_names
        self.kwargs = kwargs

        # update knobs
        self.rating_change_multiplier = rating_change_multiplier
        self.confidence_days_ago_multiplier = confidence_days_ago_multiplier
        self.confidence_max_days = confidence_max_days
        self.confidence_value_denom = confidence_value_denom
        self.confidence_max_sum = confidence_max_sum
        self.confidence_weight = confidence_weight
        self.min_rating_change_multiplier_ratio = min_rating_change_multiplier_ratio
        self.team_id_change_confidence_sum_decrease = team_id_change_confidence_sum_decrease

        self.league_identifier = (
            LeagueIdentifer2(column_names=self.column_names)
            if getattr(self.column_names, "league", None)
            else None
        )

    @abstractmethod
    def fit_transform(self, df: IntoFrameT) -> DataFrame:
        pass

    def _add_day_number(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names
        start_as_int = (
            pl.col(cn.start_date)
            .str.strptime(pl.Date, "%Y-%m-%d")
            .cast(pl.Int32)
        )
        return df.with_columns((start_as_int - start_as_int.min() + 1).alias("__day_number"))

    def _applied_multiplier(self, state: RatingState) -> float:
        min_mult = self.rating_change_multiplier * self.min_rating_change_multiplier_ratio
        conf_mult = self.rating_change_multiplier * (
            (EXPECTED_MEAN_CONFIDENCE_SUM - state.confidence_sum) / self.confidence_value_denom + 1
        )
        applied = conf_mult * self.confidence_weight + (1 - self.confidence_weight) * self.rating_change_multiplier
        return max(float(min_mult), float(applied))

    def _post_match_confidence_sum(self, state: RatingState, day_number: int, participation_weight: float) -> float:
        days_ago = 0.0 if state.last_match_day_number is None else float(day_number - state.last_match_day_number)
        val = (
            -min(days_ago, self.confidence_max_days) * self.confidence_days_ago_multiplier
            + state.confidence_sum
            + MATCH_CONTRIBUTION_TO_SUM_VALUE * participation_weight
        )
        return max(0.0, min(float(val), self.confidence_max_sum))

