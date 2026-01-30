# rating_generator_base.py
from __future__ import annotations

import inspect
import logging
from abc import abstractmethod
from typing import Any, Literal

import narwhals.stable.v2 as nw
import polars as pl
from narwhals.stable.v2 import DataFrame
from narwhals.stable.v2.typing import IntoFrameT

from spforge.base_feature_generator import FeatureGenerator
from spforge.data_structures import ColumnNames, RatingState
from spforge.feature_generator._utils import to_polars
from spforge.performance_transformers._performance_manager import (
    ColumnWeight,
    PerformanceManager,
    PerformanceWeightsManager,
)
from spforge.ratings.enums import RatingKnownFeatures, RatingUnknownFeatures
from spforge.ratings.league_identifier import LeagueIdentifer2
from spforge.ratings.player_performance_predictor import (
    PlayerRatingNonOpponentPerformancePredictor,
    RatingMeanPerformancePredictor,
    RatingPlayerDifferencePerformancePredictor,
)

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
EXPECTED_MEAN_CONFIDENCE_SUM = 30


class RatingGenerator(FeatureGenerator):
    def __init__(
        self,
        performance_column: str,
        performance_weights: list[ColumnWeight | dict[str, float]] | None,
        column_names: ColumnNames,
        features_out: list[str] | None,
        performance_manager: PerformanceManager | None,
        auto_scale_performance: bool,
        performance_predictor: Literal["difference", "mean", "ignore_opponent"],
        rating_change_multiplier_offense: float,
        rating_change_multiplier_defense: float,
        confidence_days_ago_multiplier: float,
        confidence_max_days: int,
        confidence_value_denom: float,
        confidence_max_sum: float,
        confidence_weight: float,
        non_predictor_features_out: list[RatingKnownFeatures | RatingUnknownFeatures] | None,
        min_rating_change_multiplier_ratio: float,
        league_rating_change_update_threshold: float,
        league_rating_adjustor_multiplier: float,
        output_suffix: str | None = None,
        **kwargs: Any,
    ):

        if performance_predictor == "mean":
            _performance_predictor_class = RatingMeanPerformancePredictor
            default_features = [RatingKnownFeatures.RATING_MEAN_PROJECTED]
        elif performance_predictor == "difference":
            _performance_predictor_class = RatingPlayerDifferencePerformancePredictor
            default_features = [RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED]
        elif performance_predictor == "ignore_opponent":
            _performance_predictor_class = PlayerRatingNonOpponentPerformancePredictor
            default_features = [RatingKnownFeatures.TEAM_RATING_PROJECTED]
        else:
            raise ValueError(f"performance_predictor {performance_predictor} is not supported")

        self.output_suffix = performance_column if output_suffix is None else output_suffix
        if features_out is None:
            _features_out = [self._suffix(str(f)) for f in default_features]
        else:
            _features_out = [self._suffix(str(f)) for f in features_out]
        super().__init__(features_out=_features_out)

        self.performance_predictor = performance_predictor
        self.performance_weights = performance_weights

        self.league_rating_adjustor_multiplier = league_rating_adjustor_multiplier
        self.league_rating_change_update_threshold = league_rating_change_update_threshold

        self.confidence_max_days = confidence_max_days
        self._league_rating_changes: dict[str | None, float] = {}
        self._league_rating_changes_count: dict[str, float] = {}

        self.min_rating_change_multiplier_ratio = float(min_rating_change_multiplier_ratio)

        # NEW: store both base multipliers
        self.rating_change_multiplier_offense = float(rating_change_multiplier_offense)
        self.rating_change_multiplier_defense = float(rating_change_multiplier_defense)

        self.confidence_max_sum = float(confidence_max_sum)
        self.non_predictor_features_out = non_predictor_features_out or []
        self.non_predictor_features_out = [
            self._suffix(str(c)) for c in self.non_predictor_features_out
        ]

        sig = inspect.signature(_performance_predictor_class.__init__)
        init_params = [name for name, _param in sig.parameters.items() if name != "self"]
        performance_predictor_params = {k: v for k, v in kwargs.items() if k in init_params}
        self._performance_predictor = _performance_predictor_class(**performance_predictor_params)

        self.auto_scale_performance = bool(auto_scale_performance)
        self.performance_manager = performance_manager
        self.performance_column = performance_column
        self.performance_manager = self._create_performance_manager()
        self.performance_column = (
            self.performance_manager.performance_column
            if self.performance_manager
            else self.performance_column
        )
        self.column_names = column_names
        self.kwargs = kwargs

        self.confidence_days_ago_multiplier = float(confidence_days_ago_multiplier)
        self.confidence_value_denom = float(confidence_value_denom)
        self.confidence_weight = float(confidence_weight)

        self.league_identifier = (
            LeagueIdentifer2(column_names=self.column_names)
            if getattr(self.column_names, "league", None)
            else None
        )

    def _suffix(self, col: str) -> str:
        if not self.output_suffix:
            return col
        return f"{col}_{self.output_suffix}"

    @to_polars
    @nw.narwhalify
    def fit_transform(
        self,
        df: IntoFrameT,
        column_names: ColumnNames | None = None,
    ) -> DataFrame | IntoFrameT:
        if not self.performance_manager:
            assert self.performance_column in df.columns, (
                f"{self.performance_column} not in df. If performance_weights are not set, "
                "performance_column must exist in dataframe"
            )

        self.column_names = column_names if column_names else self.column_names

        if self.column_names.league:
            self.league_identifier = LeagueIdentifer2(column_names=self.column_names)

        if self.performance_manager:
            if self.performance_manager:
                ori_perf_values = df[self.performance_manager.ori_performance_column].to_list()
                df = nw.from_native(self.performance_manager.fit_transform(df))
                assert (
                    df[self.performance_manager.ori_performance_column].to_list() == ori_perf_values
                )

        perf = df[self.performance_column]
        if perf.max() > 1.02 or perf.min() < -0.02:
            raise ValueError(
                f"Max {self.performance_column} must be less than than 1.02 and min value larger than -0.02. "
                "Either transform it manually or set auto_scale_performance to True"
            )

        if perf.mean() < 0.42 or perf.mean() > 0.58:
            raise ValueError(
                f"Mean {self.performance_column} must be between 0.42 and 0.58. "
                "Either transform it manually or set auto_scale_performance to True"
            )

        pl_df: pl.DataFrame
        pl_df = df.to_native() if df.implementation.is_polars() else df.to_polars().to_native()

        return self._historical_transform(pl_df)

    @to_polars
    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        if self.performance_manager and self.performance_manager.ori_performance_column in df.columns:
            df = nw.from_native(self.performance_manager.transform(df))

        pl_df: pl.DataFrame
        pl_df = df.to_native() if df.implementation.is_polars() else df.to_polars().to_native()
        return self._historical_transform(pl_df)

    @to_polars
    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        """
        Called only for future fixtures:
        - use existing ratings to compute pre-match ratings/features
        - do NOT update ratings
        """
        if self.performance_manager and self.performance_manager.ori_performance_column in df.columns:
            df = nw.from_native(self.performance_manager.transform(df))

        pl_df: pl.DataFrame
        pl_df = df.to_native() if df.implementation.is_polars() else df.to_polars().to_native()
        return self._future_transform(pl_df)

    @abstractmethod
    def _future_transform(self, df: pl.DataFrame):
        pass

    @abstractmethod
    def _historical_transform(self, df: pl.DataFrame):
        pass

    def _create_performance_manager(self) -> PerformanceManager | None:
        if self.performance_manager:
            if (
                self.performance_column
                and self.performance_column != self.performance_manager.performance_column
            ):
                self.performance_manager.performance_column = self.performance_column
                logging.info(f"Renamed performance column to performance_{self.performance_column}")
            elif not self.performance_column:
                self.performance_column = self.performance_manager.performance_column
            return self.performance_manager

        if self.performance_weights:
            if isinstance(self.performance_weights[0], dict):
                # Map 'col' to 'name' for backward compatibility
                converted_weights = []
                for weight in self.performance_weights:
                    weight_dict = dict(weight)
                    if "col" in weight_dict and "name" not in weight_dict:
                        weight_dict["name"] = weight_dict.pop("col")
                    converted_weights.append(ColumnWeight(**weight_dict))
                self.performance_weights = converted_weights
            return PerformanceWeightsManager(
                weights=self.performance_weights,
                performance_column=self.performance_column,
            )

        if self.auto_scale_performance and not self.performance_manager:
            assert (
                self.performance_column
            ), "performance_column must be set if auto_scale_performance is True"
            if not self.performance_weights:
                return PerformanceManager(
                    features=[self.performance_column],
                    performance_column=self.performance_column,
                )
            return PerformanceWeightsManager(
                weights=self.performance_weights,
                performance_column=self.performance_column,
            )

        return None

    def _add_day_number(
        self,
        df: pl.DataFrame,
        date_col: str | None = None,
        out_col: str = "__day_number",
        ref_date: str = "2000-01-01",  # <- new: reference epoch (YYYY-MM-DD)
        one_based: bool = True,  # <- optional: keep 1-based numbering like before
    ) -> pl.DataFrame:
        """
        Add day_number column to dataframe relative to a fixed reference date.

        day_number = (date - ref_date) + (1 if one_based else 0)

        Supports various date formats: string, Date, Datetime (with or without timezone).

        Args:
            df: Input dataframe
            date_col: Column name containing dates (defaults to column_names.start_date)
            out_col: Output column name (defaults to "__day_number")
            ref_date: Reference date in YYYY-MM-DD (defaults to 2000-01-01)
            one_based: If True, 2000-01-01 -> 1. If False, 2000-01-01 -> 0.

        Returns:
            DataFrame with added day_number column
        """
        cn = self.column_names
        date_column = date_col if date_col else cn.start_date

        if date_column not in df.columns:
            raise ValueError(
                f"Date column '{date_column}' not found in dataframe columns: {df.columns}"
            )

        dtype = df.schema.get(date_column)
        c = pl.col(date_column)

        if dtype == pl.Utf8 or (hasattr(pl, "String") and dtype == pl.String):
            dt = (
                c.str.strptime(pl.Datetime(time_zone=None), "%Y-%m-%d", strict=False)
                .fill_null(
                    c.str.strptime(pl.Datetime(time_zone=None), "%Y-%m-%d %H:%M:%S", strict=False)
                )
                .fill_null(
                    c.str.strptime(pl.Datetime(time_zone=None), "%Y-%m-%dT%H:%M:%S", strict=False)
                )
                .fill_null(c.cast(pl.Datetime(time_zone=None), strict=False))
            )
            dt = dt.dt.replace_time_zone(None)
        elif dtype == pl.Date:
            dt = c.cast(pl.Datetime(time_zone=None))
        elif isinstance(dtype, pl.Datetime):
            dt = (
                c
                if dtype.time_zone is None
                else c.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
            )
        else:
            dt = c.cast(pl.Datetime(time_zone=None), strict=False).dt.replace_time_zone(None)

        start_as_int = dt.cast(pl.Date).cast(pl.Int32)

        ref_int = pl.lit(ref_date).str.strptime(pl.Date, "%Y-%m-%d", strict=True).cast(pl.Int32)

        offset = 1 if one_based else 0
        day_number = (start_as_int - ref_int + offset).fill_null(offset)

        return df.with_columns(day_number.alias(out_col))

    def _applied_multiplier(self, state: RatingState, base_multiplier: float) -> float:
        min_mult = base_multiplier * self.min_rating_change_multiplier_ratio
        conf_mult = base_multiplier * (
            (EXPECTED_MEAN_CONFIDENCE_SUM - state.confidence_sum) / self.confidence_value_denom + 1
        )
        applied = (
            conf_mult * self.confidence_weight + (1 - self.confidence_weight) * base_multiplier
        )
        return max(float(min_mult), float(applied))

    def _post_match_confidence_sum(
        self, state: RatingState, day_number: int, participation_weight: float
    ) -> float:
        days_ago = (
            0.0
            if state.last_match_day_number is None
            else float(day_number - state.last_match_day_number)
        )
        val = (
            -min(days_ago, self.confidence_max_days) * self.confidence_days_ago_multiplier
            + state.confidence_sum
            + MATCH_CONTRIBUTION_TO_SUM_VALUE * participation_weight
        )
        return max(0.0, min(float(val), self.confidence_max_sum))

    def _calculate_days_ago_since_last_match(self, last_match_day_number, day_number: int) -> float:
        if last_match_day_number is None:
            return 0.0
        return float(day_number - last_match_day_number)
