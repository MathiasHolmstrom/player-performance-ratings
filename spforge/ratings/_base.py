import inspect
import logging
from abc import abstractmethod
from typing import Optional, Any, Literal, Union

import narwhals.stable.v2 as nw
import polars as pl
from narwhals.stable.v2 import DataFrame
from narwhals.stable.v2.typing import IntoFrameT

from spforge import ColumnNames
from spforge.data_structures import RatingState
from spforge.ratings import RatingKnownFeatures, RatingUnknownFeatures
from spforge.ratings.league_identifier import LeagueIdentifer2
from spforge.ratings.performance_predictor import RatingMeanPerformancePredictor, RatingDifferencePerformancePredictor, \
    RatingNonOpponentPerformancePredictor
from spforge.transformers.base_transformer import BaseTransformer
from spforge.transformers.fit_transformers import PerformanceWeightsManager, PerformanceManager
from spforge.transformers.fit_transformers._performance_manager import ColumnWeight
from spforge.transformers.lag_transformers._utils import to_polars

MATCH_CONTRIBUTION_TO_SUM_VALUE = 1
EXPECTED_MEAN_CONFIDENCE_SUM = 30


class RatingGenerator(BaseTransformer):

    def __init__(
            self,
            performance_column: str,
            performance_weights: Optional[
                list[Union[ColumnWeight, dict[str, float]]]
            ],
            column_names: ColumnNames,
            features_out: list[str],
            performance_manager: PerformanceManager | None,
            auto_scale_performance: bool,
            performance_predictor: Literal['difference', 'mean', 'ignore_opponent'],
            rating_change_multiplier: float,
            confidence_days_ago_multiplier: float,
            confidence_max_days: int,
            confidence_value_denom: float,
            confidence_max_sum: float,
            confidence_weight: float,
            non_predictor_features_out: Optional[list[RatingKnownFeatures | RatingUnknownFeatures]],
            min_rating_change_multiplier_ratio: float,
            league_rating_change_update_threshold: float,
            league_rating_adjustor_multiplier: float,
            output_suffix: str | None = None,
            **kwargs: Any,
    ):
        self.output_suffix = performance_column if output_suffix is None else output_suffix

        self.performance_predictor = performance_predictor
        self.performance_weights = performance_weights

        self.league_rating_adjustor_multiplier = league_rating_adjustor_multiplier
        self.league_rating_change_update_threshold = (
            league_rating_change_update_threshold
        )
        self.confidence_max_days = confidence_max_days
        self._league_rating_changes: dict[Optional[str], float] = {}
        self._league_rating_changes_count: dict[str, float] = {}
        self.min_rating_change_multiplier_ratio = min_rating_change_multiplier_ratio
        self.rating_change_multiplier = rating_change_multiplier
        self.confidence_max_sum = confidence_max_sum
        self.non_predictor_features_out = non_predictor_features_out or []
        self.non_predictor_features_out = [self._suffix(c) for c in self.non_predictor_features_out]
        self.league_identifier = None
        if performance_predictor == 'mean':
            _performance_predictor_class = RatingMeanPerformancePredictor
            self._features_out = [self._suffix(
                col=RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED)] if features_out is None else [self._suffix(f) for f in features_out]
        elif performance_predictor == 'difference':
            _performance_predictor_class = RatingDifferencePerformancePredictor
            self._features_out = [
                self._suffix(RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED)] if features_out is None else [self._suffix(f) for f in features_out]
        elif performance_predictor == 'ignore_opponent':
            _performance_predictor_class = RatingNonOpponentPerformancePredictor
            self._features_out = [self._suffix(RatingKnownFeatures.TEAM_RATING_PROJECTED)] if features_out is None else [self._suffix(f) for f in features_out]
        else:
            raise ValueError(f"performance_predictor {performance_predictor} is not supported")
        super().__init__(features_out=self._features_out, features=[])
        sig = inspect.signature(_performance_predictor_class.__init__)

        init_params = [
            name for name, param in sig.parameters.items()
            if name != "self"
        ]

        performance_predictor_params = {k: v for k, v in kwargs.items() if k in init_params}
        self._performance_predictor = _performance_predictor_class(**performance_predictor_params)

        self.auto_scale_performance = auto_scale_performance
        self.performance_manager = performance_manager
        self.performance_column = performance_column
        self.column_names = column_names
        self.kwargs = kwargs

        self.performance_manager = self._create_performance_manager()
        self.rating_change_multiplier = rating_change_multiplier
        self.confidence_days_ago_multiplier = confidence_days_ago_multiplier
        self.confidence_max_days = confidence_max_days
        self.confidence_value_denom = confidence_value_denom
        self.confidence_max_sum = confidence_max_sum
        self.confidence_weight = confidence_weight
        self.min_rating_change_multiplier_ratio = min_rating_change_multiplier_ratio

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
            column_names: Optional[ColumnNames] = None,
    ) -> DataFrame | IntoFrameT:
        if not self.performance_manager:
            assert (
                    self.performance_column in df.columns
            ), (
                f"{self.performance_column} not in df. If performance_weights are not set, "
                "performance_column must exist in dataframe"
            )
        self.column_names = column_names if column_names else self.column_names

        if self.column_names.league:
            self.league_identifier = LeagueIdentifer2(column_names=self.column_names)

        if self.performance_manager:
            df = nw.from_native(self.performance_manager.fit_transform(df))

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
        if df.implementation.is_polars():
            pl_df = df.to_native()
        else:
            pl_df = df.to_polars().to_native()

        return self._historical_transform(pl_df)

    @to_polars
    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        pl_df: pl.DataFrame
        if df.implementation.is_polars():
            pl_df = df.to_native()
        else:
            pl_df = df.to_polars().to_native()

        return self._historical_transform(pl_df)

    @to_polars
    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:

        """
        Called only for future fixtures:
        - use existing player ratings to compute pre-match ratings/features
        - do NOT update ratings
        """
        pl_df: pl.DataFrame
        if df.implementation.is_polars():
            pl_df = df.to_native()
        else:
            pl_df = df.to_polars().to_native()

        return self._future_transform(pl_df)

    @abstractmethod
    def _future_transform(self, df: pl.DataFrame):
        pass

    @abstractmethod
    def _historical_transform(self, df: pl.DataFrame):
        pass

    def _create_performance_manager(self) -> PerformanceManager | None:
        if self.performance_manager:
            if self.performance_column and self.performance_column != self.performance_manager.performance_column:
                self.performance_manager.performance_column = self.performance_column
                logging.info(f"Renamed performance column to performance_{self.performance_column}")
            elif not self.performance_column:
                self.performance_column = self.performance_manager.performance_column
            return self.performance_manager

        if self.performance_weights:
            if isinstance(self.performance_weights[0], dict):
                self.performance_weights = [
                    ColumnWeight(**weight) for weight in self.performance_weights
                ]

            return PerformanceWeightsManager(
                weights=self.performance_weights,
                performance_column=self.performance_column,
            )

        if self.auto_scale_performance and not self.performance_manager:
            assert self.performance_column, (
                "performance_column must be set if auto_scale_performance is True"
            )
            if not self.performance_weights:
                return PerformanceManager(
                    features=[self.performance_column],
                    performance_column=self.performance_column,
                )
            else:
                return PerformanceWeightsManager(
                    weights=self.performance_weights,
                    performance_column=self.performance_column,
                )

        return None

    @to_polars
    @nw.narwhalify
    def fit_transform(
            self,
            df: IntoFrameT,
            column_names: Optional[ColumnNames] = None,
    ) -> DataFrame | IntoFrameT:
        if not self.performance_manager:
            assert (
                    self.performance_column in df.columns
            ), (
                f"{self.performance_column} not in df. If performance_weights are not set, "
                "performance_column must exist in dataframe"
            )
        self.column_names = column_names if column_names else self.column_names

        if self.column_names.league:
            self.league_identifier = LeagueIdentifer2(column_names=self.column_names)

        if self.performance_manager:
            df = nw.from_native(self.performance_manager.fit_transform(df))

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
        if df.implementation.is_polars():
            pl_df = df.to_native()
        else:
            pl_df = df.to_polars().to_native()

        return self._historical_transform(pl_df)

    def transform(self, df: IntoFrameT) -> IntoFrameT:
        pl_df: pl.DataFrame
        if df.implementation.is_polars():
            pl_df = df.to_native()
        else:
            pl_df = df.to_polars().to_native()

        return self._historical_transform(pl_df)

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

    def _calculate_days_ago_since_last_match(
            self, last_match_day_number, day_number: int
    ) -> float:
        match_day_number = day_number
        if last_match_day_number is None:
            return 0.0
        return match_day_number - last_match_day_number
