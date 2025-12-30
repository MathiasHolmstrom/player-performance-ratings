from __future__ import annotations

import math
from typing import Optional, Literal, Any, Union

import polars as pl

from spforge.data_structures import ColumnNames
from spforge.ratings._base import RatingGenerator, RatingState
from spforge.ratings import RatingKnownFeatures, RatingUnknownFeatures
from spforge.ratings.utils import add_day_number_utc
from spforge.transformers.fit_transformers._performance_manager import ColumnWeight, PerformanceManager

TEAM_PRED_COL = "__TEAM_PREDICTED_PERFORMANCE"


class TeamRatingGenerator(RatingGenerator):
    def __init__(
        self,
        performance_column: str,
        performance_weights: Optional[list[Union[ColumnWeight, dict[str, float]]]] = None,
        performance_manager: PerformanceManager | None = None,
        auto_scale_performance: bool = False,
        performance_predictor: Literal["difference", "mean", "ignore_opponent"] = "difference",
        rating_change_multiplier: float = 50,
        confidence_days_ago_multiplier: float = 0.06,
        confidence_max_days: int = 90,
        confidence_value_denom: float = 140,
        confidence_max_sum: float = 150,
        confidence_weight: float = 0.9,
        features_out: Optional[list[RatingKnownFeatures]] = None,
        non_predictor_features_out: Optional[list[RatingKnownFeatures | RatingUnknownFeatures]] = None,
        min_rating_change_multiplier_ratio: float = 0.1,
        league_rating_change_update_threshold: float = 100,
        league_rating_adjustor_multiplier: float = 0.05,
        column_names: Optional[ColumnNames] = None,
        output_suffix: Optional[str] = None,
        start_team_rating: float = 1000.0,
        rating_center: Optional[float] = None,
        rating_diff_coef: float = 0.001,
        rating_mean_coef: float = 0.001,
        team_rating_coef: float = 0.001,
        **kwargs: Any,
    ):



        super().__init__(
            output_suffix=output_suffix,
            performance_column=performance_column,
            performance_weights=performance_weights,
            column_names=column_names,
            features_out=features_out,
            performance_manager=performance_manager,
            auto_scale_performance=auto_scale_performance,
            performance_predictor=performance_predictor,
            rating_change_multiplier=rating_change_multiplier,
            confidence_days_ago_multiplier=confidence_days_ago_multiplier,
            confidence_max_days=confidence_max_days,
            confidence_value_denom=confidence_value_denom,
            confidence_max_sum=confidence_max_sum,
            confidence_weight=confidence_weight,
            non_predictor_features_out=non_predictor_features_out,
            min_rating_change_multiplier_ratio=min_rating_change_multiplier_ratio,
            league_rating_change_update_threshold=league_rating_change_update_threshold,
            league_rating_adjustor_multiplier=league_rating_adjustor_multiplier,
            **kwargs,
        )

        self.TEAM_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.TEAM_RATING_PROJECTED))
        self.OPP_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.OPPONENT_RATING_PROJECTED))
        self.DIFF_PROJ_COL = self._suffix(str(RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED))
        self.MEAN_PROJ_COL = self._suffix(str(RatingKnownFeatures.RATING_MEAN_PROJECTED))
        self.TEAM_PRED_PERF_COL = self._suffix(TEAM_PRED_COL)
        self.start_team_rating = float(start_team_rating)
        self.rating_center = float(self.start_team_rating if rating_center is None else rating_center)
        self._team_ratings: dict[str, RatingState] = {}

        self.performance_predictor = performance_predictor
        self.rating_diff_coef = float(rating_diff_coef)
        self.rating_mean_coef = float(rating_mean_coef)
        self.team_rating_coef = float(team_rating_coef)

    def _ensure_team(self, team_id: str) -> RatingState:
        if team_id not in self._team_ratings:
            self._team_ratings[team_id] = RatingState(id=team_id, rating_value=self.start_team_rating)
        return self._team_ratings[team_id]

    def _calculate_applied_rating_change_multiplier(self, team_id: str) -> float:
        return self._applied_multiplier(self._team_ratings[team_id])

    def _calculate_post_match_confidence_sum(self, state: RatingState, day_number: int) -> float:
        return self._post_match_confidence_sum(state=state, day_number=day_number, participation_weight=1.0)

    def _create_match_df(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        base_cols = list(dict.fromkeys([cn.match_id, cn.team_id, cn.start_date, cn.update_match_id]))
        if self.performance_column in df.columns:
            base_cols.append(self.performance_column)

        team_rows = df.select([c for c in base_cols if c in df.columns]).unique([cn.match_id, cn.team_id])

        match_df = (
            team_rows.join(team_rows, on=cn.match_id, how="inner", suffix="_opponent")
            .filter(pl.col(cn.team_id) != pl.col(f"{cn.team_id}_opponent"))
            .sort(list(set([cn.start_date, cn.match_id, cn.update_match_id])))
        )

        match_df = add_day_number_utc(match_df, cn.start_date, "__day_number")
        return match_df

    def _predict_performance(self, team_rating: float, opp_rating: float) -> float:
        if self.performance_predictor == "difference":
            pred = 0.5 + self.rating_diff_coef * (team_rating - opp_rating)
        elif self.performance_predictor == "mean":
            mean_rating = (team_rating + opp_rating) / 2.0
            pred = 0.5 + self.rating_mean_coef * (mean_rating - self.rating_center)
        elif self.performance_predictor == "ignore_opponent":
            pred = 0.5 + self.team_rating_coef * (team_rating - self.rating_center)
        else:
            raise ValueError(f"Unsupported performance_predictor={self.performance_predictor}")
        return max(0.0, min(1.0, float(pred)))

    def _historical_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names
        assert df.group_by(cn.match_id).agg(pl.col(cn.team_id).n_unique())[cn.team_id].n_unique() ==1
        assert df.group_by(cn.match_id).agg(pl.col(cn.team_id).n_unique())[cn.team_id].max() ==2
        match_df = self._create_match_df(df)
        ratings = self._calculate_ratings(match_df)

        cols = [
            c for c in df.columns
            if c not in (self.TEAM_PRED_PERF_COL, self.TEAM_RATING_PROJ_COL)
        ]

        df2 = df.select(cols).join(ratings, on=[cn.match_id, cn.team_id], how="left")
        return self._add_rating_features(df2)

    def _calculate_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        rows: list[dict] = []
        pending_updates: list[tuple[str, float, int]] = []
        last_update_id = None

        for r in match_df.iter_rows(named=True):
            team_id = r[cn.team_id]
            opp_id = r[f"{cn.team_id}_opponent"]
            update_id = r[cn.update_match_id]
            day_number = int(r["__day_number"])

            s = self._ensure_team(team_id)
            o = self._ensure_team(opp_id)

            pre = float(s.rating_value)
            opp_pre = float(o.rating_value)

            pred = self._predict_performance(pre, opp_pre)
            perf = float(r[self.performance_column]) if r.get(self.performance_column) is not None else 0.0

            mult = self._calculate_applied_rating_change_multiplier(team_id)
            change = (perf - pred) * mult

            if math.isnan(change):
                raise ValueError(f"NaN rating change for team_id={team_id}, match_id={r[cn.match_id]}")

            rows.append(
                {
                    cn.match_id: r[cn.match_id],
                    cn.team_id: team_id,
                    self.TEAM_RATING_PROJ_COL: pre,
                    self.TEAM_PRED_PERF_COL: pred,
                }
            )

            pending_updates.append((team_id, change, day_number))

            if last_update_id is None:
                last_update_id = update_id

            if update_id != last_update_id:
                self._apply_team_updates(pending_updates[:-1])
                pending_updates = pending_updates[-1:]
                last_update_id = update_id

        if pending_updates:
            self._apply_team_updates(pending_updates)

        return pl.DataFrame(rows, strict=False)

    def _apply_team_updates(self, updates: list[tuple[str, float, int]]) -> None:
        for team_id, change, day_number in updates:
            s = self._team_ratings[team_id]
            s.confidence_sum = self._calculate_post_match_confidence_sum(s, day_number)
            s.rating_value += change
            s.games_played += 1.0
            s.last_match_day_number = day_number

    def _add_rating_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cols_to_add = set((self.features_out or []) + (self.non_predictor_features_out or []))
        cn = self.column_names

        need_opp = (self.OPP_RATING_PROJ_COL in cols_to_add) or (self.DIFF_PROJ_COL in cols_to_add) or (self.MEAN_PROJ_COL in cols_to_add)
        need_diff = self.DIFF_PROJ_COL in cols_to_add
        need_mean = self.MEAN_PROJ_COL in cols_to_add

        if need_opp and self.OPP_RATING_PROJ_COL not in df.columns:
            df = df.with_columns(
                pl.col(self.TEAM_RATING_PROJ_COL).reverse().over(cn.match_id).alias(self.OPP_RATING_PROJ_COL)
            )

        if need_diff and self.DIFF_PROJ_COL not in df.columns:
            df = df.with_columns((pl.col(self.TEAM_RATING_PROJ_COL) - pl.col(self.OPP_RATING_PROJ_COL)).alias(self.DIFF_PROJ_COL))

        if need_mean and self.MEAN_PROJ_COL not in df.columns:
            df = df.with_columns(((pl.col(self.TEAM_RATING_PROJ_COL) + pl.col(self.OPP_RATING_PROJ_COL)) / 2.0).alias(self.MEAN_PROJ_COL))


        candidates = {self.TEAM_RATING_PROJ_COL, self.OPP_RATING_PROJ_COL, self.DIFF_PROJ_COL, self.MEAN_PROJ_COL, self.TEAM_PRED_PERF_COL}
        drop_cols = [c for c in candidates if (c in df.columns and c not in cols_to_add)]

        return df.drop(drop_cols)

    def _future_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names
        assert df.group_by(cn.match_id).agg(pl.col(cn.team_id).n_unique())[cn.team_id].n_unique() ==1
        assert df.group_by(cn.match_id).agg(pl.col(cn.team_id).n_unique())[cn.team_id].max() ==2
        match_df = self._create_match_df(df)
        ratings = self._calculate_future_ratings(match_df)

        cols = [
            c for c in df.columns
            if c not in (self.TEAM_PRED_PERF_COL, self.TEAM_RATING_PROJ_COL)
        ]

        df2 = df.select(cols).join(ratings, on=[cn.match_id, cn.team_id], how="left")
        return self._add_rating_features(df2)

    def _calculate_future_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        rows: list[dict] = []
        for r in match_df.iter_rows(named=True):
            team_id = r[cn.team_id]
            opp_id = r[f"{cn.team_id}_opponent"]

            s = self._ensure_team(team_id)
            o = self._ensure_team(opp_id)

            pre = float(s.rating_value)
            opp_pre = float(o.rating_value)

            pred = self._predict_performance(pre, opp_pre)

            rows.append(
                {
                    cn.match_id: r[cn.match_id],
                    cn.team_id: team_id,
                    self.TEAM_RATING_PROJ_COL: pre,
                    self.TEAM_PRED_PERF_COL: pred,
                }
            )

        return pl.DataFrame(rows, strict=False)
