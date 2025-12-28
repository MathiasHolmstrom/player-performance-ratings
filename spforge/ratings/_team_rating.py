from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal, Any

import polars as pl


from spforge.data_structures import ColumnNames
from spforge.ratings._base import RatingGenerator, RatingState

from spforge.ratings import RatingKnownFeatures


TEAM_PRED_COL = "__TEAM_PREDICTED_PERFORMANCE"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class TeamRatingGenerator(RatingGenerator):

    def __init__(
        self,
        performance_column: str,
        column_names: ColumnNames,
        performance_predictor: Literal["difference", "mean", "ignore_opponent"] = "difference",
        rating_diff_coef: float = 0.001,
        rating_mean_coef: float = 0.001,
        team_rating_coef: float = 0.001,
        start_team_rating: float = 1000.0,
        rating_center: Optional[float] = None,
        features_out: Optional[list[str]] = None,
        rating_change_multiplier: float = 50,
        confidence_days_ago_multiplier: float = 0.06,
        confidence_max_days: int = 90,
        confidence_value_denom: float = 140,
        confidence_max_sum: float = 150,
        confidence_weight: float = 0.9,
        min_rating_change_multiplier_ratio: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(
            performance_column=performance_column,
            column_names=column_names,
            rating_change_multiplier=rating_change_multiplier,
            confidence_days_ago_multiplier=confidence_days_ago_multiplier,
            confidence_max_days=confidence_max_days,
            confidence_value_denom=confidence_value_denom,
            confidence_max_sum=confidence_max_sum,
            confidence_weight=confidence_weight,
            min_rating_change_multiplier_ratio=min_rating_change_multiplier_ratio,
            team_id_change_confidence_sum_decrease=0.0,
            **kwargs,
        )

        self.performance_predictor = performance_predictor
        self.rating_diff_coef = rating_diff_coef
        self.rating_mean_coef = rating_mean_coef
        self.team_rating_coef = team_rating_coef

        self.start_team_rating = float(start_team_rating)
        self.rating_center = float(start_team_rating if rating_center is None else rating_center)

        self.features_out = features_out or []
        self._team_ratings: dict[str, RatingState] = {}

    def _build_match_df(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        cols = [cn.match_id, cn.team_id, cn.start_date, cn.update_match_id, self.performance_column]
        if getattr(cn, "league", None):
            cols.append(cn.league)

        team_rows = df.select(cols).unique([cn.match_id, cn.team_id])

        return (
            team_rows.join(team_rows, on=cn.match_id, how="inner", suffix="_opponent")
            .filter(pl.col(cn.team_id) != pl.col(f"{cn.team_id}_opponent"))
            .sort(list(set([cn.start_date, cn.match_id, cn.update_match_id])))
        )

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
        return _clip01(float(pred))

    def _ensure_team(self, team_id: str) -> RatingState:
        if team_id not in self._team_ratings:
            self._team_ratings[team_id] = RatingState(id=team_id, rating_value=self.start_team_rating)
        return self._team_ratings[team_id]

    def _calculate_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        team_ids_out = []
        match_ids_out = []
        team_rating_projected_out = []
        team_pred_perf_out = []

        for match_id, g in match_df.group_by(cn.match_id, maintain_order=True):
            rows = g.to_dicts()
            by_team = {r[cn.team_id]: r for r in rows}

            if len(by_team) != 2:
                raise ValueError(f"Expected exactly 2 teams for match_id={match_id}, got {len(by_team)}")

            t1, t2 = list(by_team.keys())[0], list(by_team.keys())[1]
            r1, r2 = by_team[t1], by_team[t2]

            day_number = int(r1["__day_number"])

            s1 = self._ensure_team(t1)
            s2 = self._ensure_team(t2)

            pre1 = float(s1.rating_value)
            pre2 = float(s2.rating_value)

            pred1 = self._predict_performance(pre1, pre2)
            pred2 = self._predict_performance(pre2, pre1)

            perf1 = float(r1[self.performance_column])
            perf2 = float(r2[self.performance_column])

            mult1 = super()._applied_multiplier(s1)
            mult2 = super()._applied_multiplier(s2)

            change1 = (perf1 - pred1) * mult1
            change2 = (perf2 - pred2) * mult2

            if math.isnan(change1) or math.isnan(change2):
                raise ValueError(f"NaN rating change in match_id={match_id}")

            s1.confidence_sum = super()._post_match_confidence_sum(s1, day_number, 1.0)
            s2.confidence_sum = super()._post_match_confidence_sum(s2, day_number, 1.0)

            s1.rating_value += change1
            s2.rating_value += change2

            s1.games_played += 1.0
            s2.games_played += 1.0

            s1.last_match_day_number = day_number
            s2.last_match_day_number = day_number

            team_ids_out.extend([t1, t2])
            match_ids_out.extend([match_id, match_id])
            team_rating_projected_out.extend([pre1, pre2])
            team_pred_perf_out.extend([pred1, pred2])

        return pl.DataFrame(
            {
                cn.team_id: team_ids_out,
                cn.match_id: match_ids_out,
                RatingKnownFeatures.TEAM_RATING_PROJECTED: team_rating_projected_out,
                TEAM_PRED_COL: team_pred_perf_out,
            },
            strict=False,
        )


    def _add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        if RatingKnownFeatures.TEAM_RATING_PROJECTED in df.columns:
            df = df.with_columns(
                pl.col(RatingKnownFeatures.TEAM_RATING_PROJECTED)
                .reverse()
                .over(cn.match_id)
                .alias(RatingKnownFeatures.OPPONENT_RATING_PROJECTED)
            )

        if RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED in self.features_out:
            df = df.with_columns(
                (
                    pl.col(RatingKnownFeatures.TEAM_RATING_PROJECTED)
                    - pl.col(RatingKnownFeatures.OPPONENT_RATING_PROJECTED)
                ).alias(RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED)
            )

        if RatingKnownFeatures.RATING_MEAN_PROJECTED in self.features_out:
            df = df.with_columns(
                (
                    pl.col(RatingKnownFeatures.TEAM_RATING_PROJECTED)
                    + pl.col(RatingKnownFeatures.OPPONENT_RATING_PROJECTED)
                ).truediv(2.0).alias(RatingKnownFeatures.RATING_MEAN_PROJECTED)
            )

        return df