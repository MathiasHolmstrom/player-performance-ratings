from __future__ import annotations

import copy
from dataclasses import dataclass

import narwhals.stable.v2 as nw
import polars as pl
from narwhals.stable.v2.typing import IntoFrameT


DEFAULT_START_RATING = 1000.0


@dataclass
class LeagueStartRatingOptimizationResult:
    league_ratings: dict[str, float]
    iteration_errors: list[dict[str, float]]


class LeagueStartRatingOptimizer:
    def __init__(
        self,
        rating_generator: object,
        n_iterations: int = 3,
        learning_rate: float = 0.2,
        min_cross_region_rows: int = 10,
        rating_scale: float | None = None,
    ):
        self.rating_generator = rating_generator
        self.n_iterations = int(n_iterations)
        self.learning_rate = float(learning_rate)
        self.min_cross_region_rows = int(min_cross_region_rows)
        self.rating_scale = rating_scale

    @nw.narwhalify
    def optimize(self, df: IntoFrameT) -> LeagueStartRatingOptimizationResult:
        pl_df = df.to_native() if df.implementation.is_polars() else df.to_polars()
        league_ratings = self._get_league_ratings(self.rating_generator)
        iteration_errors: list[dict[str, float]] = []

        for _ in range(self.n_iterations):
            gen = copy.deepcopy(self.rating_generator)
            self._set_league_ratings(gen, league_ratings)
            self._ensure_prediction_columns(gen)

            pred_df = gen.fit_transform(pl_df)
            error_df = self._cross_region_error_df(pl_df, pred_df, gen)
            if error_df.is_empty():
                break

            error_summary = (
                error_df.group_by(self._league_column_name(gen))
                .agg(
                    pl.col("error").mean().alias("mean_error"),
                    pl.len().alias("row_count"),
                )
                .to_dicts()
            )
            league_key = self._league_column_name(gen)
            iteration_errors.append({r[league_key]: r["mean_error"] for r in error_summary})
            league_ratings = self._apply_error_updates(
                gen, league_ratings, error_summary, league_key
            )

        self._set_league_ratings(self.rating_generator, league_ratings)
        return LeagueStartRatingOptimizationResult(
            league_ratings=league_ratings, iteration_errors=iteration_errors
        )

    def _cross_region_error_df(
        self,
        df: pl.DataFrame,
        pred_df: pl.DataFrame,
        rating_generator: object,
    ) -> pl.DataFrame:
        column_names = getattr(rating_generator, "column_names", None)
        if column_names is None:
            raise ValueError("rating_generator must define column_names")

        match_id = getattr(column_names, "match_id", None)
        team_id = getattr(column_names, "team_id", None)
        league_col = getattr(column_names, "league", None)
        if not match_id or not team_id or not league_col:
            raise ValueError("column_names must include match_id, team_id, and league")

        pred_col, entity_cols, perf_col = self._prediction_spec(rating_generator)
        base_cols = [match_id, team_id, league_col, perf_col]
        for col in base_cols + entity_cols:
            if col not in df.columns:
                raise ValueError(f"{col} missing from input dataframe")

        join_cols = [match_id, team_id] + entity_cols
        joined = df.select(base_cols + entity_cols).join(
            pred_df.select(join_cols + [pred_col]),
            on=join_cols,
            how="inner",
        )
        opp_league = self._opponent_mode_league(joined, match_id, team_id, league_col)
        enriched = joined.join(opp_league, on=[match_id, team_id], how="left").with_columns(
            (pl.col(perf_col) - pl.col(pred_col)).alias("error")
        )
        return enriched.filter(pl.col("opp_mode_league").is_not_null()).filter(
            pl.col(league_col) != pl.col("opp_mode_league")
        )

    def _opponent_mode_league(
        self, df: pl.DataFrame, match_id: str, team_id: str, league_col: str
    ) -> pl.DataFrame:
        team_mode = (
            df.group_by([match_id, team_id, league_col])
            .agg(pl.len().alias("__count"))
            .sort(["__count"], descending=True)
            .unique([match_id, team_id])
            .select([match_id, team_id, league_col])
            .rename({league_col: "team_mode_league"})
        )
        opponents = (
            team_mode.join(team_mode, on=match_id, suffix="_opp")
            .filter(pl.col(team_id) != pl.col(f"{team_id}_opp"))
            .group_by([match_id, team_id, "team_mode_league_opp"])
            .agg(pl.len().alias("__count"))
            .sort(["__count"], descending=True)
            .unique([match_id, team_id])
            .select([match_id, team_id, "team_mode_league_opp"])
            .rename({"team_mode_league_opp": "opp_mode_league"})
        )
        return opponents

    def _prediction_spec(self, rating_generator: object) -> tuple[str, list[str], str]:
        perf_col = getattr(rating_generator, "performance_column", None)
        if not perf_col:
            raise ValueError("rating_generator must define performance_column")
        if hasattr(rating_generator, "PLAYER_PRED_PERF_COL"):
            pred_col = rating_generator.PLAYER_PRED_PERF_COL
            column_names = rating_generator.column_names
            player_id = getattr(column_names, "player_id", None)
            if not player_id:
                raise ValueError("column_names must include player_id for player ratings")
            return pred_col, [player_id], perf_col
        if hasattr(rating_generator, "TEAM_PRED_OFF_PERF_COL"):
            pred_col = rating_generator.TEAM_PRED_OFF_PERF_COL
            return pred_col, [], perf_col
        raise ValueError("rating_generator must expose a predicted performance column")

    def _ensure_prediction_columns(self, rating_generator: object) -> None:
        pred_cols: list[str] = []
        if hasattr(rating_generator, "PLAYER_PRED_PERF_COL"):
            pred_cols.append(rating_generator.PLAYER_PRED_PERF_COL)
        elif hasattr(rating_generator, "TEAM_PRED_OFF_PERF_COL"):
            pred_cols.append(rating_generator.TEAM_PRED_OFF_PERF_COL)

        if not pred_cols:
            return

        existing = list(getattr(rating_generator, "non_predictor_features_out", []) or [])
        for col in pred_cols:
            if col not in existing:
                existing.append(col)
        rating_generator.non_predictor_features_out = existing

    def _apply_error_updates(
        self,
        rating_generator: object,
        league_ratings: dict[str, float],
        error_summary: list[dict[str, float]],
        league_key: str,
    ) -> dict[str, float]:
        scale = self.rating_scale
        if scale is None:
            scale = getattr(rating_generator, "rating_change_multiplier_offense", 1.0)

        updated = dict(league_ratings)
        for row in error_summary:
            if row["row_count"] < self.min_cross_region_rows:
                continue
            league = row[league_key]
            mean_error = row["mean_error"]
            base_rating = updated.get(league, DEFAULT_START_RATING)
            updated[league] = base_rating + self.learning_rate * mean_error * scale
        return updated

    def _league_column_name(self, rating_generator: object) -> str:
        column_names = getattr(rating_generator, "column_names", None)
        league_col = getattr(column_names, "league", None)
        if not league_col:
            raise ValueError("column_names must include league for league adjustments")
        return league_col

    def _get_league_ratings(self, rating_generator: object) -> dict[str, float]:
        start_gen = getattr(rating_generator, "start_rating_generator", None)
        if start_gen is None or not hasattr(start_gen, "league_ratings"):
            raise ValueError("rating_generator must define start_rating_generator.league_ratings")
        return dict(start_gen.league_ratings)

    def _set_league_ratings(self, rating_generator: object, league_ratings: dict[str, float]) -> None:
        start_gen = getattr(rating_generator, "start_rating_generator", None)
        if start_gen is None or not hasattr(start_gen, "league_ratings"):
            raise ValueError("rating_generator must define start_rating_generator.league_ratings")
        start_gen.league_ratings = dict(league_ratings)
        if hasattr(rating_generator, "start_league_ratings"):
            rating_generator.start_league_ratings = dict(league_ratings)
