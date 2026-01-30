# player_rating_generator.py
from __future__ import annotations

import copy
import math
import logging
from typing import Any, Literal

import narwhals.stable.v2 as nw
import polars as pl
from narwhals.stable.v2 import DataFrame
from narwhals.stable.v2.typing import IntoFrameT

from spforge.data_structures import (
    ColumnNames,
    MatchPerformance,
    MatchPlayer,
    PlayerRating,
    PlayerRatingChange,
    PlayerRatingsResult,
    PreMatchPlayerRating,
    PreMatchPlayersCollection,
    PreMatchTeamRating,
)
from spforge.performance_transformers._performance_manager import ColumnWeight, PerformanceManager
from spforge.ratings._base import RatingGenerator, RatingKnownFeatures, RatingUnknownFeatures
from spforge.ratings.start_rating_generator import StartRatingGenerator
from spforge.ratings.utils import (
    add_opp_team_rating,
    add_rating_difference_projected,
    add_rating_mean_projected,
    add_team_rating,
    add_team_rating_projected,
)
from spforge.feature_generator._utils import to_polars

PLAYER_STATS = "__PLAYER_STATS"
_SCALED_PW = "__scaled_participation_weight__"
_SCALED_PPW = "__scaled_projected_participation_weight__"


class PlayerRatingGenerator(RatingGenerator):
    """
    Offense/Defense version:

    Player OFF rating is updated from the player's own performance_column.
    Player DEF rating is updated from a derived defensive performance:
        def_perf(team) = 1 - off_perf(opponent_team)
    which is then applied to all players on that team in that match.

    Predictions:
      - player offense vs opponent TEAM defense
      - player defense vs opponent TEAM offense
    """

    def __init__(
        self,
        performance_column: str,
        performance_weights: list[ColumnWeight | dict[str, float]] | None = None,
        performance_manager: PerformanceManager | None = None,
        auto_scale_performance: bool = False,
        performance_predictor: Literal["difference", "mean", "ignore_opponent"] = "difference",
        use_off_def_split: bool = True,
        rating_change_multiplier_offense: float = 50,
        rating_change_multiplier_defense: float = 50,
        confidence_days_ago_multiplier: float = 0.06,
        confidence_max_days: int = 90,
        confidence_value_denom: float = 140,
        confidence_max_sum: float = 150,
        confidence_weight: float = 0.9,
        features_out: list[RatingKnownFeatures] | None = None,
        non_predictor_features_out: list[RatingKnownFeatures | RatingUnknownFeatures] | None = None,
        min_rating_change_multiplier_ratio: float = 0.1,
        league_rating_change_update_threshold: float = 100,
        league_rating_adjustor_multiplier: float = 0.05,
        team_id_change_confidence_sum_decrease: float = 3,
        start_league_ratings: dict[str, float] | None = None,
        start_league_quantile: float = 0.2,
        start_min_count_for_percentiles: int = 50,
        start_team_rating_subtract: float = 80,
        start_team_weight: float = 0,
        start_max_days_ago_league_entities: int = 600,
        start_min_match_count_team_rating: int = 2,
        start_harcoded_start_rating: float | None = None,
        column_names: ColumnNames | None = None,
        output_suffix: str | None = None,
        scale_participation_weights: bool = False,
        auto_scale_participation_weights: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            performance_column=performance_column,
            column_names=column_names,
            output_suffix=output_suffix,
            rating_change_multiplier_offense=rating_change_multiplier_offense,
            rating_change_multiplier_defense=rating_change_multiplier_defense,
            confidence_days_ago_multiplier=confidence_days_ago_multiplier,
            confidence_max_days=confidence_max_days,
            confidence_value_denom=confidence_value_denom,
            confidence_max_sum=confidence_max_sum,
            confidence_weight=confidence_weight,
            min_rating_change_multiplier_ratio=min_rating_change_multiplier_ratio,
            team_id_change_confidence_sum_decrease=team_id_change_confidence_sum_decrease,
            features_out=features_out,  # unsuffixed (enums)
            performance_manager=performance_manager,
            auto_scale_performance=auto_scale_performance,
            performance_predictor=performance_predictor,
            performance_weights=performance_weights,
            non_predictor_features_out=non_predictor_features_out,  # unsuffixed (enums)
            league_rating_adjustor_multiplier=league_rating_adjustor_multiplier,
            league_rating_change_update_threshold=league_rating_change_update_threshold,
            **kwargs,
        )

        self.PLAYER_OFF_RATING_COL = self._suffix(str(RatingKnownFeatures.PLAYER_OFF_RATING))
        self.PLAYER_DEF_RATING_COL = self._suffix(str(RatingKnownFeatures.PLAYER_DEF_RATING))

        self.PLAYER_PRED_OFF_PERF_COL = self._suffix(
            str(RatingUnknownFeatures.PLAYER_PREDICTED_OFF_PERFORMANCE)
        )
        self.PLAYER_PRED_DEF_PERF_COL = self._suffix(
            str(RatingUnknownFeatures.PLAYER_PREDICTED_DEF_PERFORMANCE)
        )

        self.PLAYER_RATING_COL = self._suffix(str(RatingKnownFeatures.PLAYER_RATING))
        self.PLAYER_PRED_PERF_COL = self._suffix(
            str(RatingUnknownFeatures.PLAYER_PREDICTED_PERFORMANCE)
        )

        self.TEAM_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.TEAM_RATING_PROJECTED))
        self.OPP_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.OPPONENT_RATING_PROJECTED))
        self.DIFF_PROJ_COL = self._suffix(str(RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED))
        self.PLAYER_DIFF_PROJ_COL = self._suffix(
            str(RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED)
        )
        self.MEAN_PROJ_COL = self._suffix(str(RatingKnownFeatures.RATING_MEAN_PROJECTED))
        self.PLAYER_DIFF_FROM_TEAM_PROJ_COL = self._suffix(
            str(RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_FROM_TEAM_PROJECTED)
        )

        self.TEAM_OFF_RATING_PROJ_COL = self._suffix(
            str(RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED)
        )
        self.TEAM_DEF_RATING_PROJ_COL = self._suffix(
            str(RatingKnownFeatures.TEAM_DEF_RATING_PROJECTED)
        )
        self.OPP_OFF_RATING_PROJ_COL = self._suffix(
            str(RatingKnownFeatures.OPPONENT_OFF_RATING_PROJECTED)
        )
        self.OPP_DEF_RATING_PROJ_COL = self._suffix(
            str(RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED)
        )

        self.TEAM_RATING_COL = self._suffix(str(RatingUnknownFeatures.TEAM_RATING))
        self.OPP_RATING_COL = self._suffix(str(RatingUnknownFeatures.OPPONENT_RATING))
        self.DIFF_COL = self._suffix(str(RatingUnknownFeatures.TEAM_RATING_DIFFERENCE))

        self.start_league_ratings = start_league_ratings
        self.start_league_quantile = start_league_quantile
        self.start_min_count_for_percentiles = start_min_count_for_percentiles
        self.start_team_rating_subtract = start_team_rating_subtract
        self.start_team_weight = start_team_weight
        self.start_max_days_ago_league_entities = start_max_days_ago_league_entities
        self.start_min_match_count_team_rating = start_min_match_count_team_rating
        self.start_hardcoded_start_rating = start_harcoded_start_rating

        self.team_id_change_confidence_sum_decrease = team_id_change_confidence_sum_decrease
        self.column_names = column_names

        self.use_off_def_split = bool(use_off_def_split)
        self.scale_participation_weights = bool(scale_participation_weights)
        self.auto_scale_participation_weights = bool(auto_scale_participation_weights)
        self._participation_weight_max: float | None = None
        self._projected_participation_weight_max: float | None = None

        self._player_off_ratings: dict[str, PlayerRating] = {}
        self._player_def_ratings: dict[str, PlayerRating] = {}

        self.start_rating_generator = StartRatingGenerator(
            league_ratings=self.start_league_ratings,
            league_quantile=self.start_league_quantile,
            min_match_count_team_rating=self.start_min_match_count_team_rating,
            team_weight=self.start_team_weight,
            team_rating_subtract=self.start_team_rating_subtract,
            max_days_ago_league_entities=self.start_max_days_ago_league_entities,
            min_count_for_percentiles=self.start_min_count_for_percentiles,
            harcoded_start_rating=self.start_hardcoded_start_rating,
        )

    @to_polars
    @nw.narwhalify
    def fit_transform(
        self,
        df: IntoFrameT,
        column_names: ColumnNames | None = None,
    ) -> DataFrame | IntoFrameT:
        self.column_names = column_names if column_names else self.column_names
        self._maybe_enable_participation_weight_scaling(df)
        self._set_participation_weight_max(df)
        return super().fit_transform(df, column_names)

    def _maybe_enable_participation_weight_scaling(self, df: DataFrame) -> None:
        if self.scale_participation_weights or not self.auto_scale_participation_weights:
            return
        cn = self.column_names
        if not cn:
            return

        pl_df = df.to_native() if df.implementation.is_polars() else df.to_polars().to_native()

        def _out_of_bounds(col_name: str | None) -> bool:
            if not col_name or col_name not in df.columns:
                return False
            col = pl_df[col_name]
            min_val = col.min()
            max_val = col.max()
            if min_val is None or max_val is None:
                return False
            eps = 1e-6
            return min_val < -eps or max_val > (1.0 + eps)

        if _out_of_bounds(cn.participation_weight) or _out_of_bounds(
            cn.projected_participation_weight
        ):
            self.scale_participation_weights = True
            logging.warning(
                "Auto-scaling participation weights because values exceed [0, 1]. "
                "Set scale_participation_weights=True explicitly to silence this warning."
            )

    def _ensure_player_off(self, player_id: str) -> PlayerRating:
        if player_id not in self._player_off_ratings:
            # create with start generator later; initialize to 0 now; overwritten when needed
            self._player_off_ratings[player_id] = PlayerRating(id=player_id, rating_value=0.0)
        return self._player_off_ratings[player_id]

    def _ensure_player_def(self, player_id: str) -> PlayerRating:
        if player_id not in self._player_def_ratings:
            self._player_def_ratings[player_id] = PlayerRating(id=player_id, rating_value=0.0)
        return self._player_def_ratings[player_id]

    def _applied_multiplier_off(self, state: PlayerRating) -> float:
        return self._applied_multiplier(state, self.rating_change_multiplier_offense)

    def _applied_multiplier_def(self, state: PlayerRating) -> float:
        return self._applied_multiplier(state, self.rating_change_multiplier_defense)

    def _calculate_post_match_confidence_sum(
        self, entity_rating: PlayerRating, day_number: int, particpation_weight: float
    ) -> float:
        return self._post_match_confidence_sum(
            state=entity_rating,
            day_number=day_number,
            participation_weight=particpation_weight,
        )

    def _set_participation_weight_max(self, df: DataFrame) -> None:
        if not self.scale_participation_weights:
            return
        cn = self.column_names
        if not cn:
            return

        pl_df = df.to_native() if df.implementation.is_polars() else df.to_polars().to_native()

        if cn.participation_weight and cn.participation_weight in df.columns:
            q_val = pl_df[cn.participation_weight].quantile(0.99, "linear")
            if q_val is not None:
                self._participation_weight_max = float(q_val)

        if cn.projected_participation_weight and cn.projected_participation_weight in df.columns:
            q_val = pl_df[cn.projected_participation_weight].quantile(0.99, "linear")
            if q_val is not None:
                self._projected_participation_weight_max = float(q_val)
        elif self._participation_weight_max is not None:
            self._projected_participation_weight_max = self._participation_weight_max

    def _scale_participation_weight_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create internal scaled participation weight columns without mutating originals."""
        if not self.scale_participation_weights:
            return df
        if self._participation_weight_max is None or self._participation_weight_max <= 0:
            return df

        cn = self.column_names
        if not cn:
            return df

        if cn.participation_weight and cn.participation_weight in df.columns:
            denom = float(self._participation_weight_max)
            df = df.with_columns(
                (pl.col(cn.participation_weight) / denom)
                .clip(0.0, 1.0)
                .alias(_SCALED_PW)
            )

        if (
            cn.projected_participation_weight
            and cn.projected_participation_weight in df.columns
            and self._projected_participation_weight_max is not None
            and self._projected_participation_weight_max > 0
        ):
            denom = float(self._projected_participation_weight_max)
            df = df.with_columns(
                (pl.col(cn.projected_participation_weight) / denom)
                .clip(0.0, 1.0)
                .alias(_SCALED_PPW)
            )

        return df

    def _get_participation_weight_col(self) -> str:
        """Get the column name to use for participation weight (scaled if available)."""
        cn = self.column_names
        if self.scale_participation_weights and cn and cn.participation_weight:
            return _SCALED_PW
        return cn.participation_weight if cn else ""

    def _get_projected_participation_weight_col(self) -> str:
        """Get the column name to use for projected participation weight (scaled if available)."""
        cn = self.column_names
        if self.scale_participation_weights and cn and cn.projected_participation_weight:
            return _SCALED_PPW
        return cn.projected_participation_weight if cn else ""

    def _remove_internal_scaled_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove internal scaled columns before returning."""
        cols_to_drop = [c for c in [_SCALED_PW, _SCALED_PPW] if c in df.columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop)
        return df

    def _historical_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self._scale_participation_weight_columns(df)
        match_df = self._create_match_df(df)
        ratings = self._calculate_ratings(match_df)

        # Keep scaled columns for now - they're needed by _add_rating_features
        cols = [
            c
            for c in df.columns
            if c
            not in (
                self.PLAYER_OFF_RATING_COL,
                self.PLAYER_DEF_RATING_COL,
                self.PLAYER_PRED_OFF_PERF_COL,
                self.PLAYER_PRED_DEF_PERF_COL,
                self.PLAYER_RATING_COL,
                self.PLAYER_PRED_PERF_COL,
            )
        ]

        df = df.select(cols).join(
            ratings,
            on=[self.column_names.player_id, self.column_names.match_id, self.column_names.team_id],
        )

        result = self._add_rating_features(df)
        return self._remove_internal_scaled_columns(result)

    def _future_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self._scale_participation_weight_columns(df)
        match_df = self._create_match_df(df)
        ratings = self._calculate_future_ratings(match_df)

        # Keep scaled columns for now - they're needed by _add_rating_features
        cols = [
            c
            for c in df.columns
            if c
            not in (
                self.PLAYER_OFF_RATING_COL,
                self.PLAYER_DEF_RATING_COL,
                self.PLAYER_PRED_OFF_PERF_COL,
                self.PLAYER_PRED_DEF_PERF_COL,
                self.PLAYER_RATING_COL,
                self.PLAYER_PRED_PERF_COL,
            )
        ]

        df_with_ratings = df.select(cols).join(
            ratings,
            on=[
                self.column_names.player_id,
                self.column_names.match_id,
                self.column_names.team_id,
            ],
            how="left",
        )

        result = self._add_rating_features(df_with_ratings)
        return self._remove_internal_scaled_columns(result)

    def _calculate_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        pending_team_updates: list[tuple[str, str, float, float, int]] = []

        last_update_id = None

        out = {
            cn.player_id: [],
            cn.match_id: [],
            cn.team_id: [],
            self.PLAYER_OFF_RATING_COL: [],
            self.PLAYER_DEF_RATING_COL: [],
            self.PLAYER_PRED_OFF_PERF_COL: [],
            self.PLAYER_PRED_DEF_PERF_COL: [],
            self.PLAYER_RATING_COL: [],
            self.PLAYER_PRED_PERF_COL: [],
        }

        for r in match_df.iter_rows(named=True):
            update_id = r[cn.update_match_id]

            if last_update_id is not None and update_id != last_update_id:
                if pending_team_updates:
                    self._apply_player_updates(pending_team_updates)
                    pending_team_updates = []
                last_update_id = update_id
            day_number = int(r["__day_number"])

            team1 = r[cn.team_id]
            team2 = r[f"{cn.team_id}_opponent"]

            c1 = self._create_pre_match_players_collection(
                r=r, stats_col=PLAYER_STATS, day_number=day_number, team_id=team1
            )
            c2 = self._create_pre_match_players_collection(
                r=r, stats_col=f"{PLAYER_STATS}_opponent", day_number=day_number, team_id=team2
            )

            team1_off_perf = self._team_off_perf_from_collection(c1)
            team2_off_perf = self._team_off_perf_from_collection(c2)

            team1_def_perf: float | None = None
            team2_def_perf: float | None = None

            if self.use_off_def_split:
                team1_def_perf = (
                    1.0 - team2_off_perf if team2_off_perf is not None else None
                )
                team2_def_perf = (
                    1.0 - team1_off_perf if team1_off_perf is not None else None
                )
            else:
                team1_def_perf = team1_off_perf
                team2_def_perf = team2_off_perf

            team1_off_rating, team1_def_rating = self._team_off_def_rating_from_collection(c1)
            team2_off_rating, team2_def_rating = self._team_off_def_rating_from_collection(c2)

            player_updates: list[
                tuple[str, str, float, float, float, float, float, float, int, str | None]
            ] = []

            for pre_player in c1.pre_match_player_ratings:
                pid = pre_player.id

                off_state = self._player_off_ratings[pid]
                def_state = self._player_def_ratings[pid]

                off_pre = float(off_state.rating_value)
                def_pre = float(def_state.rating_value)

                pred_off = self._performance_predictor.predict_performance(
                    player_rating=pre_player,
                    opponent_team_rating=PreMatchTeamRating(
                        id=team2, players=[], rating_value=team2_def_rating
                    ),
                    team_rating=PreMatchTeamRating(
                        id=team1, players=[], rating_value=team1_off_rating
                    ),
                )

                pred_def = self._performance_predictor.predict_performance(
                    player_rating=PreMatchPlayerRating(
                        id=pid,
                        rating_value=def_pre,
                        match_performance=pre_player.match_performance,
                        games_played=pre_player.games_played,
                        league=pre_player.league,
                        position=pre_player.position,
                        other=getattr(pre_player, "other", None),
                    ),
                    opponent_team_rating=PreMatchTeamRating(
                        id=team2, players=[], rating_value=team2_off_rating
                    ),
                    team_rating=PreMatchTeamRating(
                        id=team1, players=[], rating_value=team1_def_rating
                    ),
                )

                perf_value = pre_player.match_performance.performance_value
                if perf_value is None:
                    off_change = 0.0
                else:
                    off_perf = float(perf_value)
                    mult_off = self._applied_multiplier_off(off_state)
                    off_change = (
                        (off_perf - float(pred_off))
                        * mult_off
                        * float(pre_player.match_performance.participation_weight)
                    )

                if perf_value is None or team1_def_perf is None:
                    def_change = 0.0
                else:
                    def_perf = float(team1_def_perf)

                    if not self.use_off_def_split:
                        pred_def = pred_off
                        def_perf = float(perf_value)

                    mult_def = self._applied_multiplier_def(def_state)
                    def_change = (
                        (def_perf - float(pred_def))
                        * mult_def
                        * float(pre_player.match_performance.participation_weight)
                    )

                if math.isnan(off_change) or math.isnan(def_change):
                    raise ValueError(
                        f"NaN player rating change for player_id={pid}, match_id={r[cn.match_id]}"
                    )

                player_updates.append(
                    (
                        pid,
                        team1,
                        off_pre,
                        def_pre,
                        float(pred_off),
                        float(pred_def),
                        float(off_change),
                        float(def_change),
                        day_number,
                        pre_player.league,
                    )
                )

            for pre_player in c2.pre_match_player_ratings:
                pid = pre_player.id

                off_state = self._player_off_ratings[pid]
                def_state = self._player_def_ratings[pid]

                off_pre = float(off_state.rating_value)
                def_pre = float(def_state.rating_value)

                pred_off = self._performance_predictor.predict_performance(
                    player_rating=pre_player,
                    opponent_team_rating=PreMatchTeamRating(
                        id=team1, players=[], rating_value=team1_def_rating
                    ),
                    team_rating=PreMatchTeamRating(
                        id=team2, players=[], rating_value=team2_off_rating
                    ),
                )

                pred_def = self._performance_predictor.predict_performance(
                    player_rating=PreMatchPlayerRating(
                        id=pid,
                        rating_value=def_pre,
                        match_performance=pre_player.match_performance,
                        games_played=pre_player.games_played,
                        league=pre_player.league,
                        position=pre_player.position,
                        other=getattr(pre_player, "other", None),
                    ),
                    opponent_team_rating=PreMatchTeamRating(
                        id=team1, players=[], rating_value=team1_off_rating
                    ),
                    team_rating=PreMatchTeamRating(
                        id=team2, players=[], rating_value=team2_def_rating
                    ),
                )

                perf_value = pre_player.match_performance.performance_value
                if perf_value is None:
                    off_change = 0.0
                else:
                    off_perf = float(perf_value)
                    mult_off = self._applied_multiplier_off(off_state)
                    off_change = (
                        (off_perf - float(pred_off))
                        * mult_off
                        * float(pre_player.match_performance.participation_weight)
                    )

                if perf_value is None or team2_def_perf is None:
                    def_change = 0.0
                else:
                    def_perf = float(team2_def_perf)

                    if not self.use_off_def_split:
                        pred_def = pred_off
                        def_perf = float(perf_value)

                    mult_def = self._applied_multiplier_def(def_state)
                    def_change = (
                        (def_perf - float(pred_def))
                        * mult_def
                        * float(pre_player.match_performance.participation_weight)
                    )

                if math.isnan(off_change) or math.isnan(def_change):
                    raise ValueError(
                        f"NaN player rating change for player_id={pid}, match_id={r[cn.match_id]}"
                    )

                player_updates.append(
                    (
                        pid,
                        team2,
                        off_pre,
                        def_pre,
                        float(pred_off),
                        float(pred_def),
                        float(off_change),
                        float(def_change),
                        day_number,
                        pre_player.league,
                    )
                )

            match_id = r[cn.match_id]
            for (
                pid,
                team_id,
                off_pre,
                def_pre,
                pred_off,
                pred_def,
                _off_change,
                _def_change,
                _dn,
                _league,
            ) in player_updates:
                out[cn.player_id].append(pid)
                out[cn.match_id].append(match_id)
                out[cn.team_id].append(team_id)

                out[self.PLAYER_OFF_RATING_COL].append(off_pre)
                out[self.PLAYER_DEF_RATING_COL].append(def_pre)
                out[self.PLAYER_PRED_OFF_PERF_COL].append(pred_off)
                out[self.PLAYER_PRED_DEF_PERF_COL].append(pred_def)

                out[self.PLAYER_RATING_COL].append(off_pre)
                out[self.PLAYER_PRED_PERF_COL].append(pred_off)

            for (
                pid,
                team_id,
                off_pre,
                _def_pre,
                _pred_off,
                _pred_def,
                off_change,
                def_change,
                dn,
                league,
            ) in player_updates:
                pending_team_updates.append(
                    (pid, team_id, off_pre, off_change, def_change, dn, league)
                )

            if last_update_id is None:
                last_update_id = update_id

        if pending_team_updates:
            self._apply_player_updates(pending_team_updates)

        return pl.DataFrame(out, strict=False)

    def _apply_player_updates(
        self, updates: list[tuple[str, str, float, float, float, int, str | None]]
    ) -> None:

        for player_id, team_id, pre_rating, off_change, def_change, day_number, league in updates:
            off_state = self._player_off_ratings[player_id]
            off_state.confidence_sum = self._calculate_post_match_confidence_sum(
                entity_rating=off_state,
                day_number=day_number,
                particpation_weight=1.0,
            )
            off_state.rating_value += float(off_change)
            off_state.games_played += 1.0
            off_state.last_match_day_number = int(day_number)
            off_state.most_recent_team_id = team_id

            def_state = self._player_def_ratings[player_id]
            def_state.confidence_sum = self._calculate_post_match_confidence_sum(
                entity_rating=def_state,
                day_number=day_number,
                particpation_weight=1.0,
            )
            def_state.rating_value += float(def_change)
            def_state.games_played += 1.0
            def_state.last_match_day_number = int(day_number)
            def_state.most_recent_team_id = team_id

            self.start_rating_generator.update_players_to_leagues(
                PlayerRatingChange(
                    id=player_id,
                    day_number=day_number,
                    league=league,
                    participation_weight=1.0,
                    predicted_performance=0.0,
                    performance=0.0,
                    pre_match_rating_value=pre_rating,
                    rating_change_value=off_change,
                )
            )

    def _add_rating_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cols_to_add = set((self._features_out or []) + (self.non_predictor_features_out or []))

        cn = self.column_names

        if (
            self.TEAM_OFF_RATING_PROJ_COL in cols_to_add
            or self.TEAM_RATING_PROJ_COL in cols_to_add
            or self.OPP_OFF_RATING_PROJ_COL in cols_to_add
            or self.OPP_RATING_PROJ_COL in cols_to_add
            or self.DIFF_PROJ_COL in cols_to_add
            or self.MEAN_PROJ_COL in cols_to_add
            or self.PLAYER_DIFF_FROM_TEAM_PROJ_COL in cols_to_add
        ):
            df = add_team_rating_projected(
                df=df,
                column_names=cn,
                player_rating_col=self.PLAYER_OFF_RATING_COL,
                team_rating_out=self.TEAM_OFF_RATING_PROJ_COL,
            )
            df = df.with_columns(
                pl.col(self.TEAM_OFF_RATING_PROJ_COL).alias(self.TEAM_RATING_PROJ_COL)
            )

        if (
            self.TEAM_DEF_RATING_PROJ_COL in cols_to_add
            or self.OPP_DEF_RATING_PROJ_COL in cols_to_add
            or self.OPP_RATING_PROJ_COL in cols_to_add
            or self.DIFF_PROJ_COL in cols_to_add
            or self.PLAYER_DIFF_PROJ_COL in cols_to_add
        ):
            df = add_team_rating_projected(
                df=df,
                column_names=cn,
                player_rating_col=self.PLAYER_DEF_RATING_COL,
                team_rating_out=self.TEAM_DEF_RATING_PROJ_COL,
            )

        if self.OPP_OFF_RATING_PROJ_COL in cols_to_add:
            df = add_opp_team_rating(
                df=df,
                column_names=cn,
                team_rating_col=self.TEAM_OFF_RATING_PROJ_COL,
                opp_team_rating_out=self.OPP_OFF_RATING_PROJ_COL,
            )

        if (
            self.OPP_DEF_RATING_PROJ_COL in cols_to_add
            or self.OPP_RATING_PROJ_COL in cols_to_add
            or self.DIFF_PROJ_COL in cols_to_add
            or self.PLAYER_DIFF_PROJ_COL in cols_to_add
        ):
            df = add_opp_team_rating(
                df=df,
                column_names=cn,
                team_rating_col=self.TEAM_DEF_RATING_PROJ_COL,
                opp_team_rating_out=self.OPP_DEF_RATING_PROJ_COL,
            )
            df = df.with_columns(
                pl.col(self.OPP_DEF_RATING_PROJ_COL).alias(self.OPP_RATING_PROJ_COL)
            )
        if self.PLAYER_DIFF_PROJ_COL in cols_to_add:
            df = df.with_columns(
                (pl.col(self.PLAYER_RATING_COL) - pl.col(self.OPP_DEF_RATING_PROJ_COL)).alias(
                    self.PLAYER_DIFF_PROJ_COL
                )
            )

        if self.PLAYER_DIFF_FROM_TEAM_PROJ_COL in cols_to_add:
            df = df.with_columns(
                (pl.col(self.PLAYER_OFF_RATING_COL) - pl.col(self.TEAM_OFF_RATING_PROJ_COL)).alias(
                    self.PLAYER_DIFF_FROM_TEAM_PROJ_COL
                )
            )

        if (
            self.TEAM_RATING_COL in cols_to_add
            or self.OPP_RATING_COL in cols_to_add
            or self.DIFF_COL in cols_to_add
        ):
            df = add_team_rating(
                df=df,
                column_names=cn,
                player_rating_col=self.PLAYER_OFF_RATING_COL,
                team_rating_out=self.TEAM_RATING_COL,
            )

        if self.OPP_RATING_COL in cols_to_add or self.DIFF_COL in cols_to_add:
            df = add_opp_team_rating(
                df=df,
                column_names=cn,
                team_rating_col=self.TEAM_RATING_COL,
                opp_team_rating_out=self.OPP_RATING_COL,
            )

        if self.DIFF_PROJ_COL in cols_to_add:
            df = add_rating_difference_projected(
                df=df,
                team_rating_col=self.TEAM_RATING_PROJ_COL,  # OFF
                opp_team_rating_col=self.OPP_RATING_PROJ_COL,  # DEF
                rating_diff_out=self.DIFF_PROJ_COL,
            )

        if self.MEAN_PROJ_COL in cols_to_add:
            df = add_rating_mean_projected(
                df=df,
                column_names=cn,
                player_rating_col=self.PLAYER_OFF_RATING_COL,
                rating_mean_out=self.MEAN_PROJ_COL,
            )

        if self.DIFF_COL in cols_to_add and self.DIFF_COL not in df.columns:
            if self.TEAM_RATING_COL not in df.columns:
                df = add_team_rating(
                    df=df,
                    column_names=cn,
                    player_rating_col=self.PLAYER_OFF_RATING_COL,
                    team_rating_out=self.TEAM_RATING_COL,
                )
            if self.OPP_RATING_COL not in df.columns:
                df = add_opp_team_rating(
                    df=df,
                    column_names=cn,
                    team_rating_col=self.TEAM_RATING_COL,
                    opp_team_rating_out=self.OPP_RATING_COL,
                )
            if self.TEAM_RATING_COL in df.columns and self.OPP_RATING_COL in df.columns:
                df = df.with_columns(
                    (pl.col(self.TEAM_RATING_COL) - pl.col(self.OPP_RATING_COL)).alias(
                        self.DIFF_COL
                    )
                )

        base_known = [f.value for f in RatingKnownFeatures]
        base_unknown = [f.value for f in RatingUnknownFeatures]
        cols_to_eval = [self._suffix(c) for c in (base_known + base_unknown)]

        cols_to_drop = [c for c in cols_to_eval if (c in df.columns and c not in cols_to_add)]
        return df.drop(cols_to_drop)

    def _create_match_df(self, df: pl.DataFrame) -> pl.DataFrame:
        if len(df[self.column_names.team_id].unique()) < 2:
            raise ValueError("df needs at least two different team ids")
        if self.league_identifier:
            df = self.league_identifier.add_leagues(df)

        cn = self.column_names

        player_stat_cols = [cn.player_id]
        if self.performance_column in df.columns:
            player_stat_cols.append(self.performance_column)

        if cn.participation_weight and cn.participation_weight in df.columns:
            player_stat_cols.append(cn.participation_weight)
        if _SCALED_PW in df.columns:
            player_stat_cols.append(_SCALED_PW)

        if cn.projected_participation_weight and cn.projected_participation_weight in df.columns:
            player_stat_cols.append(cn.projected_participation_weight)
        if _SCALED_PPW in df.columns:
            player_stat_cols.append(_SCALED_PPW)

        if cn.position and cn.position in df.columns:
            player_stat_cols.append(cn.position)

        if cn.league and cn.league in df.columns:
            player_stat_cols.append(cn.league)

        df = df.with_columns(pl.struct(player_stat_cols).alias(PLAYER_STATS))

        group_cols = [cn.match_id, cn.team_id, cn.start_date]
        if cn.update_match_id != cn.match_id:
            group_cols.append(cn.update_match_id)

        agg_df = df.group_by(group_cols).agg(pl.col(PLAYER_STATS).alias(PLAYER_STATS))

        team_id_col = cn.team_id

        right_cols = [
            c for c in agg_df.columns if c not in [cn.match_id, cn.start_date, cn.update_match_id]
        ]
        right_df = agg_df.select(
            [cn.match_id] + [pl.col(c).alias(f"{c}_opponent") for c in right_cols]
        )
        match_df = (
            agg_df.join(right_df, on=cn.match_id, how="inner")
            .filter(pl.col(team_id_col) != pl.col(f"{team_id_col}_opponent"))
            .unique(cn.match_id)
        )

        sort_cols = [cn.start_date, cn.match_id]
        if cn.update_match_id != cn.match_id and cn.update_match_id in match_df.columns:
            sort_cols.append(cn.update_match_id)
        match_df = match_df.sort(sort_cols)

        match_df = self._add_day_number(match_df, cn.start_date, "__day_number")
        return match_df

    def _create_pre_match_players_collection(
        self, r: dict, stats_col: str, day_number: int, team_id: str
    ) -> PreMatchPlayersCollection:
        cn = self.column_names

        pre_match_player_ratings: list[PreMatchPlayerRating] = []
        new_players: list[MatchPlayer] = []
        player_ids: list[str] = []
        player_off_rating_values: list[float] = []
        projected_participation_weights: list[float] = []

        for team_player in r[stats_col]:
            player_id = team_player[cn.player_id]
            player_ids.append(player_id)

            position = team_player.get(cn.position)
            player_league = team_player.get(cn.league, None)

            # Use scaled participation weight if available, otherwise use original
            if _SCALED_PW in team_player:
                participation_weight = team_player.get(_SCALED_PW, 1.0)
            elif cn.participation_weight:
                participation_weight = team_player.get(cn.participation_weight, 1.0)
            else:
                participation_weight = 1.0

            # Use scaled projected participation weight if available, otherwise use original
            if _SCALED_PPW in team_player:
                projected_participation_weight = team_player.get(_SCALED_PPW, participation_weight)
            elif cn.projected_participation_weight:
                projected_participation_weight = team_player.get(
                    cn.projected_participation_weight, participation_weight
                )
            else:
                projected_participation_weight = participation_weight
            projected_participation_weights.append(projected_participation_weight)

            perf_val = (
                float(team_player[self.performance_column])
                if (
                    self.performance_column in team_player
                    and team_player[self.performance_column] is not None
                )
                else None
            )

            mp = MatchPerformance(
                performance_value=perf_val,
                projected_participation_weight=projected_participation_weight,
                participation_weight=participation_weight,
            )

            if player_id in self._player_off_ratings and player_id in self._player_def_ratings:
                off_state = self._player_off_ratings[player_id]
                pre = PreMatchPlayerRating(
                    id=player_id,
                    rating_value=off_state.rating_value,
                    match_performance=mp,
                    games_played=off_state.games_played,
                    league=player_league,
                    position=position,
                )
                pre_match_player_ratings.append(pre)
                player_off_rating_values.append(float(off_state.rating_value))
            else:
                # unseen player -> create start rating (OFF + DEF)
                new_players.append(
                    MatchPlayer(
                        id=player_id,
                        performance=mp,
                        league=player_league,
                        position=position,
                    )
                )

        if new_players:
            new_pre, new_vals = self._generate_new_player_pre_match_ratings(
                day_number=day_number,
                new_players=new_players,
                team_pre_match_player_ratings=pre_match_player_ratings,
            )
            pre_match_player_ratings.extend(new_pre)
            player_off_rating_values.extend(new_vals)

        return PreMatchPlayersCollection(
            pre_match_player_ratings=pre_match_player_ratings,
            new_players=[],  # already handled
            player_ids=player_ids,
            player_rating_values=player_off_rating_values,  # OFF values
            projected_particiation_weights=projected_participation_weights,
        )

    def _generate_new_player_pre_match_ratings(
        self,
        day_number: int,
        new_players: list[MatchPlayer],
        team_pre_match_player_ratings: list[PreMatchPlayerRating],
    ) -> tuple[list[PreMatchPlayerRating], list[float]]:
        """
        Creates BOTH off+def states for new players using the same start rating.
        """
        pre_match_player_ratings: list[PreMatchPlayerRating] = []
        pre_match_player_off_values: list[float] = []

        for match_player in new_players:
            pid = match_player.id

            start_val = self.start_rating_generator.generate_rating_value(
                day_number=day_number,
                match_player=match_player,
                team_pre_match_player_ratings=team_pre_match_player_ratings,
            )

            self._player_off_ratings[pid] = PlayerRating(id=pid, rating_value=start_val)
            self._player_def_ratings[pid] = PlayerRating(id=pid, rating_value=start_val)

            pre_match_player_off_values.append(float(start_val))

            pre = PreMatchPlayerRating(
                id=pid,
                rating_value=float(start_val),  # OFF rating value for predictor
                match_performance=match_player.performance,
                games_played=self._player_off_ratings[pid].games_played,
                league=match_player.league,
                position=match_player.position,
                other=match_player.others,
            )
            pre_match_player_ratings.append(pre)

        return pre_match_player_ratings, pre_match_player_off_values

    def _team_off_perf_from_collection(
        self, c: PreMatchPlayersCollection
    ) -> float | None:
        # observed offense perf = weighted mean of player performance_value using participation_weight if present
        # skip players with null performance
        cn = self.column_names
        if not c.pre_match_player_ratings:
            return None
        wsum = 0.0
        psum = 0.0
        for pre in c.pre_match_player_ratings:
            perf_val = pre.match_performance.performance_value
            if perf_val is None:
                continue
            w = (
                float(pre.match_performance.participation_weight)
                if cn.participation_weight
                else 1.0
            )
            psum += float(perf_val) * w
            wsum += w
        return psum / wsum if wsum else None

    def _team_off_def_rating_from_collection(
        self, c: PreMatchPlayersCollection
    ) -> tuple[float, float]:
        cn = self.column_names
        if not c.player_ids:
            return 0.0, 0.0

        off_vals = [
            float(self._player_off_ratings[pid].rating_value)
            for pid in c.player_ids
            if pid in self._player_off_ratings
        ]
        def_vals = off_vals if not self.use_off_def_split else [
            float(self._player_def_ratings[pid].rating_value)
            for pid in c.player_ids
            if pid in self._player_def_ratings
        ]
        if not off_vals or not def_vals:
            return 0.0, 0.0

        if cn.projected_participation_weight and c.projected_particiation_weights:
            w = [float(x) for x in c.projected_particiation_weights]
            wsum = sum(w) if w else 0.0
            off = (
                (sum(v * ww for v, ww in zip(off_vals, w, strict=False)) / wsum)
                if wsum
                else (sum(off_vals) / len(off_vals))
            )
            dff = (
                (sum(v * ww for v, ww in zip(def_vals, w, strict=False)) / wsum)
                if wsum
                else (sum(def_vals) / len(def_vals))
            )
            return float(off), float(dff)

        return float(sum(off_vals) / len(off_vals)), float(sum(def_vals) / len(def_vals))

    def _calculate_future_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        """
        Same as historical, but:
        - do NOT mutate self._player_*_ratings
        - create local-only start ratings for unseen players
        """
        cn = self.column_names

        base_off = self._player_off_ratings
        base_def = self._player_def_ratings
        local_off: dict[str, PlayerRating] = copy.deepcopy(base_off)
        local_def: dict[str, PlayerRating] = copy.deepcopy(base_def)

        out = {
            cn.player_id: [],
            cn.match_id: [],
            cn.team_id: [],
            self.PLAYER_OFF_RATING_COL: [],
            self.PLAYER_DEF_RATING_COL: [],
            self.PLAYER_PRED_OFF_PERF_COL: [],
            self.PLAYER_PRED_DEF_PERF_COL: [],
            # back-compat
            self.PLAYER_RATING_COL: [],
            self.PLAYER_PRED_PERF_COL: [],
        }

        def get_perf_value(team_player: dict) -> float | None:
            if (
                self.performance_column in team_player
                and team_player[self.performance_column] is not None
            ):
                return float(team_player[self.performance_column])
            return None

        def ensure_new_player(
            pid: str,
            day_number: int,
            mp: MatchPerformance,
            league,
            position,
            team_pre: list[PreMatchPlayerRating],
        ) -> None:
            if pid in local_off and pid in local_def:
                return
            start_val = self.start_rating_generator.generate_rating_value(
                day_number=day_number,
                match_player=MatchPlayer(id=pid, performance=mp, league=league, position=position),
                team_pre_match_player_ratings=team_pre,
            )
            local_off[pid] = PlayerRating(id=pid, rating_value=start_val)
            local_def[pid] = PlayerRating(id=pid, rating_value=start_val)

        for r in match_df.iter_rows(named=True):
            match_id = r[cn.match_id]
            day_number = int(r["__day_number"])

            team1 = r[cn.team_id]
            team2 = r[f"{cn.team_id}_opponent"]

            def build_local_team(
                stats_col: str,
            ) -> tuple[list[PreMatchPlayerRating], list[str], list[float], list[float], float]:
                pre_list: list[PreMatchPlayerRating] = []
                player_ids: list[str] = []
                proj_w: list[float] = []
                off_vals: list[float] = []
                psum, wsum = 0.0, 0.0

                for tp in r[stats_col]:  # noqa: B023
                    pid = tp[cn.player_id]
                    player_ids.append(pid)

                    position = tp.get(cn.position)
                    league = tp.get(cn.league, None)

                    # Use scaled participation weight if available, otherwise use original
                    if _SCALED_PW in tp:
                        pw = tp.get(_SCALED_PW, 1.0)
                    elif cn.participation_weight:
                        pw = tp.get(cn.participation_weight, 1.0)
                    else:
                        pw = 1.0

                    # Use scaled projected participation weight if available, otherwise use original
                    if _SCALED_PPW in tp:
                        ppw = tp.get(_SCALED_PPW, pw)
                    elif cn.projected_participation_weight:
                        ppw = tp.get(cn.projected_participation_weight, pw)
                    else:
                        ppw = pw
                    proj_w.append(float(ppw))

                    mp = MatchPerformance(
                        performance_value=get_perf_value(tp),
                        projected_participation_weight=ppw,
                        participation_weight=pw,
                    )

                    ensure_new_player(pid, day_number, mp, league, position, pre_list)  # noqa: B023

                    pre_list.append(
                        PreMatchPlayerRating(
                            id=pid,
                            rating_value=float(local_off[pid].rating_value),
                            match_performance=mp,
                            games_played=float(local_off[pid].games_played),
                            league=league,
                            position=position,
                        )
                    )
                    off_vals.append(float(local_off[pid].rating_value))

                    if mp.performance_value is not None:
                        psum += float(mp.performance_value) * float(pw)
                        wsum += float(pw)

                team_off_perf = psum / wsum if wsum else 0.0
                return pre_list, player_ids, off_vals, proj_w, team_off_perf

            t1_pre, t1_ids, t1_off_vals, t1_proj_w, t1_off_perf = build_local_team(PLAYER_STATS)
            t2_pre, t2_ids, t2_off_vals, t2_proj_w, t2_off_perf = build_local_team(
                f"{PLAYER_STATS}_opponent"
            )

            def team_off_def_rating(ids: list[str], w: list[float]) -> tuple[float, float]:
                if not ids:
                    return 0.0, 0.0
                off = [float(local_off[pid].rating_value) for pid in ids]
                dff = [float(local_def[pid].rating_value) for pid in ids]
                if cn.projected_participation_weight and w and sum(w):
                    ws = sum(w)
                    return (
                        sum(v * ww for v, ww in zip(off, w, strict=False)) / ws,
                        sum(v * ww for v, ww in zip(dff, w, strict=False)) / ws,
                    )
                return sum(off) / len(off), sum(dff) / len(dff)

            t1_off_rating, t1_def_rating = team_off_def_rating(t1_ids, t1_proj_w)
            t2_off_rating, t2_def_rating = team_off_def_rating(t2_ids, t2_proj_w)

            for pre in t1_pre:
                pid = pre.id
                off_pre = float(local_off[pid].rating_value)
                def_pre = off_pre if not self.use_off_def_split else float(local_def[pid].rating_value)

                pred_off = self._performance_predictor.predict_performance(
                    player_rating=pre,
                    opponent_team_rating=PreMatchTeamRating(
                        id=team2, players=[], rating_value=t2_def_rating
                    ),
                    team_rating=PreMatchTeamRating(
                        id=team1, players=[], rating_value=t1_off_rating
                    ),
                )

                pred_def = self._performance_predictor.predict_performance(
                    player_rating=PreMatchPlayerRating(
                        id=pid,
                        rating_value=def_pre,
                        match_performance=pre.match_performance,
                        games_played=pre.games_played,
                        league=pre.league,
                        position=pre.position,
                    ),
                    opponent_team_rating=PreMatchTeamRating(
                        id=team2, players=[], rating_value=t2_off_rating
                    ),
                    team_rating=PreMatchTeamRating(
                        id=team1, players=[], rating_value=t1_def_rating
                    ),
                )

                if not self.use_off_def_split:
                    pred_def = pred_off

                out[cn.player_id].append(pid)
                out[cn.match_id].append(match_id)
                out[cn.team_id].append(team1)
                out[self.PLAYER_OFF_RATING_COL].append(off_pre)
                out[self.PLAYER_DEF_RATING_COL].append(def_pre)
                out[self.PLAYER_PRED_OFF_PERF_COL].append(float(pred_off))
                out[self.PLAYER_PRED_DEF_PERF_COL].append(float(pred_def))
                out[self.PLAYER_RATING_COL].append(off_pre)
                out[self.PLAYER_PRED_PERF_COL].append(float(pred_off))

            for pre in t2_pre:
                pid = pre.id
                off_pre = float(local_off[pid].rating_value)
                def_pre = off_pre if not self.use_off_def_split else float(local_def[pid].rating_value)

                pred_off = self._performance_predictor.predict_performance(
                    player_rating=pre,
                    opponent_team_rating=PreMatchTeamRating(
                        id=team1, players=[], rating_value=t1_def_rating
                    ),
                    team_rating=PreMatchTeamRating(
                        id=team2, players=[], rating_value=t2_off_rating
                    ),
                )

                pred_def = self._performance_predictor.predict_performance(
                    player_rating=PreMatchPlayerRating(
                        id=pid,
                        rating_value=def_pre,
                        match_performance=pre.match_performance,
                        games_played=pre.games_played,
                        league=pre.league,
                        position=pre.position,
                    ),
                    opponent_team_rating=PreMatchTeamRating(
                        id=team1, players=[], rating_value=t1_off_rating
                    ),
                    team_rating=PreMatchTeamRating(
                        id=team2, players=[], rating_value=t2_def_rating
                    ),
                )

                if not self.use_off_def_split:
                    pred_def = pred_off

                out[cn.player_id].append(pid)
                out[cn.match_id].append(match_id)
                out[cn.team_id].append(team2)
                out[self.PLAYER_OFF_RATING_COL].append(off_pre)
                out[self.PLAYER_DEF_RATING_COL].append(def_pre)
                out[self.PLAYER_PRED_OFF_PERF_COL].append(float(pred_off))
                out[self.PLAYER_PRED_DEF_PERF_COL].append(float(pred_def))
                out[self.PLAYER_RATING_COL].append(off_pre)
                out[self.PLAYER_PRED_PERF_COL].append(float(pred_off))

        if not out[cn.player_id]:
            return pl.DataFrame(
                {
                    cn.player_id: pl.Series([], dtype=pl.Utf8),
                    cn.match_id: pl.Series([], dtype=pl.Utf8),
                    cn.team_id: pl.Series([], dtype=pl.Utf8),
                    self.PLAYER_OFF_RATING_COL: pl.Series([], dtype=pl.Float64),
                    self.PLAYER_DEF_RATING_COL: pl.Series([], dtype=pl.Float64),
                    self.PLAYER_PRED_OFF_PERF_COL: pl.Series([], dtype=pl.Float64),
                    self.PLAYER_PRED_DEF_PERF_COL: pl.Series([], dtype=pl.Float64),
                    self.PLAYER_RATING_COL: pl.Series([], dtype=pl.Float64),
                    self.PLAYER_PRED_PERF_COL: pl.Series([], dtype=pl.Float64),
                }
            )

        return pl.DataFrame(out, strict=False)

    @property
    def player_ratings(self) -> dict[str, PlayerRatingsResult]:
        """Return combined offense and defense ratings for all players."""
        result: dict[str, PlayerRatingsResult] = {}
        all_player_ids = set(self._player_off_ratings.keys()) | set(self._player_def_ratings.keys())

        for player_id in all_player_ids:
            off_state = self._player_off_ratings.get(player_id)
            def_state = self._player_def_ratings.get(player_id)

            result[player_id] = PlayerRatingsResult(
                id=player_id,
                offense_rating=off_state.rating_value if off_state else 0.0,
                defense_rating=def_state.rating_value if def_state else 0.0,
                offense_games_played=off_state.games_played if off_state else 0.0,
                defense_games_played=def_state.games_played if def_state else 0.0,
                offense_confidence_sum=off_state.confidence_sum if off_state else 0.0,
                defense_confidence_sum=def_state.confidence_sum if def_state else 0.0,
                most_recent_team_id=off_state.most_recent_team_id if off_state else None,
            )

        return result
