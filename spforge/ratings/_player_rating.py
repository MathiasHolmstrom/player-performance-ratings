# player_rating_generator.py
from __future__ import annotations

import copy
import math
from typing import Optional, Union, Literal, Any

import polars as pl

from spforge.data_structures import (
    PlayerRating,
    ColumnNames,
    PlayerRatingChange,
    TeamRatingChange,
    PreMatchTeamRating,
    PreMatchPlayerRating,
    PreMatchPlayersCollection,
    MatchPerformance,
    MatchPlayer,
)
from spforge.ratings import RatingKnownFeatures, RatingUnknownFeatures
from spforge.ratings._base import RatingGenerator
from spforge.ratings.start_rating_generator import StartRatingGenerator
from spforge.ratings.utils import (
    add_opp_team_rating,
    add_team_rating,
    add_team_rating_projected,
    add_rating_difference_projected,
    add_rating_mean_projected,
)
from spforge.transformers.fit_transformers._performance_manager import ColumnWeight, PerformanceManager

PLAYER_STATS = "__PLAYER_STATS"


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
        performance_weights: Optional[list[Union[ColumnWeight, dict[str, float]]]] = None,
        performance_manager: PerformanceManager | None = None,
        auto_scale_performance: bool = False,
        performance_predictor: Literal["difference", "mean", "ignore_opponent"] = "difference",
        # NEW: separate multipliers
        rating_change_multiplier_offense: float = 50,
        rating_change_multiplier_defense: float = 50,
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
        team_id_change_confidence_sum_decrease: float = 3,
        start_league_ratings: dict[str, float] | None = None,
        start_league_quantile: float = 0.2,
        start_min_count_for_percentiles: int = 50,
        start_team_rating_subtract: float = 80,
        start_team_weight: float = 0,
        start_max_days_ago_league_entities: int = 120,
        start_min_match_count_team_rating: int = 2,
        start_harcoded_start_rating: float | None = None,
        column_names: Optional[ColumnNames] = None,
        output_suffix: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            performance_column=performance_column,
            column_names=column_names,
            output_suffix=output_suffix,
            # NEW base class args:
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

        # --- Player OFF/DEF rating output cols (requires new enum values) ---
        # You should add these enum values:
        #   RatingKnownFeatures.PLAYER_OFF_RATING = "player_off_rating"
        #   RatingKnownFeatures.PLAYER_DEF_RATING = "player_def_rating"
        # And optionally predicted perf:
        #   RatingUnknownFeatures.PLAYER_PREDICTED_OFF_PERFORMANCE = "player_predicted_off_performance"
        #   RatingUnknownFeatures.PLAYER_PREDICTED_DEF_PERFORMANCE = "player_predicted_def_performance"

        self.PLAYER_OFF_RATING_COL = self._suffix(str(RatingKnownFeatures.PLAYER_OFF_RATING))
        self.PLAYER_DEF_RATING_COL = self._suffix(str(RatingKnownFeatures.PLAYER_DEF_RATING))

        self.PLAYER_PRED_OFF_PERF_COL = self._suffix(str(RatingUnknownFeatures.PLAYER_PREDICTED_OFF_PERFORMANCE))
        self.PLAYER_PRED_DEF_PERF_COL = self._suffix(str(RatingUnknownFeatures.PLAYER_PREDICTED_DEF_PERFORMANCE))

        # Back-compat: if someone still wants PLAYER_RATING, treat as OFF rating
        self.PLAYER_RATING_COL = self._suffix(str(RatingKnownFeatures.PLAYER_RATING))
        self.PLAYER_PRED_PERF_COL = self._suffix(str(RatingUnknownFeatures.PLAYER_PREDICTED_PERFORMANCE))

        # Team-level (projected) columns (back-compat semantics)
        # TEAM_RATING_PROJECTED := TEAM_OFF_RATING_PROJECTED
        # OPPONENT_RATING_PROJECTED := OPPONENT_DEF_RATING_PROJECTED
        self.TEAM_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.TEAM_RATING_PROJECTED))
        self.OPP_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.OPPONENT_RATING_PROJECTED))
        self.DIFF_PROJ_COL = self._suffix(str(RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED))
        self.MEAN_PROJ_COL = self._suffix(str(RatingKnownFeatures.RATING_MEAN_PROJECTED))

        # Extra explicit team off/def projected (requires new enum values if you want them exposed)
        self.TEAM_OFF_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED))
        self.TEAM_DEF_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.TEAM_DEF_RATING_PROJECTED))
        self.OPP_OFF_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.OPPONENT_OFF_RATING_PROJECTED))
        self.OPP_DEF_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED))

        # Unknown (unprojected) team cols (optional)
        self.TEAM_RATING_COL = self._suffix(str(RatingUnknownFeatures.TEAM_RATING))
        self.OPP_RATING_COL = self._suffix(str(RatingUnknownFeatures.OPPONENT_RATING))
        self.DIFF_COL = self._suffix(str(RatingUnknownFeatures.RATING_DIFFERENCE))

        # Start rating machinery
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

        # NEW: two states per player
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

    # --------------------
    # Rating state helpers
    # --------------------
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

    def _calculate_post_match_confidence_sum(self, entity_rating: PlayerRating, day_number: int, particpation_weight: float) -> float:
        return self._post_match_confidence_sum(
            state=entity_rating,
            day_number=day_number,
            participation_weight=particpation_weight,
        )

    # --------------------
    # Transforms
    # --------------------
    def _historical_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        match_df = self._create_match_df(df)
        ratings = self._calculate_ratings(match_df)

        cols = [c for c in df.columns if c not in (
            self.PLAYER_OFF_RATING_COL,
            self.PLAYER_DEF_RATING_COL,
            self.PLAYER_PRED_OFF_PERF_COL,
            self.PLAYER_PRED_DEF_PERF_COL,
            self.PLAYER_RATING_COL,
            self.PLAYER_PRED_PERF_COL,
        )]

        df = df.select(cols).join(
            ratings,
            on=[self.column_names.player_id, self.column_names.match_id, self.column_names.team_id],
        )

        return self._add_rating_features(df)

    def _future_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        match_df = self._create_match_df(df)
        ratings = self._calculate_future_ratings(match_df)

        cols = [c for c in df.columns if c not in (
            self.PLAYER_OFF_RATING_COL,
            self.PLAYER_DEF_RATING_COL,
            self.PLAYER_PRED_OFF_PERF_COL,
            self.PLAYER_PRED_DEF_PERF_COL,
            self.PLAYER_RATING_COL,
            self.PLAYER_PRED_PERF_COL,
        )]

        df = df.select(cols).join(
            ratings,
            on=[self.column_names.player_id, self.column_names.match_id, self.column_names.team_id],
        )

        return self._add_rating_features(df)

    # --------------------
    # Core rating logic
    # --------------------
    def _calculate_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        pending_team_updates: list[tuple[str, str, float, float, int]] = []
        # tuple(kind, player_id, off_change, def_change, day_number) isn't right at team-level,
        # we store (player_id, team_id, off_change, def_change, day_number)
        last_update_id = None

        out = {
            cn.player_id: [],
            cn.match_id: [],
            cn.team_id: [],
            self.PLAYER_OFF_RATING_COL: [],
            self.PLAYER_DEF_RATING_COL: [],
            self.PLAYER_PRED_OFF_PERF_COL: [],
            self.PLAYER_PRED_DEF_PERF_COL: [],
            # back-compat:
            self.PLAYER_RATING_COL: [],
            self.PLAYER_PRED_PERF_COL: [],
        }

        for r in match_df.iter_rows(named=True):
            update_id = r[cn.update_match_id]
            day_number = int(r["__day_number"])

            team1 = r[cn.team_id]
            team2 = r[f"{cn.team_id}_opponent"]

            # Build pre-match collections
            c1 = self._create_pre_match_players_collection(r=r, stats_col=PLAYER_STATS, day_number=day_number, team_id=team1)
            c2 = self._create_pre_match_players_collection(r=r, stats_col=f"{PLAYER_STATS}_opponent", day_number=day_number, team_id=team2)

            # Team offense performance (observed): weighted mean of player performance_column
            team1_off_perf = self._team_off_perf_from_collection(c1)
            team2_off_perf = self._team_off_perf_from_collection(c2)

            # Team defense performance (observed): 1 - opponent offense perf
            team1_def_perf = 1.0 - team2_off_perf
            team2_def_perf = 1.0 - team1_off_perf

            # Team projected OFF/DEF ratings (from player OFF/DEF ratings)
            team1_off_rating, team1_def_rating = self._team_off_def_rating_from_collection(c1)
            team2_off_rating, team2_def_rating = self._team_off_def_rating_from_collection(c2)

            pre_match_team_ratings = [
                PreMatchTeamRating(id=team1, players=c1.pre_match_player_ratings, rating_value=team1_off_rating),
                PreMatchTeamRating(id=team2, players=c2.pre_match_player_ratings, rating_value=team2_off_rating),
            ]

            # For each player, compute predicted OFF and predicted DEF performance and rating changes.
            # Prediction uses the shared predictor but with "opponent team rating" set to the relevant TEAM unit:
            # - OFF: opponent team DEF rating
            # - DEF: opponent team OFF rating
            player_updates: list[tuple[str, str, float, float, float, float, float, float, int]] = []
            # tuple: (player_id, team_id, off_pre, def_pre, pred_off, pred_def, off_change, def_change, day_number)

            # Team1 players vs Team2
            for pre_player in c1.pre_match_player_ratings:
                pid = pre_player.id

                off_state = self._player_off_ratings[pid]
                def_state = self._player_def_ratings[pid]

                off_pre = float(off_state.rating_value)
                def_pre = float(def_state.rating_value)

                # predicted OFF: player OFF vs opponent TEAM DEF
                pred_off = self._performance_predictor.predict_performance(
                    player_rating=pre_player,
                    opponent_team_rating=PreMatchTeamRating(id=team2, players=[], rating_value=team2_def_rating),
                    team_rating=PreMatchTeamRating(id=team1, players=[], rating_value=team1_off_rating),
                )

                # predicted DEF: player DEF vs opponent TEAM OFF
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
                    opponent_team_rating=PreMatchTeamRating(id=team2, players=[], rating_value=team2_off_rating),
                    team_rating=PreMatchTeamRating(id=team1, players=[], rating_value=team1_def_rating),
                )

                off_perf = float(pre_player.match_performance.performance_value)
                def_perf = float(team1_def_perf)  # same for all players on team1 (derived)

                mult_off = self._applied_multiplier_off(off_state)
                mult_def = self._applied_multiplier_def(def_state)

                off_change = (off_perf - float(pred_off)) * mult_off * float(pre_player.match_performance.participation_weight)
                def_change = (def_perf - float(pred_def)) * mult_def * float(pre_player.match_performance.participation_weight)

                if math.isnan(off_change) or math.isnan(def_change):
                    raise ValueError(f"NaN player rating change for player_id={pid}, match_id={r[cn.match_id]}")

                player_updates.append((pid, team1, off_pre, def_pre, float(pred_off), float(pred_def), float(off_change), float(def_change), day_number))

            # Team2 players vs Team1
            for pre_player in c2.pre_match_player_ratings:
                pid = pre_player.id

                off_state = self._player_off_ratings[pid]
                def_state = self._player_def_ratings[pid]

                off_pre = float(off_state.rating_value)
                def_pre = float(def_state.rating_value)

                pred_off = self._performance_predictor.predict_performance(
                    player_rating=pre_player,
                    opponent_team_rating=PreMatchTeamRating(id=team1, players=[], rating_value=team1_def_rating),
                    team_rating=PreMatchTeamRating(id=team2, players=[], rating_value=team2_off_rating),
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
                    opponent_team_rating=PreMatchTeamRating(id=team1, players=[], rating_value=team1_off_rating),
                    team_rating=PreMatchTeamRating(id=team2, players=[], rating_value=team2_def_rating),
                )

                off_perf = float(pre_player.match_performance.performance_value)
                def_perf = float(team2_def_perf)

                mult_off = self._applied_multiplier_off(off_state)
                mult_def = self._applied_multiplier_def(def_state)

                off_change = (off_perf - float(pred_off)) * mult_off * float(pre_player.match_performance.participation_weight)
                def_change = (def_perf - float(pred_def)) * mult_def * float(pre_player.match_performance.participation_weight)

                if math.isnan(off_change) or math.isnan(def_change):
                    raise ValueError(f"NaN player rating change for player_id={pid}, match_id={r[cn.match_id]}")

                player_updates.append((pid, team2, off_pre, def_pre, float(pred_off), float(pred_def), float(off_change), float(def_change), day_number))

            # Emit pre-match rows for all players in both teams
            match_id = r[cn.match_id]
            for (pid, team_id, off_pre, def_pre, pred_off, pred_def, off_change, def_change, dn) in player_updates:
                out[cn.player_id].append(pid)
                out[cn.match_id].append(match_id)
                out[cn.team_id].append(team_id)

                out[self.PLAYER_OFF_RATING_COL].append(off_pre)
                out[self.PLAYER_DEF_RATING_COL].append(def_pre)
                out[self.PLAYER_PRED_OFF_PERF_COL].append(pred_off)
                out[self.PLAYER_PRED_DEF_PERF_COL].append(pred_def)

                # back-compat: "player_rating" is offense rating, "player_predicted_performance" is predicted offense
                out[self.PLAYER_RATING_COL].append(off_pre)
                out[self.PLAYER_PRED_PERF_COL].append(pred_off)

            # Queue updates and apply at update_id boundary (same pattern as team generator)
            for (pid, team_id, _off_pre, _def_pre, _pred_off, _pred_def, off_change, def_change, dn) in player_updates:
                pending_team_updates.append((pid, team_id, off_change, def_change, dn))

            if last_update_id is None:
                last_update_id = update_id

            if update_id != last_update_id:
                # apply everything except current match's updates:
                # since we appended current match updates already, we need to apply all but the last match updates.
                # easiest: apply all except the last len(player_updates) entries.
                cutoff = len(pending_team_updates) - len(player_updates)
                if cutoff > 0:
                    self._apply_player_updates(pending_team_updates[:cutoff])
                    pending_team_updates = pending_team_updates[cutoff:]
                last_update_id = update_id

        # apply remaining
        if pending_team_updates:
            self._apply_player_updates(pending_team_updates)

        return pl.DataFrame(out, strict=False)

    def _apply_player_updates(self, updates: list[tuple[str, str, float, float, int]]) -> None:
        """
        updates: (player_id, team_id, off_change, def_change, day_number)
        """
        for player_id, team_id, off_change, def_change, day_number in updates:
            # OFF state update
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

            # DEF state update
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

    # --------------------
    # Feature engineering
    # --------------------
    def _add_rating_features(self, df: pl.DataFrame) -> pl.DataFrame:
        non_predictor_features_out = self.non_predictor_features_out or []
        cols_to_add = set((self.features_out or []) + non_predictor_features_out)

        cn = self.column_names

        # Team OFF projected from PLAYER_OFF_RATING_COL
        if (
            self.TEAM_OFF_RATING_PROJ_COL in cols_to_add
            or self.TEAM_RATING_PROJ_COL in cols_to_add
            or self.OPP_OFF_RATING_PROJ_COL in cols_to_add
            or self.OPP_RATING_PROJ_COL in cols_to_add
            or self.DIFF_PROJ_COL in cols_to_add
            or self.MEAN_PROJ_COL in cols_to_add
        ):
            df = add_team_rating_projected(
                df=df,
                column_names=cn,
                player_rating_col=self.PLAYER_OFF_RATING_COL,
                team_rating_out=self.TEAM_OFF_RATING_PROJ_COL,
            )
            # back-compat: TEAM_RATING_PROJECTED := TEAM_OFF_RATING_PROJECTED
            df = df.with_columns(pl.col(self.TEAM_OFF_RATING_PROJ_COL).alias(self.TEAM_RATING_PROJ_COL))

        # Team DEF projected from PLAYER_DEF_RATING_COL
        if self.TEAM_DEF_RATING_PROJ_COL in cols_to_add or self.OPP_DEF_RATING_PROJ_COL in cols_to_add or self.OPP_RATING_PROJ_COL in cols_to_add or self.DIFF_PROJ_COL in cols_to_add:
            df = add_team_rating_projected(
                df=df,
                column_names=cn,
                player_rating_col=self.PLAYER_DEF_RATING_COL,
                team_rating_out=self.TEAM_DEF_RATING_PROJ_COL,
            )

        # Opponent OFF/DEF projected
        if self.OPP_OFF_RATING_PROJ_COL in cols_to_add:
            df = add_opp_team_rating(
                df=df,
                column_names=cn,
                team_rating_col=self.TEAM_OFF_RATING_PROJ_COL,
                opp_team_rating_out=self.OPP_OFF_RATING_PROJ_COL,
            )

        if self.OPP_DEF_RATING_PROJ_COL in cols_to_add or self.OPP_RATING_PROJ_COL in cols_to_add or self.DIFF_PROJ_COL in cols_to_add:
            df = add_opp_team_rating(
                df=df,
                column_names=cn,
                team_rating_col=self.TEAM_DEF_RATING_PROJ_COL,
                opp_team_rating_out=self.OPP_DEF_RATING_PROJ_COL,
            )
            # back-compat: OPPONENT_RATING_PROJECTED := OPPONENT_DEF_RATING_PROJECTED
            df = df.with_columns(pl.col(self.OPP_DEF_RATING_PROJ_COL).alias(self.OPP_RATING_PROJ_COL))

        # Unknown (unprojected) team rating from OFF ratings (keep old behavior)
        if self.TEAM_RATING_COL in cols_to_add or self.OPP_RATING_COL in cols_to_add or self.DIFF_COL in cols_to_add:
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

        # Difference projected: OFF vs DEF
        if self.DIFF_PROJ_COL in cols_to_add:
            df = add_rating_difference_projected(
                df=df,
                team_rating_col=self.TEAM_RATING_PROJ_COL,   # OFF
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

        # Drop unrequested columns using enum iteration (since you converted to Enums)
        base_known = [f.value for f in RatingKnownFeatures]
        base_unknown = [f.value for f in RatingUnknownFeatures]
        cols_to_eval = [self._suffix(c) for c in (base_known + base_unknown)]

        cols_to_drop = [c for c in cols_to_eval if (c in df.columns and c not in cols_to_add)]
        return df.drop(cols_to_drop)

    # --------------------
    # Match DF creation
    # --------------------
    def _create_match_df(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.league_identifier:
            df = self.league_identifier.add_leagues(df)

        cn = self.column_names

        player_stat_cols = [cn.player_id]
        if self.performance_column in df.columns:
            player_stat_cols.append(self.performance_column)

        if cn.participation_weight and cn.participation_weight in df.columns:
            player_stat_cols.append(cn.participation_weight)

        if cn.projected_participation_weight and cn.projected_participation_weight in df.columns:
            player_stat_cols.append(cn.projected_participation_weight)

        if cn.position and cn.position in df.columns:
            player_stat_cols.append(cn.position)

        if cn.league and cn.league in df.columns:
            player_stat_cols.append(cn.league)

        df = df.with_columns(pl.struct(player_stat_cols).alias(PLAYER_STATS))

        agg_df = (
            df.group_by([cn.match_id, cn.team_id, cn.start_date, cn.update_match_id])
            .agg(pl.col(PLAYER_STATS).alias(PLAYER_STATS))
        )

        team_id_col = cn.team_id
        match_df = (
            agg_df.join(agg_df, on=cn.match_id, how="inner", suffix="_opponent")
            .filter(pl.col(team_id_col) != pl.col(f"{team_id_col}_opponent"))
            .unique(cn.match_id)
            .sort(list(set([cn.start_date, cn.match_id, cn.update_match_id])))
        )

        start_as_int = (
            pl.col(cn.start_date)
            .str.strptime(pl.Datetime, strict=False)
            .cast(pl.Date)
            .cast(pl.Int32)
        )
        return match_df.with_columns((start_as_int - start_as_int.min() + 1).alias("__day_number"))

    # --------------------
    # Collection creation (OFF/DEF)
    # --------------------
    def _create_pre_match_players_collection(self, r: dict, stats_col: str, day_number: int, team_id: str) -> PreMatchPlayersCollection:
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

            participation_weight = team_player.get(cn.participation_weight, 1.0) if cn.participation_weight else 1.0
            projected_participation_weight = (
                team_player.get(cn.projected_participation_weight, participation_weight)
                if cn.projected_participation_weight
                else participation_weight
            )
            projected_participation_weights.append(projected_participation_weight)

            perf_val = float(team_player[self.performance_column]) if (self.performance_column in team_player and team_player[self.performance_column] is not None) else 0.0

            mp = MatchPerformance(
                performance_value=perf_val,
                projected_participation_weight=projected_participation_weight,
                participation_weight=participation_weight,
            )

            if player_id in self._player_off_ratings and player_id in self._player_def_ratings:
                off_state = self._player_off_ratings[player_id]
                # pre_match_player_rating uses OFF rating for predictor inputs
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

        # Generate start ratings for new players (OFF + DEF)
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

            # create both states
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

    # --------------------
    # Team-level helpers
    # --------------------
    def _team_off_perf_from_collection(self, c: PreMatchPlayersCollection) -> float:
        # observed offense perf = weighted mean of player performance_value using participation_weight if present
        cn = self.column_names
        if not c.pre_match_player_ratings:
            return 0.0
        wsum = 0.0
        psum = 0.0
        for pre in c.pre_match_player_ratings:
            w = float(pre.match_performance.participation_weight) if cn.participation_weight else 1.0
            psum += float(pre.match_performance.performance_value) * w
            wsum += w
        return psum / wsum if wsum else 0.0

    def _team_off_def_rating_from_collection(self, c: PreMatchPlayersCollection) -> tuple[float, float]:
        cn = self.column_names
        if not c.player_ids:
            return 0.0, 0.0

        # OFF rating aggregate
        off_vals = [float(self._player_off_ratings[pid].rating_value) for pid in c.player_ids if pid in self._player_off_ratings]
        def_vals = [float(self._player_def_ratings[pid].rating_value) for pid in c.player_ids if pid in self._player_def_ratings]
        if not off_vals or not def_vals:
            return 0.0, 0.0

        if cn.projected_participation_weight and c.projected_particiation_weights:
            w = [float(x) for x in c.projected_particiation_weights]
            wsum = sum(w) if w else 0.0
            off = (sum(v * ww for v, ww in zip(off_vals, w)) / wsum) if wsum else (sum(off_vals) / len(off_vals))
            dff = (sum(v * ww for v, ww in zip(def_vals, w)) / wsum) if wsum else (sum(def_vals) / len(def_vals))
            return float(off), float(dff)

        return float(sum(off_vals) / len(off_vals)), float(sum(def_vals) / len(def_vals))

    # --------------------
    # Future ratings (no updates)
    # --------------------
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

        def get_perf_value(team_player: dict) -> float:
            if self.performance_column in team_player and team_player[self.performance_column] is not None:
                return float(team_player[self.performance_column])
            return 0.0

        def ensure_new_player(pid: str, day_number: int, mp: MatchPerformance, league, position, team_pre: list[PreMatchPlayerRating]) -> None:
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

            # Build local collections (do not touch self._player_*_ratings)
            def build_local_team(stats_col: str) -> tuple[list[PreMatchPlayerRating], list[str], list[float], list[float], float]:
                pre_list: list[PreMatchPlayerRating] = []
                player_ids: list[str] = []
                proj_w: list[float] = []
                off_vals: list[float] = []
                # offense perf for team
                psum, wsum = 0.0, 0.0

                for tp in r[stats_col]:
                    pid = tp[cn.player_id]
                    player_ids.append(pid)

                    position = tp.get(cn.position)
                    league = tp.get(cn.league, None)

                    pw = tp.get(cn.participation_weight, 1.0) if cn.participation_weight else 1.0
                    ppw = tp.get(cn.projected_participation_weight, pw) if cn.projected_participation_weight else pw
                    proj_w.append(float(ppw))

                    mp = MatchPerformance(
                        performance_value=get_perf_value(tp),
                        projected_participation_weight=ppw,
                        participation_weight=pw,
                    )

                    ensure_new_player(pid, day_number, mp, league, position, pre_list)

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

                    # offense perf aggregation
                    psum += float(mp.performance_value) * float(pw)
                    wsum += float(pw)

                team_off_perf = psum / wsum if wsum else 0.0
                return pre_list, player_ids, off_vals, proj_w, team_off_perf

            t1_pre, t1_ids, t1_off_vals, t1_proj_w, t1_off_perf = build_local_team(PLAYER_STATS)
            t2_pre, t2_ids, t2_off_vals, t2_proj_w, t2_off_perf = build_local_team(f"{PLAYER_STATS}_opponent")

            t1_def_perf = 1.0 - t2_off_perf
            t2_def_perf = 1.0 - t1_off_perf

            def team_off_def_rating(ids: list[str], w: list[float]) -> tuple[float, float]:
                if not ids:
                    return 0.0, 0.0
                off = [float(local_off[pid].rating_value) for pid in ids]
                dff = [float(local_def[pid].rating_value) for pid in ids]
                if cn.projected_participation_weight and w and sum(w):
                    ws = sum(w)
                    return (
                        sum(v * ww for v, ww in zip(off, w)) / ws,
                        sum(v * ww for v, ww in zip(dff, w)) / ws,
                    )
                return sum(off) / len(off), sum(dff) / len(dff)

            t1_off_rating, t1_def_rating = team_off_def_rating(t1_ids, t1_proj_w)
            t2_off_rating, t2_def_rating = team_off_def_rating(t2_ids, t2_proj_w)

            # Predict (no updates)
            # Team1 players
            for pre in t1_pre:
                pid = pre.id
                off_pre = float(local_off[pid].rating_value)
                def_pre = float(local_def[pid].rating_value)

                pred_off = self._performance_predictor.predict_performance(
                    player_rating=pre,
                    opponent_team_rating=PreMatchTeamRating(id=team2, players=[], rating_value=t2_def_rating),
                    team_rating=PreMatchTeamRating(id=team1, players=[], rating_value=t1_off_rating),
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
                    opponent_team_rating=PreMatchTeamRating(id=team2, players=[], rating_value=t2_off_rating),
                    team_rating=PreMatchTeamRating(id=team1, players=[], rating_value=t1_def_rating),
                )

                out[cn.player_id].append(pid)
                out[cn.match_id].append(match_id)
                out[cn.team_id].append(team1)
                out[self.PLAYER_OFF_RATING_COL].append(off_pre)
                out[self.PLAYER_DEF_RATING_COL].append(def_pre)
                out[self.PLAYER_PRED_OFF_PERF_COL].append(float(pred_off))
                out[self.PLAYER_PRED_DEF_PERF_COL].append(float(pred_def))
                out[self.PLAYER_RATING_COL].append(off_pre)
                out[self.PLAYER_PRED_PERF_COL].append(float(pred_off))

            # Team2 players
            for pre in t2_pre:
                pid = pre.id
                off_pre = float(local_off[pid].rating_value)
                def_pre = float(local_def[pid].rating_value)

                pred_off = self._performance_predictor.predict_performance(
                    player_rating=pre,
                    opponent_team_rating=PreMatchTeamRating(id=team1, players=[], rating_value=t1_def_rating),
                    team_rating=PreMatchTeamRating(id=team2, players=[], rating_value=t2_off_rating),
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
                    opponent_team_rating=PreMatchTeamRating(id=team1, players=[], rating_value=t1_off_rating),
                    team_rating=PreMatchTeamRating(id=team2, players=[], rating_value=t2_def_rating),
                )

                out[cn.player_id].append(pid)
                out[cn.match_id].append(match_id)
                out[cn.team_id].append(team2)
                out[self.PLAYER_OFF_RATING_COL].append(off_pre)
                out[self.PLAYER_DEF_RATING_COL].append(def_pre)
                out[self.PLAYER_PRED_OFF_PERF_COL].append(float(pred_off))
                out[self.PLAYER_PRED_DEF_PERF_COL].append(float(pred_def))
                out[self.PLAYER_RATING_COL].append(off_pre)
                out[self.PLAYER_PRED_PERF_COL].append(float(pred_off))

        return pl.DataFrame(out, strict=False)
