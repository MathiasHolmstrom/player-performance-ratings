# team_rating_generator.py
from __future__ import annotations

import inspect
import math
from typing import Literal

import polars as pl

from spforge.data_structures import ColumnNames, GameColumnNames, TeamRatingsResult
from spforge.feature_generator._utils import _to_polars_eager
from spforge.performance_transformers._performance_manager import ColumnWeight, PerformanceManager
from spforge.ratings._base import RatingGenerator, RatingState
from spforge.ratings.enums import RatingKnownFeatures, RatingUnknownFeatures
from spforge.ratings.team_performance_predictor import (
    TeamPerformancePredictor,
    TeamRatingDifferencePerformancePredictor,
    TeamRatingMeanPerformancePredictor,
    TeamRatingNonOpponentPerformancePredictor,
)
from spforge.ratings.team_start_rating_generator import TeamStartRatingGenerator

TEAM_PRED_OFF_COL = "__TEAM_PREDICTED_OFF_PERFORMANCE"
TEAM_PRED_DEF_COL = "__TEAM_PREDICTED_DEF_PERFORMANCE"


class TeamRatingGenerator(RatingGenerator):
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
        column_names: ColumnNames | GameColumnNames | None = None,
        output_suffix: str | None = None,
        start_harcoded_start_rating: float | None = None,
        start_league_ratings: dict[str, float] | None = None,
        start_league_quantile: float = 0.2,
        start_min_count_for_percentiles: int = 50,
        **kwargs,
    ):
        # Handle GameColumnNames vs ColumnNames
        if isinstance(column_names, GameColumnNames):
            self._game_column_names: GameColumnNames | None = column_names
            # Create ColumnNames for internal use after conversion
            _column_names = ColumnNames(
                match_id=column_names.match_id,
                start_date=column_names.start_date,
                team_id="team_id",  # Standard name after conversion
                league=column_names.league,
                update_match_id=column_names.update_match_id,
            )
        else:
            self._game_column_names = None
            _column_names = column_names

        super().__init__(
            output_suffix=output_suffix,
            performance_column=performance_column,
            performance_weights=performance_weights,
            column_names=_column_names,
            features_out=features_out,
            performance_manager=performance_manager,
            auto_scale_performance=auto_scale_performance,
            performance_predictor=performance_predictor,
            rating_change_multiplier_offense=rating_change_multiplier_offense,
            rating_change_multiplier_defense=rating_change_multiplier_defense,
            confidence_days_ago_multiplier=confidence_days_ago_multiplier,
            confidence_max_days=confidence_max_days,
            confidence_value_denom=confidence_value_denom,
            confidence_max_sum=confidence_max_sum,
            confidence_weight=confidence_weight,
            non_predictor_features_out=non_predictor_features_out,
            min_rating_change_multiplier_ratio=min_rating_change_multiplier_ratio,
            league_rating_change_update_threshold=league_rating_change_update_threshold,
            league_rating_adjustor_multiplier=league_rating_adjustor_multiplier,
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
        self.start_league_ratings = start_league_ratings
        self.start_league_quantile = start_league_quantile
        self.start_min_count_for_percentiles = start_min_count_for_percentiles
        self.start_harcoded_start_rating = start_harcoded_start_rating
        self.start_rating_generator = TeamStartRatingGenerator(
            league_ratings=self.start_league_ratings,
            league_quantile=self.start_league_quantile,
            min_count_for_percentiles=self.start_min_count_for_percentiles,
            harcoded_start_rating=start_harcoded_start_rating,
        )

        self.TEAM_PRED_OFF_PERF_COL = self._suffix(TEAM_PRED_OFF_COL)
        self.TEAM_PRED_DEF_PERF_COL = self._suffix(TEAM_PRED_DEF_COL)

        self.TEAM_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.TEAM_RATING_PROJECTED))

        self.OPP_RATING_PROJ_COL = self._suffix(str(RatingKnownFeatures.OPPONENT_RATING_PROJECTED))
        self.DIFF_PROJ_COL = self._suffix(str(RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED))
        self.MEAN_PROJ_COL = self._suffix(str(RatingKnownFeatures.RATING_MEAN_PROJECTED))

        self.DIFF_COL = self._suffix(str(RatingUnknownFeatures.TEAM_RATING_DIFFERENCE))

        self._team_off_ratings: dict[str, RatingState] = {}
        self._team_def_ratings: dict[str, RatingState] = {}
        if performance_predictor == "mean":
            _performance_predictor_class = TeamRatingMeanPerformancePredictor
        elif performance_predictor == "difference":
            _performance_predictor_class = TeamRatingDifferencePerformancePredictor
        elif performance_predictor == "ignore_opponent":
            _performance_predictor_class = TeamRatingNonOpponentPerformancePredictor
        else:
            raise ValueError(f"performance_predictor {performance_predictor} is not supported")

        self.performance_predictor = performance_predictor
        self.use_off_def_split = bool(use_off_def_split)
        sig = inspect.signature(_performance_predictor_class.__init__)
        init_params = [name for name, _param in sig.parameters.items() if name != "self"]
        performance_predictor_params = {k: v for k, v in kwargs.items() if k in init_params}
        self._performance_predictor: TeamPerformancePredictor = _performance_predictor_class(
            **performance_predictor_params
        )

    def fit_transform(self, df: pl.DataFrame, column_names: ColumnNames | None = None):
        """Override to handle game-level data conversion before base class processing."""
        # Convert game-level data to game+team format if needed
        if self._game_column_names is not None:
            df = self._convert_game_to_game_team(df, self._game_column_names)

        # Call parent fit_transform
        return super().fit_transform(df, column_names)

    def transform(self, df: pl.DataFrame):
        """Override to handle game-level data conversion before base class processing."""
        # Convert game-level data to game+team format if needed
        if self._game_column_names is not None:
            df = self._convert_game_to_game_team(df, self._game_column_names)

        # Call parent transform
        return super().transform(df)

    def future_transform(self, df: pl.DataFrame):
        """Override to handle game-level data conversion before base class processing."""
        # Convert game-level data to game+team format if needed
        if self._game_column_names is not None:
            df = self._convert_game_to_game_team(df, self._game_column_names)

        # Call parent future_transform
        return super().future_transform(df)

    def _ensure_team_off(self, team_id: str, day_number: int, league: str) -> RatingState:
        if team_id not in self._team_off_ratings:
            rating = self.start_rating_generator.generate_rating_value(
                day_number=day_number, league=league
            )
            self._team_off_ratings[team_id] = RatingState(id=team_id, rating_value=rating)
        return self._team_off_ratings[team_id]

    def _ensure_team_def(self, team_id: str, day_number: int, league: str) -> RatingState:
        if team_id not in self._team_def_ratings:
            rating = self.start_rating_generator.generate_rating_value(
                day_number=day_number, league=league
            )
            self._team_def_ratings[team_id] = RatingState(id=team_id, rating_value=rating)
        return self._team_def_ratings[team_id]

    def _calculate_post_match_confidence_sum(self, state: RatingState, day_number: int) -> float:
        return self._post_match_confidence_sum(
            state=state, day_number=day_number, participation_weight=1.0
        )

    def _convert_game_to_game_team(
        self, df: pl.DataFrame, game_column_names: GameColumnNames
    ) -> pl.DataFrame:
        """Convert game-level data (1 row per match) to game+team format (2 rows per match).

        Args:
            df: DataFrame with 1 row per match (pandas or polars)
            game_column_names: Configuration specifying column mappings

        Returns:
            DataFrame with 2 rows per match (one per team)
        """

        # Convert to polars for internal processing
        df = _to_polars_eager(df)

        gcn = game_column_names

        # Collect all columns that should be preserved (not part of team-specific data)
        base_cols = [gcn.match_id, gcn.start_date]
        if gcn.league and gcn.league in df.columns:
            base_cols.append(gcn.league)
        if gcn.update_match_id and gcn.update_match_id != gcn.match_id:
            base_cols.append(gcn.update_match_id)

        # Create team1 rows
        team1_select = base_cols + [gcn.team1_name]
        team1_rename = {gcn.team1_name: "team_id"}

        # Add performance columns for team1
        for output_col, (team1_col, _) in gcn.performance_column_pairs.items():
            if team1_col in df.columns:
                team1_select.append(team1_col)
                team1_rename[team1_col] = output_col

        team1_df = df.select([c for c in team1_select if c in df.columns]).rename(team1_rename)

        # Create team2 rows
        team2_select = base_cols + [gcn.team2_name]
        team2_rename = {gcn.team2_name: "team_id"}

        # Add performance columns for team2
        for output_col, (_, team2_col) in gcn.performance_column_pairs.items():
            if team2_col in df.columns:
                team2_select.append(team2_col)
                team2_rename[team2_col] = output_col

        team2_df = df.select([c for c in team2_select if c in df.columns]).rename(team2_rename)

        # Union team1 and team2 rows
        result_df = pl.concat([team1_df, team2_df], how="vertical")

        # Sort chronologically
        sort_cols = [gcn.start_date, gcn.match_id]
        if gcn.update_match_id and gcn.update_match_id != gcn.match_id:
            sort_cols.append(gcn.update_match_id)
        result_df = result_df.sort(sort_cols)

        return result_df

    def _create_match_df(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        base_cols = list(
            dict.fromkeys([cn.match_id, cn.team_id, cn.start_date, cn.update_match_id])
        )
        if self.performance_column in df.columns:
            base_cols.append(self.performance_column)
        if self.column_names.league:
            base_cols.append(self.column_names.league)

        team_rows = df.select([c for c in base_cols if c in df.columns]).unique(
            [cn.match_id, cn.team_id]
        )

        match_df = (
            team_rows.join(team_rows, on=cn.match_id, how="inner", suffix="_opponent")
            .filter(pl.col(cn.team_id) != pl.col(f"{cn.team_id}_opponent"))
            .sort(list(set([cn.start_date, cn.match_id, cn.update_match_id])))
        )

        match_df = self._add_day_number(match_df, cn.start_date, "__day_number")
        return match_df

    def _historical_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names
        assert (
            df.group_by(cn.match_id).agg(pl.col(cn.team_id).n_unique())[cn.team_id].n_unique() == 1
        )
        assert df.group_by(cn.match_id).agg(pl.col(cn.team_id).n_unique())[cn.team_id].max() == 2

        match_df = self._create_match_df(df)
        ratings = self._calculate_ratings(match_df)

        cols = [
            c
            for c in df.columns
            if c not in (self.TEAM_PRED_OFF_PERF_COL, self.TEAM_PRED_DEF_PERF_COL)
        ]
        df2 = df.select(cols).join(ratings, on=[cn.match_id, cn.team_id], how="left")
        return self._add_rating_features(df2)

    def _calculate_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        rows: list[dict] = []
        pending_updates: list[tuple[str, str, float, int]] = []
        last_update_id = None

        perf_opp_col = f"{self.performance_column}_opponent"

        for r in match_df.iter_rows(named=True):
            team_id = r[cn.team_id]
            opp_id = r[f"{cn.team_id}_opponent"]
            update_id = r[cn.update_match_id]

            if last_update_id is not None and update_id != last_update_id:
                if pending_updates:
                    self._apply_team_updates(pending_updates)
                    pending_updates = []
                last_update_id = update_id
            day_number = int(r["__day_number"])
            league = r[self.column_names.league] if self.column_names.league else None

            s_off = self._ensure_team_off(team_id, league=league, day_number=day_number)
            s_def = self._ensure_team_def(team_id, league=league, day_number=day_number)
            o_off = self._ensure_team_off(opp_id, league=league, day_number=day_number)
            o_def = self._ensure_team_def(opp_id, league=league, day_number=day_number)

            team_off_pre = float(s_off.rating_value)
            team_def_pre = float(s_def.rating_value)
            opp_off_pre = float(o_off.rating_value)
            opp_def_pre = float(o_def.rating_value)

            off_perf = (
                float(r[self.performance_column])
                if r.get(self.performance_column) is not None
                else 0.0
            )
            opp_off_perf = float(r[perf_opp_col]) if r.get(perf_opp_col) is not None else 0.0
            if self.use_off_def_split:
                def_perf = 1.0 - opp_off_perf
            else:
                def_perf = off_perf

            pred_off = self._performance_predictor.predict_performance(
                rating_value=s_off.rating_value, opponent_team_rating_value=o_def.rating_value
            )
            pred_def = self._performance_predictor.predict_performance(
                rating_value=s_def.rating_value, opponent_team_rating_value=o_off.rating_value
            )
            if not self.use_off_def_split:
                pred_def = pred_off

            mult_off = self._applied_multiplier(s_off, self.rating_change_multiplier_offense)
            mult_def = self._applied_multiplier(s_def, self.rating_change_multiplier_defense)

            off_change = (off_perf - pred_off) * mult_off
            def_change = (def_perf - pred_def) * mult_def

            if math.isnan(off_change) or math.isnan(def_change):
                raise ValueError(
                    f"NaN rating change for team_id={team_id}, match_id={r[cn.match_id]}"
                )

            rows.append(
                {
                    cn.match_id: r[cn.match_id],
                    cn.team_id: team_id,
                    self.TEAM_OFF_RATING_PROJ_COL: team_off_pre,
                    self.TEAM_DEF_RATING_PROJ_COL: team_def_pre,
                    self.OPP_OFF_RATING_PROJ_COL: opp_off_pre,
                    self.OPP_DEF_RATING_PROJ_COL: opp_def_pre,
                    self.TEAM_PRED_OFF_PERF_COL: pred_off,
                    self.TEAM_PRED_DEF_PERF_COL: pred_def,
                    self.TEAM_RATING_PROJ_COL: team_off_pre,
                }
            )

            pending_updates.append(("off", team_id, off_change, day_number))
            pending_updates.append(("def", team_id, def_change, day_number))

            if last_update_id is None:
                last_update_id = update_id

        if pending_updates:
            self._apply_team_updates(pending_updates)

        return pl.DataFrame(rows, strict=False)

    def _apply_team_updates(self, updates: list[tuple[str, str, float, int]]) -> None:
        for kind, team_id, change, day_number in updates:
            if kind == "off":
                s = self._team_off_ratings[team_id]
            elif kind == "def":
                s = self._team_def_ratings[team_id]
            else:
                raise ValueError(f"Unknown update kind={kind}")

            s.confidence_sum = self._calculate_post_match_confidence_sum(s, day_number)
            s.rating_value += float(change)
            s.games_played += 1.0
            s.last_match_day_number = int(day_number)

    def _add_rating_features(self, df: pl.DataFrame) -> pl.DataFrame:

        cols_to_add = set((self._features_out or []) + (self.non_predictor_features_out or []))

        perf_col_name = self._suffix(str(RatingUnknownFeatures.PERFORMANCE))
        if perf_col_name in cols_to_add and perf_col_name not in df.columns:
            if self.performance_column in df.columns:
                df = df.with_columns(pl.col(self.performance_column).alias(perf_col_name))
            elif RatingUnknownFeatures.PERFORMANCE in (self.non_predictor_features_out or []):
                df = df.with_columns(pl.lit(0.0).alias(perf_col_name))

        need_opp = (
            self.OPP_RATING_PROJ_COL in cols_to_add
            or self.DIFF_PROJ_COL in cols_to_add
            or self.MEAN_PROJ_COL in cols_to_add
        )
        need_diff = self.DIFF_PROJ_COL in cols_to_add
        need_mean = self.MEAN_PROJ_COL in cols_to_add

        if need_opp and self.OPP_RATING_PROJ_COL not in df.columns:
            df = df.with_columns(
                pl.col(self.OPP_DEF_RATING_PROJ_COL).alias(self.OPP_RATING_PROJ_COL)
            )

        if need_diff and self.DIFF_PROJ_COL not in df.columns:
            df = df.with_columns(
                (pl.col(self.TEAM_RATING_PROJ_COL) - pl.col(self.OPP_RATING_PROJ_COL)).alias(
                    self.DIFF_PROJ_COL
                )
            )

        if need_mean and self.MEAN_PROJ_COL not in df.columns:
            df = df.with_columns(
                (
                    (pl.col(self.TEAM_RATING_PROJ_COL) + pl.col(self.OPP_RATING_PROJ_COL)) / 2.0
                ).alias(self.MEAN_PROJ_COL)
            )

        if self.DIFF_COL in cols_to_add and self.DIFF_COL not in df.columns:
            if self.TEAM_RATING_PROJ_COL not in df.columns:
                df = df.with_columns(
                    pl.col(self.TEAM_OFF_RATING_PROJ_COL).alias(self.TEAM_RATING_PROJ_COL)
                )
            if (
                self.OPP_RATING_PROJ_COL not in df.columns
                and self.OPP_DEF_RATING_PROJ_COL in df.columns
            ):
                df = df.with_columns(
                    pl.col(self.OPP_DEF_RATING_PROJ_COL).alias(self.OPP_RATING_PROJ_COL)
                )
            if self.OPP_RATING_PROJ_COL in df.columns:
                df = df.with_columns(
                    (pl.col(self.TEAM_RATING_PROJ_COL) - pl.col(self.OPP_RATING_PROJ_COL)).alias(
                        self.DIFF_COL
                    )
                )

        candidates = {
            self.TEAM_OFF_RATING_PROJ_COL,
            self.TEAM_DEF_RATING_PROJ_COL,
            self.OPP_OFF_RATING_PROJ_COL,
            self.OPP_DEF_RATING_PROJ_COL,
            self.TEAM_PRED_OFF_PERF_COL,
            self.TEAM_PRED_DEF_PERF_COL,
            self.TEAM_RATING_PROJ_COL,
            self.OPP_RATING_PROJ_COL,
            self.DIFF_PROJ_COL,
            self.DIFF_COL,
            self.MEAN_PROJ_COL,
            perf_col_name,
        }

        drop_cols = [c for c in candidates if (c in df.columns and c not in cols_to_add)]
        result = df.drop(drop_cols)

        return result

    def _future_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names
        assert (
            df.group_by(cn.match_id).agg(pl.col(cn.team_id).n_unique())[cn.team_id].n_unique() == 1
        )
        assert df.group_by(cn.match_id).agg(pl.col(cn.team_id).n_unique())[cn.team_id].max() == 2

        match_df = self._create_match_df(df)
        ratings = self._calculate_future_ratings(match_df)

        cols = [
            c
            for c in df.columns
            if c not in (self.TEAM_PRED_OFF_PERF_COL, self.TEAM_PRED_DEF_PERF_COL)
        ]
        df2 = df.select(cols).join(ratings, on=[cn.match_id, cn.team_id], how="left")
        return self._add_rating_features(df2)

    def _calculate_future_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        cn = self.column_names

        rows: list[dict] = []
        for r in match_df.iter_rows(named=True):
            day_number = r["__day_number"]
            league = r[self.column_names.league] if self.column_names.league else None
            team_id = r[cn.team_id]
            opp_id = r[f"{cn.team_id}_opponent"]

            s_off = self._ensure_team_off(team_id, day_number=day_number, league=league)
            s_def = self._ensure_team_def(team_id, day_number=day_number, league=league)
            o_off = self._ensure_team_off(opp_id, day_number=day_number, league=league)
            o_def = self._ensure_team_def(opp_id, day_number=day_number, league=league)

            team_off_pre = float(s_off.rating_value)
            team_def_pre = float(s_def.rating_value)
            opp_off_pre = float(o_off.rating_value)
            opp_def_pre = float(o_def.rating_value)

            pred_off = self._performance_predictor.predict_performance(
                rating_value=s_off.rating_value, opponent_team_rating_value=o_off.rating_value
            )
            pred_def = self._performance_predictor.predict_performance(
                rating_value=s_def.rating_value, opponent_team_rating_value=o_off.rating_value
            )
            if not self.use_off_def_split:
                pred_def = pred_off

            rows.append(
                {
                    cn.match_id: r[cn.match_id],
                    cn.team_id: team_id,
                    self.TEAM_OFF_RATING_PROJ_COL: team_off_pre,
                    self.TEAM_DEF_RATING_PROJ_COL: team_def_pre,
                    self.OPP_OFF_RATING_PROJ_COL: opp_off_pre,
                    self.OPP_DEF_RATING_PROJ_COL: opp_def_pre,
                    self.TEAM_PRED_OFF_PERF_COL: pred_off,
                    self.TEAM_PRED_DEF_PERF_COL: pred_def,
                    self.TEAM_RATING_PROJ_COL: team_off_pre,  # backwards compat
                }
            )

        return pl.DataFrame(rows, strict=False)

    @property
    def team_ratings(self) -> dict[str, TeamRatingsResult]:
        """Return combined offense and defense ratings for all teams."""
        result: dict[str, TeamRatingsResult] = {}
        all_team_ids = set(self._team_off_ratings.keys()) | set(self._team_def_ratings.keys())

        for team_id in all_team_ids:
            off_state = self._team_off_ratings.get(team_id)
            def_state = self._team_def_ratings.get(team_id)

            result[team_id] = TeamRatingsResult(
                id=team_id,
                offense_rating=off_state.rating_value if off_state else 0.0,
                defense_rating=def_state.rating_value if def_state else 0.0,
                offense_games_played=off_state.games_played if off_state else 0.0,
                defense_games_played=def_state.games_played if def_state else 0.0,
                offense_confidence_sum=off_state.confidence_sum if off_state else 0.0,
                defense_confidence_sum=def_state.confidence_sum if def_state else 0.0,
            )

        return result
