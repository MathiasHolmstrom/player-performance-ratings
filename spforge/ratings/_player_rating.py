import copy
import inspect
import logging
import math
from typing import Optional, Union, Literal

import narwhals as nw
import polars as pl
from narwhals import DataFrame
from narwhals.typing import IntoFrameT

from spforge.data_structures import PlayerRating, \
    ColumnNames, PlayerRatingChange, TeamRatingChange, PreMatchTeamRating, PreMatchPlayerRating, \
    PreMatchPlayersCollection, MatchPerformance, MatchPlayer
from spforge.ratings import RatingKnownFeatures, RatingUnknownFeatures
from spforge.ratings._base import RatingGenerator
from spforge.ratings.league_identifier import LeagueIdentifer2
from spforge.ratings.performance_predictor import RatingMeanPerformancePredictor, RatingNonOpponentPerformancePredictor, \
    RatingDifferencePerformancePredictor
from spforge.ratings.start_rating_generator import StartRatingGenerator
from spforge.ratings.utils import add_team_rating_projected, add_opp_team_rating, add_team_rating, \
    add_rating_difference_projected, add_rating_mean_projected
from spforge.transformers.fit_transformers import PerformanceWeightsManager
from spforge.transformers.fit_transformers._performance_manager import ColumnWeight, PerformanceManager
from spforge.transformers.lag_transformers._utils import to_polars

PLAYER_STATS = '__PLAYER_STATS'


class PlayerRatingGenerator(RatingGenerator):

    def __init__(self,
                 performance_column: str,
                 performance_weights: Optional[
                     list[Union[ColumnWeight, dict[str, float]]]
                 ] = None,
                 performance_manager: PerformanceManager | None = None,
                 auto_scale_performance: bool = False,
                 performance_predictor: Literal['difference', 'mean', 'ignore_opponent'] = 'difference',
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
                 **kwargs):

        # ---- NEW: output suffix (defaults to performance_column) ----
        self.output_suffix = performance_column if output_suffix is None else output_suffix

        def _suffix(col: str) -> str:
            # Empty string / falsy suffix => no suffix
            if not self.output_suffix:
                return col
            return f"{col}_{self.output_suffix}"

        def _suffix_feature_list(feats: Optional[list]) -> Optional[list[str]]:
            if not feats:
                return feats
            return [_suffix(str(f)) for f in feats]

        # canonical suffixed columns used throughout this generator
        self._suffix_col = _suffix

        self.PLAYER_RATING_COL = _suffix(str(RatingKnownFeatures.PLAYER_RATING))
        self.PLAYER_PRED_PERF_COL = _suffix(str(RatingUnknownFeatures.PLAYER_PREDICTED_PERFORMANCE))

        self.TEAM_RATING_PROJ_COL = _suffix(str(RatingKnownFeatures.TEAM_RATING_PROJECTED))
        self.OPP_RATING_PROJ_COL = _suffix(str(RatingKnownFeatures.OPPONENT_RATING_PROJECTED))
        self.DIFF_PROJ_COL = _suffix(str(RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED))
        self.MEAN_PROJ_COL = _suffix(str(RatingKnownFeatures.RATING_MEAN_PROJECTED))

        self.TEAM_RATING_COL = _suffix(str(RatingUnknownFeatures.TEAM_RATING))
        self.OPP_RATING_COL = _suffix(str(RatingUnknownFeatures.OPPONENT_RATING))
        self.DIFF_COL = _suffix(str(RatingUnknownFeatures.RATING_DIFFERENCE))
        # ------------------------------------------------------------

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
            team_id_change_confidence_sum_decrease=team_id_change_confidence_sum_decrease,
            features_out=_suffix_feature_list(features_out),
            performance_manager=performance_manager,
            auto_scale_performance=auto_scale_performance,
            performance_predictor=performance_predictor,
            performance_weights=performance_weights,
            non_predictor_features_out=_suffix_feature_list(non_predictor_features_out),
            league_rating_adjustor_multiplier=league_rating_adjustor_multiplier,
            league_rating_change_update_threshold=league_rating_change_update_threshold
        )

        self.start_league_ratings = start_league_ratings
        self.start_league_quantile = start_league_quantile
        self.start_min_count_for_percentiles = start_min_count_for_percentiles
        self.start_team_rating_subtract = start_team_rating_subtract
        self.start_team_weight = start_team_weight
        self.start_max_days_ago_league_entities = start_max_days_ago_league_entities
        self.start_min_match_count_team_rating = start_min_match_count_team_rating
        self.start_hardcoded_start_rating = start_harcoded_start_rating

        self.team_id_change_confidence_sum_decrease = (
            team_id_change_confidence_sum_decrease
        )
        self.column_names = column_names
        self._player_ratings: dict[str, PlayerRating] = {}

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

    def _calculate_applied_rating_change_multiplier(
            self,
            player_id: str,
            team_id: str,
    ) -> float:
        if (
                self._player_ratings[player_id].most_recent_team_id
                and self._player_ratings[player_id].most_recent_team_id != team_id
        ):
            self._player_ratings[player_id].confidence_sum -= self.team_id_change_confidence_sum_decrease

        return self._applied_multiplier(self._player_ratings[player_id])

    def _calculate_post_match_confidence_sum(
            self, entity_rating: PlayerRating, day_number: int, particpation_weight: float
    ) -> float:
        return self._post_match_confidence_sum(
            state=entity_rating,
            day_number=day_number,
            participation_weight=particpation_weight,
        )

    def _historical_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        match_df = self._create_match_df(df)
        ratings = self._calculate_ratings(match_df)

        cols = [
            c for c in df.columns
            if c not in (self.PLAYER_PRED_PERF_COL, self.PLAYER_RATING_COL)
        ]
        df = (
            df.select(cols)
            .join(
                ratings,
                on=[self.column_names.player_id, self.column_names.match_id, self.column_names.team_id],
            )
        )

        return self._add_rating_features(df)

    def _calculate_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        team_rating_changes = []
        update_ids = []
        new_ratings_data = {
            self.column_names.player_id: [],
            self.column_names.match_id: [],
            self.column_names.team_id: [],
            self.PLAYER_RATING_COL: [],
            self.PLAYER_PRED_PERF_COL: [],
        }
        for r in match_df.iter_rows(named=True):
            team1 = r[self.column_names.team_id]
            team2 = r[f"{self.column_names.team_id}_opponent"]
            update_ids.append(r[self.column_names.update_match_id])
            pre_match_players_collection_team1 = self._create_pre_match_players_collection(
                r=r, stats_col=PLAYER_STATS)
            pre_match_players_collection_team2 = self._create_pre_match_players_collection(
                r=r, stats_col=f"{PLAYER_STATS}_opponent")
            new_player_pre_match_ratings_team1, new_player_pre_match_ratings_values_team1 = self._generate_new_player_pre_match_ratings(
                day_number=r['__day_number'],
                new_players=pre_match_players_collection_team1.new_players,
                team_pre_match_player_ratings=pre_match_players_collection_team1.pre_match_player_ratings)
            new_player_pre_match_ratings_team2, new_player_pre_match_ratings_values_team2 = self._generate_new_player_pre_match_ratings(
                day_number=r['__day_number'],
                new_players=pre_match_players_collection_team2.new_players,
                team_pre_match_player_ratings=pre_match_players_collection_team2.pre_match_player_ratings)

            pre_match_players_collection_team1.pre_match_player_ratings.extend(new_player_pre_match_ratings_team1)
            pre_match_players_collection_team2.pre_match_player_ratings.extend(new_player_pre_match_ratings_team2)
            pre_match_players_collection_team1.player_rating_values.extend(new_player_pre_match_ratings_values_team1)
            pre_match_players_collection_team2.player_rating_values.extend(new_player_pre_match_ratings_values_team2)
            player_rating_values_team1 = pre_match_players_collection_team1.player_rating_values
            player_rating_values_team2 = pre_match_players_collection_team2.player_rating_values
            if self.column_names.participation_weight:
                team1_proj_weights = pre_match_players_collection_team1.projected_particiation_weights
                team2_proj_weights = pre_match_players_collection_team2.projected_particiation_weights

                team1_rating_value = (
                        sum(x * y for x, y in zip(player_rating_values_team1, team1_proj_weights))
                        / sum(team1_proj_weights)
                )

                team2_rating_value = (
                        sum(x * y for x, y in zip(player_rating_values_team2, team2_proj_weights))
                        / sum(team2_proj_weights)
                )
            else:
                team1_rating_value = sum(pre_match_players_collection_team1.player_rating_values) / len(
                    pre_match_players_collection_team1.player_rating_values)
                team2_rating_value = sum(pre_match_players_collection_team2.player_rating_values) / len(
                    pre_match_players_collection_team2.player_rating_values)

            pre_match_team_ratings = [
                PreMatchTeamRating(
                    id=team1,
                    players=pre_match_players_collection_team1.pre_match_player_ratings,
                    rating_value=team1_rating_value,
                ),
                PreMatchTeamRating(
                    id=team2,
                    players=pre_match_players_collection_team2.pre_match_player_ratings,
                    rating_value=team2_rating_value,
                )
            ]
            match_team_rating_changes = self._create_match_team_rating_changes(
                day_number=r['__day_number'],
                pre_match_team_ratings=pre_match_team_ratings
            )

            team_rating_changes += match_team_rating_changes

            if (
                    len(update_ids)
                    or len(update_ids) == len(match_df)
                    or update_ids[len(update_ids) - 1].update_id != r[self.column_names.update_match_id]
            ):
                self._update_ratings(team_rating_changes=team_rating_changes)
                team_rating_changes = []

            player_ratings = player_rating_values_team1 + player_rating_values_team2
            match_ids = [r[self.column_names.match_id]] * (
                    len(player_rating_values_team1) + len(player_rating_values_team2))
            team_ids = [team1] * len(player_rating_values_team1) + [team2] * len(player_rating_values_team2)
            predicted_player_performances = match_team_rating_changes[0].predicted_player_performances + \
                                            match_team_rating_changes[1].predicted_player_performances
            player_ids = pre_match_players_collection_team1.player_ids + pre_match_players_collection_team2.player_ids
            new_ratings_data[self.PLAYER_RATING_COL].extend(player_ratings)
            new_ratings_data[self.column_names.match_id].extend(match_ids)
            new_ratings_data[self.column_names.team_id].extend(team_ids)
            new_ratings_data[self.PLAYER_PRED_PERF_COL].extend(predicted_player_performances)

            new_ratings_data[self.column_names.player_id].extend(player_ids)

        return pl.DataFrame(new_ratings_data, strict=False)

    def _add_rating_columns(self, df: pl.DataFrame, known_features_to_return: list[str],
                            unknown_features_to_return: list[str]) -> pl.DataFrame:
        input_cols = df.columns

        player_rating_col = self.PLAYER_RATING_COL

        if self.column_names.participation_weight:
            df = (df
                  .with_columns(
                (pl.col(self.column_names.participation_weight) * (pl.col(player_rating_col)).alias(
                    '__raw_player_rating')))
                  .with_columns(pl.col(self.column_names.participation_weight).sum().over(
                [self.column_names.match_id, self.column_names.team_id]).alias('__sum_participation_weight'))
                  ).with_columns(
                (pl.col('__raw_player_rating') / pl.col('__sum_participation_weight')).alias(
                    self.TEAM_RATING_COL
                )
            ).drop(['__raw_player_rating', '__sum_participation_weight'])
        else:
            df = df.with_columns(pl.col(player_rating_col).mean().over(
                [self.column_names.team_id, self.column_names.match_id]).alias(self.TEAM_RATING_COL))

        if self.column_names.projected_participation_weight:
            df = (df
                  .with_columns(
                (pl.col(self.column_names.participation_weight) * (pl.col(player_rating_col)).alias(
                    '__raw_player_rating')))
                  .with_columns(pl.col(self.column_names.projected_participation_weight).sum().over(
                [self.column_names.match_id, self.column_names.team_id]).alias('__sum_participation_weight'))
                  ).with_columns(
                (pl.col('__raw_player_rating') / pl.col('__sum_participation_weight')).alias(
                    self.TEAM_RATING_PROJ_COL
                )
            )
        else:
            df = df.with_columns(
                pl.col(self.TEAM_RATING_COL).alias(self.TEAM_RATING_PROJ_COL))

        if self.DIFF_PROJ_COL in known_features_to_return:
            df = df.with_columns((pl.col(self.TEAM_RATING_PROJ_COL) - pl.col(
                self.OPP_RATING_PROJ_COL)).alias(self.DIFF_PROJ_COL))

        # fixed: don't wrap list-of-cols inside another list
        return df.select(list(dict.fromkeys(input_cols + known_features_to_return + unknown_features_to_return)))

    def _create_pre_match_players_collection(
            self, r: dict, stats_col: str
    ) -> PreMatchPlayersCollection:

        pre_match_player_ratings = []
        player_count = 0
        projected_participation_weights = []

        new_match_entities = []
        pre_match_player_ratings_values = []
        player_ids = []

        for team_player in r[stats_col]:
            player_id = team_player[self.column_names.player_id]
            position = team_player.get(self.column_names.position)
            player_league = team_player.get(self.column_names.league, None)
            participation_weight = team_player.get(self.column_names.participation_weight, 1.0)
            projected_participation_weight = team_player.get(self.column_names.projected_participation_weight,
                                                             participation_weight)
            projected_participation_weights.append(projected_participation_weight)
            match_performance = MatchPerformance(
                performance_value=team_player[self.performance_column],
                projected_participation_weight=projected_participation_weight,
                participation_weight=participation_weight,
            )
            player_ids.append(player_id)
            if player_id in self._player_ratings:

                player_rating = self._player_ratings[player_id]

                pre_match_player_rating = PreMatchPlayerRating(
                    id=player_id,
                    rating_value=player_rating.rating_value,
                    match_performance=match_performance,
                    games_played=player_rating.games_played,
                    league=player_league,
                    position=position,
                )

            else:
                new_match_entities.append(
                    MatchPlayer(
                        id=player_id,
                        performance=match_performance,
                        league=player_league,
                        position=position,
                    )
                )
                continue

            player_count += self._player_ratings[player_id].games_played

            pre_match_player_ratings.append(pre_match_player_rating)
            pre_match_player_ratings_values.append(pre_match_player_rating.rating_value)

        return PreMatchPlayersCollection(
            pre_match_player_ratings=pre_match_player_ratings,
            new_players=new_match_entities,
            player_ids=player_ids,
            player_rating_values=pre_match_player_ratings_values,
            projected_particiation_weights=projected_participation_weights,
        )

    def _generate_new_player_pre_match_ratings(
            self,
            day_number: int,
            new_players: list[MatchPlayer],
            team_pre_match_player_ratings: list[PreMatchPlayerRating],
    ) -> tuple[list[PreMatchPlayerRating], list[float]]:

        pre_match_player_ratings = []
        pre_match_player_rating_values = []
        for match_player in new_players:
            id = match_player.id

            rating_value = self.start_rating_generator.generate_rating_value(
                day_number=day_number,
                match_player=match_player,
                team_pre_match_player_ratings=team_pre_match_player_ratings,
            )
            pre_match_player_rating_values.append(rating_value)
            self._player_ratings[match_player.id] = PlayerRating(
                id=id, rating_value=rating_value
            )

            pre_match_player_rating = PreMatchPlayerRating(
                id=id,
                rating_value=rating_value,
                match_performance=match_player.performance,
                games_played=self._player_ratings[match_player.id].games_played,
                league=match_player.league,
                position=match_player.position,
                other=match_player.others,
            )

            pre_match_player_ratings.append(pre_match_player_rating)

        return pre_match_player_ratings, pre_match_player_rating_values

    def _create_match_team_rating_changes(
            self, day_number: int, pre_match_team_ratings: list[PreMatchTeamRating]
    ) -> list[TeamRatingChange]:
        team_rating_changes = []

        for team_idx, pre_match_team_rating in enumerate(pre_match_team_ratings):
            team_rating_change = self._generate_rating_change(
                day_number=day_number,
                pre_match_team_rating=pre_match_team_rating,
                pre_match_opponent_team_rating=pre_match_team_ratings[-team_idx + 1],
            )
            team_rating_changes.append(team_rating_change)

        return team_rating_changes

    def _generate_rating_change(
            self,
            day_number: int,
            pre_match_team_rating: PreMatchTeamRating,
            pre_match_opponent_team_rating: PreMatchTeamRating,
    ) -> TeamRatingChange:

        player_rating_changes = []
        sum_participation_weight = 0
        sum_predicted_performance = 0
        sum_performance_value = 0
        sum_rating_change = 0
        predicted_player_performances = []

        leagues = {}
        for pre_player_rating in pre_match_team_rating.players:

            predicted_performance = self._performance_predictor.predict_performance(
                player_rating=pre_player_rating,
                opponent_team_rating=pre_match_opponent_team_rating,
                team_rating=pre_match_team_rating,
            )
            predicted_player_performances.append(predicted_performance)

            applied_rating_change_multiplier = (
                self._calculate_applied_rating_change_multiplier(
                    player_id=pre_player_rating.id, team_id=pre_match_team_rating.id
                )
            )
            performance_difference = (
                    pre_player_rating.match_performance.performance_value
                    - predicted_performance
            )
            rating_change_value = (
                    performance_difference
                    * applied_rating_change_multiplier
                    * pre_player_rating.match_performance.participation_weight
            )
            if math.isnan(rating_change_value):
                raise ValueError(
                    f"rating_change_value is nan for {pre_player_rating.id}"
                )

            player_rating_change = PlayerRatingChange(
                id=pre_player_rating.id,
                predicted_performance=predicted_performance,
                participation_weight=pre_player_rating.match_performance.participation_weight,
                performance=pre_player_rating.match_performance.performance_value,
                rating_change_value=rating_change_value,
                league=pre_player_rating.league,
                day_number=day_number,
                pre_match_rating_value=pre_player_rating.rating_value,
            )
            player_rating_changes.append(player_rating_change)
            if player_rating_change.league not in leagues:
                leagues[player_rating_change.league] = 0
            leagues[player_rating_change.league] += 1
            sum_predicted_performance += (
                    player_rating_change.predicted_performance
                    * pre_player_rating.match_performance.participation_weight
            )
            sum_performance_value += (
                    pre_player_rating.match_performance.performance_value
                    * pre_player_rating.match_performance.participation_weight
            )
            sum_rating_change += (
                    player_rating_change.rating_change_value
                    * pre_player_rating.match_performance.participation_weight
            )
            sum_participation_weight += (
                pre_player_rating.match_performance.participation_weight
            )

        predicted_performance = (
            sum_predicted_performance / sum_participation_weight
            if sum_participation_weight > 0
            else 0
        )
        performance = (
            sum_performance_value / sum_participation_weight
            if sum_participation_weight > 0
            else 0
        )

        return TeamRatingChange(
            players=player_rating_changes,
            id=pre_match_team_rating.id,
            rating_change_value=sum_rating_change,
            predicted_performance=predicted_performance,
            pre_match_projected_rating_value=pre_match_team_rating.rating_value,
            performance=performance,
            league=max(leagues, key=leagues.get) if leagues else None,
            predicted_player_performances=predicted_player_performances
        )

    def _update_ratings(self, team_rating_changes: list[TeamRatingChange]):
        for idx, team_rating_change in enumerate(team_rating_changes):
            self._update_rating_by_team_rating_change(
                team_rating_change=team_rating_change,
                opponent_team_rating_change=team_rating_changes[-idx + 1],
            )

    def _update_rating_by_team_rating_change(
            self,
            team_rating_change: TeamRatingChange,
            opponent_team_rating_change: TeamRatingChange,
    ) -> None:

        for player_rating_change in team_rating_change.players:
            id = player_rating_change.id
            self._player_ratings[id].confidence_sum = (
                self._calculate_post_match_confidence_sum(
                    entity_rating=self._player_ratings[id],
                    day_number=player_rating_change.day_number,
                    particpation_weight=player_rating_change.participation_weight,
                )
            )

            self._player_ratings[
                id
            ].rating_value += player_rating_change.rating_change_value
            self._player_ratings[
                id
            ].games_played += player_rating_change.participation_weight
            self._player_ratings[id].last_match_day_number = (
                player_rating_change.day_number
            )
            self._player_ratings[id].most_recent_team_id = team_rating_change.id

            if self.column_names.league:
                self.start_rating_generator.update_players_to_leagues(
                    rating_change=player_rating_change
                )
            if player_rating_change.league != opponent_team_rating_change.league:
                self._update_player_ratings_from_league_changes(
                    rating_change=player_rating_change
                )

    def _update_player_ratings_from_league_changes(
            self, rating_change: PlayerRatingChange
    ):

        league = rating_change.league

        if league not in self._league_rating_changes:
            self._league_rating_changes[rating_change.league] = 0
            self._league_rating_changes_count[league] = 0

        rating_change_value = rating_change.rating_change_value
        self._league_rating_changes[rating_change.league] += rating_change_value
        self._league_rating_changes_count[league] += 1

        if self._league_rating_changes[league] > abs(
                self.league_rating_change_update_threshold
        ):
            for player_id in self.start_rating_generator._league_to_player_ids[league]:
                self._player_ratings[player_id].rating_value += (
                        self._league_rating_changes[league]
                        * self.league_rating_adjustor_multiplier
                )

            self._league_rating_changes[league] = 0

    def _add_rating_features(self, df: pl.DataFrame) -> pl.DataFrame:

        def _needs_any(all_feats_out, *names) -> bool:
            return any(n in all_feats_out for n in names)

        non_predictor_features_out = self.non_predictor_features_out if self.non_predictor_features_out else []
        all_feats_out = [*self.features_out, *non_predictor_features_out]

        cn = self.column_names  # ColumnNames instance

        player_rating = self.PLAYER_RATING_COL

        team_col = self.TEAM_RATING_PROJ_COL
        opp_rating_col = self.OPP_RATING_PROJ_COL
        diff_col = self.DIFF_PROJ_COL
        mean_col = self.MEAN_PROJ_COL

        if _needs_any(all_feats_out, team_col, opp_rating_col, diff_col):
            df = add_team_rating_projected(
                df=df,
                column_names=cn,
                player_rating_col=player_rating,
                team_rating_out=team_col,
            )

        if _needs_any(all_feats_out, opp_rating_col, diff_col):
            df = add_opp_team_rating(
                df=df,
                column_names=cn,
                team_rating_col=team_col,
                opp_team_rating_out=opp_rating_col,
            )

        if _needs_any(all_feats_out, self.OPP_RATING_COL, self.DIFF_COL, self.TEAM_RATING_COL):
            df = add_team_rating(
                df=df,
                column_names=cn,
                player_rating_col=player_rating,
                team_rating_out=self.TEAM_RATING_COL,
            )

        if _needs_any(all_feats_out, self.OPP_RATING_COL, self.DIFF_COL):
            df = add_opp_team_rating(
                df=df,
                column_names=cn,
                team_rating_col=self.TEAM_RATING_COL,
                opp_team_rating_out=self.OPP_RATING_COL,
            )

        if diff_col in all_feats_out:
            df = add_rating_difference_projected(
                df=df,
                team_rating_col=team_col,
                rating_diff_out=diff_col,
                opp_team_rating_col=opp_rating_col
            )

        if mean_col in all_feats_out:
            df = add_rating_mean_projected(
                df=df,
                column_names=cn,
                player_rating_col=player_rating,
                rating_mean_out=mean_col,
            )

        base_known = [
            v for k, v in RatingKnownFeatures.__dict__.items()
            if not k.startswith("_")
        ]
        base_unknown = [
            v for k, v in RatingUnknownFeatures.__dict__.items()
            if not k.startswith("_")
        ]
        cols_to_eval = [self._suffix_col(str(c)) for c in (base_known + base_unknown)]
        cols_to_drop = [c for c in cols_to_eval if c not in all_feats_out and c in df.columns]

        return df.drop(cols_to_drop)

    def _future_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        match_df = self._create_match_df(df)
        ratings = self._calculate_future_ratings(match_df)

        cols = [
            c for c in df.columns
            if c not in (self.PLAYER_PRED_PERF_COL, self.PLAYER_RATING_COL)
        ]

        df = (
            df.select(cols)
            .join(
                ratings,
                on=[self.column_names.player_id, self.column_names.match_id, self.column_names.team_id],
            )
        )

        return self._add_rating_features(df)

    def _create_match_df(self, df: pl.DataFrame) -> pl.DataFrame:

        if self.league_identifier:
            df = self.league_identifier.add_leagues(df)

        # Build player stats struct (same as historical)
        player_stat_cols = [self.column_names.player_id]
        # performance column may be missing/null on future fixtures; include it if present
        if self.performance_column in df.columns:
            player_stat_cols.append(self.performance_column)

        if self.column_names.participation_weight and self.column_names.participation_weight in df.columns:
            player_stat_cols.append(self.column_names.participation_weight)

        if (
                self.column_names.projected_participation_weight
                and self.column_names.projected_participation_weight in df.columns
        ):
            player_stat_cols.append(self.column_names.projected_participation_weight)

        if self.column_names.position and self.column_names.position in df.columns:
            player_stat_cols.append(self.column_names.position)

        if self.column_names.league and self.column_names.league in df.columns:
            player_stat_cols.append(self.column_names.league)

        df = df.with_columns(pl.struct(player_stat_cols).alias(PLAYER_STATS))

        agg_df = (
            df.group_by([self.column_names.match_id, self.column_names.team_id, self.column_names.start_date])
            .agg(pl.col(PLAYER_STATS).alias(PLAYER_STATS))
        )
        team_id_col = self.column_names.team_id

        match_df = (
            agg_df.join(
                agg_df,
                on=self.column_names.match_id,
                how="inner",
                suffix="_opponent",
            )
            # correct column name after suffix:
            .filter(pl.col(team_id_col) != pl.col(f"{team_id_col}_opponent"))
            # create stable opponent id column:
            .with_columns(pl.col(f"{team_id_col}_opponent").alias("__team_id_opponent"))
            .unique(self.column_names.match_id)
            .sort(list(set([
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.update_match_id,
            ])))
        )

        start_as_int = (
            pl.col(self.column_names.start_date)
            .str.strptime(pl.Datetime, strict=False)
            .cast(pl.Date)
            .cast(pl.Int32)
        )
        return match_df.with_columns((start_as_int - start_as_int.min() + 1).alias("__day_number"))

    def _calculate_future_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        """
        For each future match:
        - Use current player ratings (no updates)
        - Compute per-player pre-match rating + predicted performance
        - Return a dataframe keyed by (player_id, match_id, team_id)
        """
        # IMPORTANT: do not mutate self._player_ratings
        base_ratings = self._player_ratings
        # Local-only cache for unseen players so repeated future fixtures are consistent
        local_ratings: dict[str, PlayerRating] = copy.deepcopy(base_ratings)

        new_ratings_data = {
            self.column_names.player_id: [],
            self.column_names.match_id: [],
            self.column_names.team_id: [],
            self.PLAYER_RATING_COL: [],
            self.PLAYER_PRED_PERF_COL: [],
        }

        cn = self.column_names

        def get_perf_value(team_player: dict) -> float:
            # future fixtures may not have performance; it doesn't affect prediction
            if self.performance_column in team_player and team_player[self.performance_column] is not None:
                return float(team_player[self.performance_column])
            return 0.0

        def make_collection(r: dict, stats_col: str) -> PreMatchPlayersCollection:
            pre_match_player_ratings: list[PreMatchPlayerRating] = []
            new_players: list[MatchPlayer] = []
            player_ids: list[str] = []
            player_rating_values: list[float] = []
            projected_weights: list[float] = []

            for team_player in r[stats_col]:
                player_id = team_player[cn.player_id]
                player_ids.append(player_id)

                position = team_player.get(cn.position)
                player_league = team_player.get(cn.league, None)

                participation_weight = team_player.get(cn.participation_weight, 1.0) if cn.participation_weight else 1.0
                proj_participation_weight = (
                    team_player.get(cn.projected_participation_weight, participation_weight)
                    if cn.projected_participation_weight
                    else participation_weight
                )
                projected_weights.append(proj_participation_weight)

                mp = MatchPerformance(
                    performance_value=get_perf_value(team_player),
                    projected_participation_weight=proj_participation_weight,
                    participation_weight=participation_weight,
                )

                if player_id in local_ratings:
                    pr = local_ratings[player_id]
                    pre = PreMatchPlayerRating(
                        id=player_id,
                        rating_value=pr.rating_value,
                        match_performance=mp,
                        games_played=pr.games_played,
                        league=player_league,
                        position=position,
                    )
                    pre_match_player_ratings.append(pre)
                    player_rating_values.append(pre.rating_value)
                else:
                    # create "new player" entry for start rating generation (LOCAL ONLY)
                    new_players.append(
                        MatchPlayer(
                            id=player_id,
                            performance=mp,
                            league=player_league,
                            position=position,
                        )
                    )

            return PreMatchPlayersCollection(
                pre_match_player_ratings=pre_match_player_ratings,
                new_players=new_players,
                player_ids=player_ids,
                player_rating_values=player_rating_values,
                projected_particiation_weights=projected_weights,
            )

        for r in match_df.iter_rows(named=True):
            team1 = r[cn.team_id]
            team2 = r[f"{cn.team_id}_opponent"]
            day_number = r["__day_number"]

            c1 = make_collection(r, PLAYER_STATS)
            c2 = make_collection(r, f"{PLAYER_STATS}_opponent")

            # Generate start ratings for unseen players (LOCAL ONLY, no updates from match outcomes)
            if c1.new_players:
                new_pre_1, new_vals_1 = self._generate_new_player_pre_match_ratings(
                    day_number=day_number,
                    new_players=c1.new_players,
                    team_pre_match_player_ratings=c1.pre_match_player_ratings,
                )
                # _generate_new_player_pre_match_ratings writes to self._player_ratings; prevent that:
                for p in c1.new_players:
                    local_ratings[p.id] = self._player_ratings[p.id]
                    del self._player_ratings[p.id]

                c1.pre_match_player_ratings.extend(new_pre_1)
                c1.player_rating_values.extend(new_vals_1)

            if c2.new_players:
                new_pre_2, new_vals_2 = self._generate_new_player_pre_match_ratings(
                    day_number=day_number,
                    new_players=c2.new_players,
                    team_pre_match_player_ratings=c2.pre_match_player_ratings,
                )
                for p in c2.new_players:
                    local_ratings[p.id] = self._player_ratings[p.id]
                    del self._player_ratings[p.id]

                c2.pre_match_player_ratings.extend(new_pre_2)
                c2.player_rating_values.extend(new_vals_2)

            # Team rating values (projected weights if available)
            if cn.projected_participation_weight:
                w1 = c1.projected_particiation_weights
                w2 = c2.projected_particiation_weights
                team1_rating_value = (sum(x * y for x, y in zip(c1.player_rating_values, w1)) / sum(w1)) if sum(
                    w1) else 0.0
                team2_rating_value = (sum(x * y for x, y in zip(c2.player_rating_values, w2)) / sum(w2)) if sum(
                    w2) else 0.0
            else:
                team1_rating_value = (sum(c1.player_rating_values) / len(
                    c1.player_rating_values)) if c1.player_rating_values else 0.0
                team2_rating_value = (sum(c2.player_rating_values) / len(
                    c2.player_rating_values)) if c2.player_rating_values else 0.0

            pre_match_team_ratings = [
                PreMatchTeamRating(
                    id=team1,
                    players=c1.pre_match_player_ratings,
                    rating_value=team1_rating_value,
                ),
                PreMatchTeamRating(
                    id=team2,
                    players=c2.pre_match_player_ratings,
                    rating_value=team2_rating_value,
                ),
            ]

            # Compute predicted player performances WITHOUT applying rating updates
            predicted_p1 = [
                self._performance_predictor.predict_performance(
                    player_rating=pre_player,
                    opponent_team_rating=pre_match_team_ratings[1],
                    team_rating=pre_match_team_ratings[0],
                )
                for pre_player in pre_match_team_ratings[0].players
            ]
            predicted_p2 = [
                self._performance_predictor.predict_performance(
                    player_rating=pre_player,
                    opponent_team_rating=pre_match_team_ratings[0],
                    team_rating=pre_match_team_ratings[1],
                )
                for pre_player in pre_match_team_ratings[1].players
            ]

            # Emit per-player rows
            player_ratings = c1.player_rating_values + c2.player_rating_values
            predicted_player_performances = predicted_p1 + predicted_p2
            player_ids = c1.player_ids + c2.player_ids
            team_ids = [team1] * len(c1.player_ids) + [team2] * len(c2.player_ids)
            match_ids = [r[cn.match_id]] * len(player_ids)

            new_ratings_data[cn.player_id].extend(player_ids)
            new_ratings_data[cn.match_id].extend(match_ids)
            new_ratings_data[cn.team_id].extend(team_ids)
            new_ratings_data[self.PLAYER_RATING_COL].extend(player_ratings)
            new_ratings_data[self.PLAYER_PRED_PERF_COL].extend(predicted_player_performances)

        return pl.DataFrame(new_ratings_data, strict=False)
