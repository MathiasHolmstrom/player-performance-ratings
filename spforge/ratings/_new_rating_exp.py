import inspect
import logging
import math
from typing import Tuple, Optional, Union, Any, Literal

import polars as pl
import time

from spforge.data_structures import MatchPlayer, MatchPerformance, MatchTeam, PreMatchPlayerRating, PlayerRating, \
    PlayerRatingChange, TeamRatingChange, PreMatchTeamRating, ColumnNames, PreMatchPlayersCollection
from spforge.ratings import StartRatingGenerator, RatingDifferencePerformancePredictor, RatingKnownFeatures, \
    RatingUnknownFeatures
from spforge.ratings.league_identifier import LeagueIdentifer2
from spforge.ratings.rating_calculators.match_rating_generator import MATCH_CONTRIBUTION_TO_SUM_VALUE, \
    EXPECTED_MEAN_CONFIDENCE_SUM
from spforge.ratings.rating_calculators.performance_predictor import PerformancePredictor, \
    RatingMeanPerformancePredictor, RatingNonOpponentPerformancePredictor
from spforge.transformers.fit_transformers import PerformanceWeightsManager
from spforge.transformers.fit_transformers._performance_manager import ColumnWeight, PerformanceManager


class PlayerRatingGeneratorNew():

    def __init__(self,
                 performance_column: str,
                 performance_weights: Optional[
                     list[Union[ColumnWeight, dict[str, float]]]
                 ] = None,
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
                 **kwargs):

        self.performance_predictor = performance_predictor
        self.features_out = features_out
        if self.performance_predictor == 'mean':
            _performance_predictor_class = RatingMeanPerformancePredictor
            self.features_out = [
                RatingKnownFeatures.RATING_MEAN_PROJECTED] if self.features_out is None else self.features_out
        elif self.performance_predictor == 'difference':
            _performance_predictor_class = RatingDifferencePerformancePredictor
            self.features_out = [
                RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED] if self.features_out is None else self.features_out
        elif self.performance_predictor == 'ignore_opponent':
            _performance_predictor_class = RatingNonOpponentPerformancePredictor
            self.features_out = [
                RatingKnownFeatures.TEAM_RATING_PROJECTED] if self.features_out is None else self.features_out
        else:
            raise ValueError(f"performance_predictor {self.performance_predictor} is not supported")
        sig = inspect.signature(_performance_predictor_class.__init__)

        init_params = [
            name for name, param in sig.parameters.items()
            if name != "self"
        ]

        performance_predictor_params = {k: v for k, v in kwargs.items() if k in init_params}
        self._performance_predictor = _performance_predictor_class(**performance_predictor_params)

        self.non_predictor_features_out = non_predictor_features_out
        self.confidence_weight = confidence_weight
        self.performance_column = performance_column
        self.kwargs = kwargs
        self.confidence_days_ago_multiplier = confidence_days_ago_multiplier
        self.confidence_value_denom = confidence_value_denom
        self.min_rating_change_multiplier_ratio = min_rating_change_multiplier_ratio
        self.rating_change_multiplier = rating_change_multiplier
        self.confidence_max_sum = confidence_max_sum
        self.league_rating_adjustor_multiplier = league_rating_adjustor_multiplier
        self.league_rating_change_update_threshold = (
            league_rating_change_update_threshold
        )
        self.performance_manager = None
        self.league_identifier = None
        self.start_league_ratings = start_league_ratings
        self.start_league_quantile = start_league_quantile
        self.start_min_count_for_percentiles = start_min_count_for_percentiles
        self.start_team_rating_subtract = start_team_rating_subtract
        self.start_team_weight = start_team_weight
        self.start_max_days_ago_league_entities = start_max_days_ago_league_entities
        self.start_min_match_count_team_rating = start_min_match_count_team_rating
        self.start_hardcoded_start_rating = start_harcoded_start_rating
        self.confidence_max_days = confidence_max_days
        self._league_rating_changes: dict[Optional[str], float] = {}
        self._league_rating_changes_count: dict[str, float] = {}
        self.performance_weights = performance_weights
        self.auto_scale_performance = auto_scale_performance

        self.team_id_change_confidence_sum_decrease = (
            team_id_change_confidence_sum_decrease
        )
        self.column_names = column_names
        self._player_ratings: dict[str, PlayerRating] = {}

    def fit_transform(self, df: pl.DataFrame, column_names: Optional[ColumnNames] = None) -> pl.DataFrame:
        self.column_names = column_names if column_names else self.column_names
        if self.column_names.league:
            self.league_identifier = LeagueIdentifer2(column_names=self.column_names)
        if self.performance_weights:
            if isinstance(self.performance_weights[0], dict):
                self.performance_weights = [
                    ColumnWeight(**weight) for weight in self.performance_weights
                ]

            self.performance_manager = PerformanceWeightsManager(
                weights=self.performance_weights,
                performance_column=self.performance_column

            )
        else:
            assert self.performance_column in df.columns, f"{self.performance_column} not in df. If performance_weights are not set, performance_column must exist in dataframe"

        if self.auto_scale_performance:
            assert (
                self.performance_column
            ), "performance_column must be set if auto_scale_performance is True"
            if not self.performance_weights:
                self.performance_manager = PerformanceManager(
                    features=[self.performance_column],
                    performance_column=self.performance_column
                )
            else:
                self.performance_manager = PerformanceWeightsManager(
                    weights=self.performance_weights,
                    performance_column=self.performance_column
                )
            logging.info(
                f"Renamed performance column to performance_{self.performance_column}"
            )

        self.start_rating_generator = StartRatingGenerator(
            league_ratings=self.start_league_ratings,
            league_quantile=self.start_league_quantile,
            min_match_count_team_rating=self.start_min_match_count_team_rating,
            team_weight=self.start_team_weight,
            team_rating_subtract=self.start_team_rating_subtract,
            max_days_ago_league_entities=self.start_max_days_ago_league_entities,
            min_count_for_percentiles=self.start_min_count_for_percentiles,
            harcoded_start_rating=self.start_hardcoded_start_rating
        )
        if self.performance_manager:
            df = self.performance_manager.fit_transform(df)
        if df[self.performance_column].max() > 1.02 or df[self.performance_column].min() < -0.02:
            raise ValueError(
                f"Max {self.performance_column} must be less than than 1.02 and min value larger than -0.02. Either transform it manually or set auto_scale_performance to True")

        if df[self.performance_column].mean() < 0.42 or df[self.performance_column].mean() > 0.58:
            raise ValueError(
                f"Mean {self.performance_column} must be between 0.42 and 0.58. Either transform it manually or set auto_scale_performance to True")
        return self._transform(df)

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.league_identifier:
            df = self.league_identifier.add_leagues(df)

        df = df.with_columns([
            pl.struct([self.column_names.player_id, self.performance_column]).alias("player_struct")
        ])

        agg_df = df.group_by([self.column_names.match_id, self.column_names.team_id, self.column_names.start_date]).agg(
            [
                pl.col("player_struct").alias("players_with_stats"),
            ]
        )

        match_df = agg_df.join(
            agg_df,
            on=self.column_names.match_id,
            how="inner",
            suffix="_opponent"
        ).filter(
            pl.col(self.column_names.team_id) != pl.col("team_id_opponent")
        ).unique(self.column_names.match_id).sort(
            list(set([self.column_names.start_date, self.column_names.match_id, self.column_names.update_match_id])))
        match_df = match_df.with_columns(
            (pl.col(self.column_names.start_date).str.strptime(pl.Date, "%Y-%m-%d").cast(pl.Int32)
             - pl.col(self.column_names.start_date).str.strptime(pl.Date, "%Y-%m-%d").cast(pl.Int32).min()
             + 1).alias("__day_number")
        )
        ratings = self._calculate_ratings(match_df)
        df = df[[c for c in df.columns if c != RatingKnownFeatures.PLAYER_RATING]].join(ratings,
                                                                                        on=[self.column_names.player_id,
                                                                                            self.column_names.match_id,
                                                                                            self.column_names.team_id])

        return self._add_rating_features(df)

    def _calculate_ratings(self, match_df: pl.DataFrame) -> pl.DataFrame:
        team_rating_changes = []
        update_ids = []
        new_ratings_data = {
            self.column_names.player_id: [],
            self.column_names.match_id: [],
            self.column_names.team_id: [],
            RatingKnownFeatures.PLAYER_RATING: [],
            RatingUnknownFeatures.PLAYER_PREDICTED_PERFORMANCE: []
        }
        for r in match_df.iter_rows(named=True):
            team1 = r[self.column_names.team_id]
            team2 = r[f"{self.column_names.team_id}_opponent"]
            update_ids.append(r[self.column_names.update_match_id])
            pre_match_players_collection_team1 = self._create_pre_match_players_collection(
                r=r,
                players_col_name='players_with_stats')
            pre_match_players_collection_team2 = self._create_pre_match_players_collection(
                r=r,
                players_col_name='players_with_stats_opponent')
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
                if self.column_names.projected_participation_weight:
                    team1_proj_weights = pre_match_players_collection_team1.projected_particiation_weights
                    team2_proj_weights = pre_match_players_collection_team2.projected_particiation_weights
                else:
                    team1_proj_weights = pre_match_players_collection_team1.projected_particiation_weights
                    team2_proj_weights = pre_match_players_collection_team2.projected_particiation_weights

                team1_rating_value = sum(x * y for x, y in zip(player_rating_values_team1,
                                                               team1_proj_weights) / sum(team1_proj_weights))
                team2_rating_value = sum(x * y for x, y in zip(player_rating_values_team2,
                                                               team2_proj_weights) / sum(team2_proj_weights))
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
            match_ids =[r[self.column_names.match_id]] * (
                              len(player_rating_values_team1) + len(player_rating_values_team2))
            team_ids = [team1] * len(player_rating_values_team1) + [team2] * len(player_rating_values_team2)
            predicted_player_performances = match_team_rating_changes[0].predicted_player_performances + \
                                            match_team_rating_changes[1].predicted_player_performances
            player_ids = pre_match_players_collection_team1.player_ids + pre_match_players_collection_team2.player_ids
            new_ratings_data[RatingKnownFeatures.PLAYER_RATING].extend(player_ratings)
            new_ratings_data[self.column_names.match_id].extend(match_ids)
            new_ratings_data[self.column_names.team_id].extend(team_ids)
            new_ratings_data[RatingUnknownFeatures.PLAYER_PREDICTED_PERFORMANCE].extend(predicted_player_performances)

            new_ratings_data[self.column_names.player_id].extend(player_ids)

        return pl.DataFrame(new_ratings_data, strict=False)

    def _add_rating_columns(self, df: pl.DataFrame, known_features_to_return: list[str],
                            unknown_features_to_return: list[str]) -> pl.DataFrame:
        input_cols = df.columns
        if self.column_names.participation_weight:
            df = (df
                  .with_columns(
                (pl.col(self.column_names.participation_weight) * (pl.col(RatingKnownFeatures.PLAYER_RATING)).alias(
                    '__raw_player_rating')))
                  .with_columns(pl.col(self.column_names.participation_weight).sum().over(
                [self.column_names.match_id, self.column_names.team_id]).alias('__sum_participation_weight'))
                  ).with_columns(
                (pl.col('__raw_player_rating') / pl.col('__sum_participation_weight')).alias(
                    RatingUnknownFeatures.TEAM_RATING)
            ).drop(['__raw_player_rating', '__sum_participation_weight'])
        else:
            df = df.with_columns(pl.col(RatingKnownFeatures.PLAYER_RATING).mean().over(
                [self.column_names.team_id, self.column_names.match_id]).alias(RatingUnknownFeatures.TEAM_RATING))

        if self.column_names.projected_participation_weight:
            df = (df
                  .with_columns(
                (pl.col(self.column_names.participation_weight) * (pl.col(RatingKnownFeatures.PLAYER_RATING)).alias(
                    '__raw_player_rating')))
                  .with_columns(pl.col(self.column_names.projected_participation_weight).sum().over(
                [self.column_names.match_id, self.column_names.team_id]).alias('__sum_participation_weight'))
                  ).with_columns(
                (pl.col('__raw_player_rating') / pl.col('__sum_participation_weight')).alias(
                    RatingKnownFeatures.TEAM_RATING_PROJECTED)
            )
        else:
            df = df.with_columns(
                pl.col(RatingUnknownFeatures.TEAM_RATING).alias(RatingKnownFeatures.TEAM_RATING_PROJECTED))

        if RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED in known_features_to_return:
            df = df.with_columns((pl.col(RatingKnownFeatures.TEAM_RATING_PROJECTED) - pl.col(
                RatingKnownFeatures.OPPONENT_RATING_PROJECTED)).alias(RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED))

        return df.select(list(set([input_cols + known_features_to_return + unknown_features_to_return])))

    def _create_pre_match_players_collection(
            self, r: dict, players_col_name: str
    ) -> PreMatchPlayersCollection:

        pre_match_player_ratings = []
        player_count = 0
        projected_participation_weights = []

        new_match_entities = []
        pre_match_player_ratings_values = []
        player_ids = []

        for team_player in r[players_col_name]:
            player_id = team_player[self.column_names.player_id]
            position = team_player.get(self.column_names.position)
            player_league = team_player.get(self.column_names.league, None)
            projected_participation_weight = team_player.get(self.column_names.projected_participation_weight, 1.0)
            projected_participation_weights.append(projected_participation_weight)
            participation_weight = team_player.get(self.column_names.participation_weight, 1.0)
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

    def _calculate_applied_rating_change_multiplier(
            self,
            player_id: str,
            team_id: str,
    ) -> float:
        if (
                self._player_ratings[player_id].most_recent_team_id
                and self._player_ratings[player_id].most_recent_team_id != team_id
        ):
            self._player_ratings[
                player_id
            ].confidence_sum -= self.team_id_change_confidence_sum_decrease

        min_applied_rating_change_multiplier = (
                self.rating_change_multiplier * self.min_rating_change_multiplier_ratio
        )
        confidence_change_multiplier = self.rating_change_multiplier * (
                (
                        EXPECTED_MEAN_CONFIDENCE_SUM
                        - self._player_ratings[player_id].confidence_sum
                )
                / self.confidence_value_denom
                + 1
        )
        applied_rating_change_multiplier = (
                confidence_change_multiplier * self.confidence_weight
                + (1 - self.confidence_weight) * self.rating_change_multiplier
        )

        return max(
            min_applied_rating_change_multiplier, applied_rating_change_multiplier
        )

    #     return max(min_applied_rating_change_multiplier, applied_rating_change_multiplier)

    def _calculate_post_match_confidence_sum(
            self, entity_rating: PlayerRating, day_number: int, particpation_weight: float
    ) -> float:
        days_ago = self._calculate_days_ago_since_last_match(
            last_match_day_number=entity_rating.last_match_day_number,
            day_number=day_number,
        )
        confidence_sum_value = (
                -min(days_ago, self.confidence_max_days)
                * self.confidence_days_ago_multiplier
                + entity_rating.confidence_sum
                + MATCH_CONTRIBUTION_TO_SUM_VALUE * particpation_weight
        )

        return max(0.0, min(confidence_sum_value, self.confidence_max_sum))

    def _calculate_days_ago_since_last_match(
            self, last_match_day_number, day_number: int
    ) -> float:
        match_day_number = day_number
        if last_match_day_number is None:
            return 0.0

        return match_day_number - last_match_day_number

    def _get_rating_by_id(self, id: str):
        if id not in self._player_ratings:
            raise KeyError(f"{id} not in player_ratings")
        return self._player_ratings[id]

    def _add_rating_features(self, df: pl.DataFrame) -> pl.DataFrame:
        all_feats_out = [*self.features_out, *self.non_predictor_features_out]
        match_id = self.column_names.match_id
        team_id = self.column_names.team_id
        team_col = RatingKnownFeatures.TEAM_RATING_PROJECTED
        ppw = self.column_names.projected_participation_weight
        opp_col = "OPP_TEAM_RATING_PROJECTED"
        if (RatingKnownFeatures.TEAM_RATING_PROJECTED in all_feats_out
            or RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED in all_feats_out) or RatingKnownFeatures.OPPONENT_RATING_PROJECTED in all_feats_out:
            if self.column_names.projected_participation_weight:
                df = df.with_columns(
                    (
                            (pl.col(ppw) * pl.col(RatingKnownFeatures.PLAYER_RATING)).sum().over([match_id, team_id])
                            / pl.col(ppw).sum().over([match_id, team_id])
                    ).alias(RatingKnownFeatures.TEAM_RATING_PROJECTED)
                )
            else:
                df = df.with_columns(
                    pl.col(RatingKnownFeatures.PLAYER_RATING)
                    .mean()
                    .over([self.column_names.match_id, self.column_names.team_id])
                    .alias(RatingKnownFeatures.TEAM_RATING_PROJECTED)
                )
        if RatingKnownFeatures.OPPONENT_RATING_PROJECTED or RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED in all_feats_out:
            team_sums = df.select([pl.col(match_id), pl.col(team_id), pl.col(team_col)]).unique()

            opp_map = (
                team_sums.join(team_sums, on=match_id, suffix="_opp")
                .filter(pl.col(team_id) != pl.col(f"{team_id}_opp"))
                .select(
                    pl.col(match_id),
                    pl.col(team_id),
                    pl.col(f"{team_col}_opp").alias(opp_col),
                )
            )

            df = df.join(opp_map, on=[match_id, team_id], how="left")

        return df


if __name__ == '__main__':
    df = pl.read_parquet(
        r"C:\Users\Admin\PycharmProjects\spforge\examples\nba\data\game_player_subsample.parquet")
    column_names = ColumnNames(
        team_id="team_id",
        match_id="game_id",
        start_date="start_date",
        player_id="player_id",
    )
    player_rating_generator = PlayerRatingGeneratorNew(column_names=column_names, auto_scale_performance=True,
                                                       performance_column='won', rating_diff_coef=0.0056,
                                                       non_predictor_features_out=[
                                                           RatingKnownFeatures.OPPONENT_RATING_PROJECTED])
    st = time.time()
    df = player_rating_generator.fit_transform(df)
    u = 3
    print(time.time() - st)
