import copy
import inspect
import logging
from dataclasses import dataclass

import pandas as pd
from player_performance_ratings.data_structures import Match, ColumnNames

from player_performance_ratings.cross_validator._base import CrossValidator

from player_performance_ratings import PipelineFactory
from player_performance_ratings.ratings import UpdateRatingGenerator
from player_performance_ratings.ratings.enums import (
    RatingKnownFeatures,
    RatingHistoricalFeatures,
)
from player_performance_ratings.ratings.rating_calculators.start_rating_generator import (
    StartRatingGenerator,
)


@dataclass
class LeagueH2H:
    league: str
    opp_league: str
    weight: float
    mean_rating_change: float


class StartLeagueRatingOptimizer:

    def __init__(
            self,
            pipeline_factory: PipelineFactory,
            cross_validator: CrossValidator,
            max_iterations: int = 10,
            learning_step: int = 20,
            weight_div: int = 500,
            indirect_weight: float = 1.5,
            verbose: int = 1,
    ):
        self.pipeline_factory = pipeline_factory
        self.max_iterations = max_iterations
        self.learning_step = learning_step
        self.weight_div = weight_div
        self.cross_validator = cross_validator
        self.indirect_weight = indirect_weight
        self.verbose = verbose
        self._scores = []
        self._league_ratings_iterations = []

    def optimize(
            self,
            df: pd.DataFrame,
            rating_model_idx: int,
            rating_generator: UpdateRatingGenerator,
            matches: list[Match],
            column_names: ColumnNames,
    ) -> dict[str, float]:

        start_rating_generator = (
            rating_generator.match_rating_generator.start_rating_generator
        )
        league_ratings = start_rating_generator.league_ratings.copy()
        start_rating_params = list(
            inspect.signature(
                start_rating_generator.__class__.__init__
            ).parameters.keys()
        )[1:]
        non_league_ratings_params = {
            attr: getattr(start_rating_generator, attr)
            for attr in dir(start_rating_generator)
            if attr in start_rating_params and attr != "league_ratings"
        }

        for iteration in range(self.max_iterations + 1):

            rating_generator_used = copy.deepcopy(rating_generator)

            start_rating_generator = StartRatingGenerator(
                league_ratings=league_ratings, **non_league_ratings_params
            )

            if iteration == self.max_iterations:
                min_idx = self._scores.index(min(self._scores))
                logging.info(
                    f"best start rating params {self._league_ratings_iterations[min_idx]}"
                )
                return self._league_ratings_iterations[min_idx]

            rating_generator_used.match_rating_generator.start_rating_generator = (
                start_rating_generator
            )

            rating_generators = copy.deepcopy(self.pipeline_factory.rating_generators)
            rating_generators[rating_model_idx] = rating_generator_used

            rating_values = rating_generators[
                rating_model_idx
            ].generate_historical_by_matches(
                matches=matches,
                column_names=column_names,
                historical_features_out=[RatingHistoricalFeatures.PLAYER_RATING_CHANGE],
                known_features_out=[
                    RatingKnownFeatures.PLAYER_RATING,
                    RatingKnownFeatures.OPPONENT_LEAGUE,
                    RatingKnownFeatures.PLAYER_LEAGUE,
                ],
            )

            df_with_ratings = df.assign(**rating_values)

            rating_generators = copy.deepcopy(self.pipeline_factory.rating_generators)
            rating_generators[
                rating_model_idx
            ].match_rating_generator.start_rating_generator = start_rating_generator
            for r in rating_generators:
                r.reset_ratings()
            pipeline = self.pipeline_factory.create(rating_generators=rating_generators)
            score = pipeline.cross_validate_score(
                df=df_with_ratings,
                cross_validator=self.cross_validator,
                create_performance=False,
                create_rating_features=False,
            )

            if not league_ratings:
                league_ratings = pipeline.rating_generators[
                    rating_model_idx
                ].match_rating_generator.start_rating_generator.league_ratings

            self._league_ratings_iterations.append(copy.deepcopy(league_ratings))
            self._scores.append(score)
            if self.verbose:
                logging.info(
                    f"iteration {iteration} finished. Score: {score}. best startings {league_ratings}"
                )

            league_rating_changes = (
                df_with_ratings.groupby(
                    [
                        RatingKnownFeatures.PLAYER_LEAGUE,
                        RatingKnownFeatures.OPPONENT_LEAGUE,
                    ]
                )
                .agg(
                    {
                        RatingHistoricalFeatures.PLAYER_RATING_CHANGE: "mean",
                        column_names.player_id: "count",
                    }
                )
                .reset_index()
                .rename(
                    columns={
                        RatingHistoricalFeatures.PLAYER_RATING_CHANGE: "mean_rating_change",
                        column_names.player_id: "count",
                    }
                )
            )
            leagues = (
                league_rating_changes[RatingKnownFeatures.PLAYER_LEAGUE]
                .unique()
                .tolist()
            )

            league_opp_league_h2hs: dict[str, dict[str, LeagueH2H]] = {}
            league_to_played_against_leagues = {}

            for league in leagues:
                league_to_played_against_leagues[league] = (
                    league_rating_changes[
                        league_rating_changes[RatingKnownFeatures.PLAYER_LEAGUE]
                        == league
                        ][RatingKnownFeatures.OPPONENT_LEAGUE]
                    .unique()
                    .tolist()
                )

                league_opp_league_h2hs[league] = {}
                for opp_league in leagues:
                    if league == opp_league:
                        continue

                    rows = league_rating_changes[
                        (
                                league_rating_changes[RatingKnownFeatures.PLAYER_LEAGUE]
                                == league
                        )
                        & (
                                league_rating_changes[RatingKnownFeatures.OPPONENT_LEAGUE]
                                == opp_league
                        )
                        ]
                    if len(rows) == 0:
                        weight = 0
                        mean_rating_change = 0
                    else:
                        count = rows["count"].iloc[0]
                        weight = min(1, count / self.weight_div)
                        mean_rating_change = rows["mean_rating_change"].iloc[0]

                    league_h2h = LeagueH2H(
                        weight=weight,
                        mean_rating_change=mean_rating_change,
                        league=league,
                        opp_league=opp_league,
                    )
                    league_opp_league_h2hs[league][opp_league] = league_h2h

            final_league_opp_league_h2hs: dict[str, dict[str, LeagueH2H]] = {}

            league_sum_final_weights = {}
            for league, opp_league_h2h in league_opp_league_h2hs.items():

                league_sum_final_weights[league] = 0
                final_league_opp_league_h2hs[league] = {}
                for opp_league, h2h in opp_league_h2h.items():
                    shared_weight = 0
                    relative_shared_weighted_mean = 0
                    if h2h.weight < 1:
                        league_against = league_to_played_against_leagues[league]
                        opp_played_against = league_to_played_against_leagues[
                            opp_league
                        ]
                        shared_leagues = [
                            l
                            for l in league_against
                            if l in opp_played_against and l not in (league, opp_league)
                        ]

                        for shared_league in shared_leagues:
                            opp_vs_shared_h2h = league_opp_league_h2hs[opp_league][
                                shared_league
                            ]
                            league_vs_shared_h2h = league_opp_league_h2hs[league][
                                shared_league
                            ]

                            sum_rating_change = (
                                    league_vs_shared_h2h.mean_rating_change
                                    - opp_vs_shared_h2h.mean_rating_change
                            )

                            relative_shared_weighted_mean += (
                                    min(
                                        opp_vs_shared_h2h.weight,
                                        league_vs_shared_h2h.weight,
                                    )
                                    * sum_rating_change
                            )
                            shared_weight += min(
                                opp_vs_shared_h2h.weight, league_vs_shared_h2h.weight
                            )

                    weighted_mean = (
                            h2h.mean_rating_change * h2h.weight
                            + min(1, shared_weight * self.indirect_weight)
                            * relative_shared_weighted_mean
                    )
                    final_weight = min(1, h2h.weight + shared_weight)
                    final_league_h2h = LeagueH2H(
                        weight=final_weight,
                        mean_rating_change=weighted_mean,
                        league=league,
                        opp_league=opp_league,
                    )
                    final_league_opp_league_h2hs[league][opp_league] = final_league_h2h
                    league_sum_final_weights[league] += final_weight

            for league, final_opp_league_h2h in final_league_opp_league_h2hs.items():

                if league not in league_ratings:
                    continue
                start_rating_rating_change = 0

                for opp_league, final_h2h in final_opp_league_h2h.items():
                    if league_sum_final_weights[league] > 0:
                        start_rating_rating_change += (
                                final_h2h.weight
                                / league_sum_final_weights[league]
                                * final_h2h.mean_rating_change
                                * self.learning_step
                        )

                league_ratings[league] += start_rating_rating_change

        return league_ratings
