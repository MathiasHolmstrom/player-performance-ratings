from src.player_performance_ratings.enums import PredictedRatingMethod
from src.player_performance_ratings.match_rating.match_rating_calculator import RatingMeanPerformancePredictor, \
    PerformancePredictor


class MatchGeneratorFactory():

    def create(self, predicted_rating_method: PredictedRatingMethod) :

        if predicted_rating_method == PredictedRatingMethod.MEAN_RATING:
            performance_predictor = RatingMeanPerformancePredictor(
                rating_diff_coef=self.rating_diff_coef,
                rating_diff_team_from_entity_coef=self.rating_diff_team_from_entity_coef,
                team_rating_diff_coef=self.team_rating_diff_coef
            )
        else:
            performance_predictor = PerformancePredictor(
                rating_diff_coef=self.rating_diff_coef,
                rating_diff_team_from_entity_coef=self.rating_diff_team_from_entity_coef,
                team_rating_diff_coef=self.team_rating_diff_coef
            )

        league_rating_adjustor: LeagueRatingAdjustor = LeagueRatingAdjustor(
            league_rating_regularizer=self.league_rating_regularizer)
        self.match_id_to_out_df_column_values: Dict[str, MatchOutValues] = {}

        start_rating_calculator = StartRatingCalculator(
            min_count_using_percentiles=self.min_count_using_percentiles,
            league_quantile=self.start_rating_league_quantile,
            start_league_ratings=self.start_league_ratings,
            team_rating_subtract=self.start_rating_team_rating_subtract,
            team_weight=self.start_rating_team_weight,
            min_match_ratings_for_team=self.min_match_ratings_for_team,
            max_days_ago_league_entities=self.max_days_ago_league_entities,
        )

        match_rating_calculator = DefaultMatchRatingCalculator(
            start_rating_calculator=start_rating_calculator,
            offense_rating_change_multiplier=self.offense_rating_change_multiplier,
            defense_rating_change_multiplier=self.defense_rating_change_multiplier,
            rating_change_multiplier=self.rating_change_multiplier,
            reference_certain_sum_value=self.reference_certain_sum_value,
            certain_weight=self.certain_weight,
            min_rating_change_multiplier_ratio=self.min_rating_change_multiplier_ratio,
            certain_value_denom=self.certain_value_denom,
            max_certain_sum=self.max_certain_sum,
            certain_days_ago_multiplier=self.certain_days_ago_multiplier,
            league_rating_adjustor=league_rating_adjustor,
            min_rating_change_for_league=self.min_rating_change_for_league,
            performance_predictor=performance_predictor,
            past_team_ratings_for_average_rating=self.past_team_ratings_for_average_rating,
            max_days_ago=self.max_days_ago,
            rating_change_momentum_multiplier=self.rating_change_momentum_multiplier,
            rating_change_momentum_games_count=self.rating_change_momentum_games_count,
        )

        return  MatchGenerator(
            league_identifier=LeagueIdentifier(),
            match_rating_calculator=match_rating_calculator,
        )