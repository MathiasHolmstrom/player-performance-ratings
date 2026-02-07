import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from spforge.scorer import (
    Filter,
    MeanBiasScorer,
    Operator,
    OrdinalLossScorer,
    SklearnScorer,
)
from spforge.scorer._score import (
    PWMSE,
    ProbabilisticMeanBias,
    ThresholdEventScorer,
)


class TestScorerNameProperty:
    """Test the auto-generated name property for all scorers."""

    def test_simple_mean_bias_scorer(self):
        scorer = MeanBiasScorer(target="points", pred_column="pred")
        assert scorer.name == "mean_bias_scorer_points"

    def test_simple_pwmse(self):
        scorer = PWMSE(target="goals", pred_column="pred", labels=list(range(10)))
        assert scorer.name == "pwmse_goals"

    def test_simple_ordinal_loss(self):
        scorer = OrdinalLossScorer(target="points", pred_column="pred", classes=list(range(0, 41)))
        assert scorer.name == "ordinal_loss_scorer_points"

    def test_simple_sklearn_scorer(self):
        scorer = SklearnScorer(target="yards", pred_column="pred", scorer_function=mean_absolute_error)
        assert scorer.name == "mean_absolute_error_yards"

    def test_simple_probabilistic_mean_bias(self):
        scorer = ProbabilisticMeanBias(target="points", pred_column="pred")
        assert scorer.name == "probabilistic_mean_bias_points"

    def test_simple_threshold_event_scorer(self):
        scorer = ThresholdEventScorer(
            dist_column="dist",
            threshold_column="threshold",
            outcome_column="outcome",
            labels=list(range(10))
        )
        assert scorer.name == "threshold_event_scorer___event__"

    def test_with_single_granularity(self):
        scorer = MeanBiasScorer(target="points", pred_column="pred", granularity=["team_id"])
        assert scorer.name == "mean_bias_scorer_points_gran:team_id"

    def test_with_multiple_granularity(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            granularity=["game_id", "team_id"]
        )
        assert scorer.name == "mean_bias_scorer_points_gran:game_id+team_id"

    def test_with_long_granularity_abbreviated(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            granularity=["col1", "col2", "col3", "col4", "col5"]
        )
        assert scorer.name == "mean_bias_scorer_points_gran:col1+col2+col3+2more"

    def test_with_naive_comparison_no_granularity(self):
        scorer = SklearnScorer(
            target="goals",
            pred_column="pred",
            scorer_function=mean_absolute_error,
            compare_to_naive=True
        )
        assert scorer.name == "mean_absolute_error_goals_naive"

    def test_with_naive_comparison_with_naive_granularity(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            compare_to_naive=True,
            naive_granularity=["season"]
        )
        assert scorer.name == "mean_bias_scorer_yards_naive:season"

    def test_with_aggregation_level(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            aggregation_level=["game_id", "player_id"]
        )
        assert scorer.name == "mean_bias_scorer_yards_agg:game_id+player_id"

    def test_with_user_filters_only(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            filters=[
                Filter("minutes", 0, Operator.GREATER_THAN),
                Filter("position", "QB", Operator.EQUALS)
            ]
        )
        assert scorer.name == "mean_bias_scorer_yards_filters:2"

    def test_validation_column_not_counted_in_filters(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            validation_column="is_valid",
            filters=[Filter("minutes", 0, Operator.GREATER_THAN)]
        )
        # Should only count the minutes filter, not the auto-added validation filter
        assert scorer.name == "mean_bias_scorer_yards_filters:1"

    def test_validation_column_alone_not_shown(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            validation_column="is_valid"
        )
        # Validation filter auto-added but not counted
        assert scorer.name == "mean_bias_scorer_yards"

    def test_any_filter_on_validation_column_excluded_from_name(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            validation_column="is_valid",
            filters=[
                Filter("is_valid", 0, Operator.EQUALS),
                Filter("minutes", 10, Operator.GREATER_THAN),
            ]
        )
        # Any filter on validation column should be excluded, not just value=1
        assert scorer.name == "mean_bias_scorer_yards_filters:1"

    def test_complex_configuration_all_components(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            granularity=["game_id", "team_id"],
            compare_to_naive=True,
            naive_granularity=["season"],
            aggregation_level=["game_id", "player_id"],
            filters=[Filter("minutes", 0, Operator.GREATER_THAN)]
        )
        assert scorer.name == "mean_bias_scorer_yards_gran:game_id+team_id_naive:season_agg:game_id+player_id_filters:1"

    def test_sklearn_with_different_function(self):
        scorer = SklearnScorer(
            target="points",
            pred_column="pred",
            scorer_function=mean_squared_error
        )
        assert scorer.name == "mean_squared_error_points"

    def test_sklearn_with_lambda_fallback(self):
        scorer = SklearnScorer(
            target="points",
            pred_column="pred",
            scorer_function=lambda y_true, y_pred: 0.0
        )
        assert scorer.name == "custom_metric_points"

    def test_special_characters_sanitized(self):
        scorer = MeanBiasScorer(target="points-per-game", pred_column="pred")
        assert scorer.name == "mean_bias_scorer_points_per_game"

    def test_special_characters_in_target_sanitized(self):
        scorer = MeanBiasScorer(target="pass/run_ratio", pred_column="pred")
        assert scorer.name == "mean_bias_scorer_pass_run_ratio"

    def test_name_override_uses_granularity_suffix_when_set(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            granularity=["team_id"],
            _name_override="custom_name"
        )
        assert scorer.name == "custom_name_gran:team_id"

    def test_name_override_no_granularity(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            _name_override="custom_name"
        )
        assert scorer.name == "custom_name"

    def test_public_name_parameter_uses_granularity_suffix_when_set(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            granularity=["team_id"],
            name="user_metric"
        )
        assert scorer.name == "user_metric_gran:team_id"

    def test_public_name_parameter_ignores_auto_name_components_except_granularity(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            granularity=["team_id"],
            compare_to_naive=True,
            naive_granularity=["season"],
            aggregation_level=["game_id", "player_id"],
            filters=[Filter("minutes", 0, Operator.GREATER_THAN)],
            name="my_custom_metric",
        )
        assert scorer.name == "my_custom_metric_gran:team_id"

    def test_name_and_name_override_conflict_raises(self):
        with pytest.raises(ValueError, match="Received both name and _name_override"):
            MeanBiasScorer(
                target="points",
                pred_column="pred",
                name="one",
                _name_override="two",
            )

    def test_consistency_across_repeated_calls(self):
        scorer = MeanBiasScorer(
            target="yards",
            pred_column="pred",
            granularity=["game_id"],
            compare_to_naive=True
        )
        name1 = scorer.name
        name2 = scorer.name
        name3 = scorer.name
        assert name1 == name2 == name3

    def test_different_scorers_different_names(self):
        scorer1 = MeanBiasScorer(target="points", pred_column="pred")
        scorer2 = PWMSE(target="points", pred_column="pred", labels=list(range(10)))
        assert scorer1.name != scorer2.name

    def test_same_config_same_name(self):
        scorer1 = MeanBiasScorer(
            target="points",
            pred_column="pred",
            granularity=["team_id"]
        )
        scorer2 = MeanBiasScorer(
            target="points",
            pred_column="pred_2",  # Different pred column shouldn't affect name
            granularity=["team_id"]
        )
        assert scorer1.name == scorer2.name

    def test_none_granularity_excluded(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            granularity=None
        )
        assert "gran:" not in scorer.name
        assert scorer.name == "mean_bias_scorer_points"

    def test_empty_filters_excluded(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            filters=[]
        )
        assert "filters:" not in scorer.name
        assert scorer.name == "mean_bias_scorer_points"

    def test_none_aggregation_level_excluded(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            aggregation_level=None
        )
        assert "agg:" not in scorer.name
        assert scorer.name == "mean_bias_scorer_points"

    def test_pwmse_with_all_components(self):
        scorer = PWMSE(
            target="goals",
            pred_column="pred",
            labels=list(range(10)),
            granularity=["team_id"],
            compare_to_naive=True,
            naive_granularity=["season"],
            aggregation_level=["game_id"],
            filters=[Filter("minutes", 20, Operator.GREATER_THAN)]
        )
        assert scorer.name == "pwmse_goals_gran:team_id_naive:season_agg:game_id_filters:1"

    def test_ordinal_loss_with_granularity(self):
        scorer = OrdinalLossScorer(
            target="points",
            pred_column="pred",
            classes=list(range(0, 41)),
            granularity=["game_id"]
        )
        assert scorer.name == "ordinal_loss_scorer_points_gran:game_id"

    def test_threshold_event_scorer_with_components(self):
        scorer = ThresholdEventScorer(
            dist_column="dist",
            threshold_column="threshold",
            outcome_column="outcome",
            labels=list(range(10)),
            granularity=["game_id"],
            compare_to_naive=True
        )
        assert scorer.name == "threshold_event_scorer___event___gran:game_id_naive"

    def test_long_aggregation_abbreviated(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            aggregation_level=["a", "b", "c", "d", "e"]
        )
        assert scorer.name == "mean_bias_scorer_points_agg:a+b+c+2more"

    def test_long_naive_granularity_abbreviated(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            compare_to_naive=True,
            naive_granularity=["a", "b", "c", "d"]
        )
        assert scorer.name == "mean_bias_scorer_points_naive:a+b+c+1more"

    def test_exactly_three_columns_no_abbreviation(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            granularity=["a", "b", "c"]
        )
        assert scorer.name == "mean_bias_scorer_points_gran:a+b+c"

    def test_four_columns_abbreviated(self):
        scorer = MeanBiasScorer(
            target="points",
            pred_column="pred",
            granularity=["a", "b", "c", "d"]
        )
        assert scorer.name == "mean_bias_scorer_points_gran:a+b+c+1more"
