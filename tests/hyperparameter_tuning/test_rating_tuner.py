import copy
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from spforge import AutoPipeline, ColumnNames, ParamSpec, RatingHyperparameterTuner
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.ratings import PlayerRatingGenerator, TeamRatingGenerator
from spforge.scorer import OrdinalLossScorer


@pytest.fixture
def player_column_names():
    return ColumnNames(
        player_id="pid",
        team_id="tid",
        match_id="mid",
        start_date="date",
        update_match_id="mid",
        participation_weight="pw",
    )


@pytest.fixture
def sample_player_df_pd(player_column_names):
    """Create sample player data for testing (pandas)."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for i, date in enumerate(dates):
        mid = f"M{i}"
        for player_idx in range(4):
            pid = f"P{player_idx % 2}"
            tid = "T1" if player_idx < 2 else "T2"
            perf = 0.6 if player_idx < 2 else 0.4
            rows.append(
                {
                    "pid": pid,
                    "tid": tid,
                    "mid": mid,
                    "date": date,
                    "perf": perf + np.random.normal(0, 0.1),
                    "pw": 1.0,
                    "minutes": 30.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_player_df_pl(sample_player_df_pd):
    """Create sample player data for testing (polars)."""
    return pl.from_pandas(sample_player_df_pd)


@pytest.fixture
def player_rating_generator(player_column_names):
    return PlayerRatingGenerator(
        performance_column="perf",
        column_names=player_column_names,
        auto_scale_performance=True,
    )


@pytest.fixture
def cross_validator(player_rating_generator):
    pipeline = AutoPipeline(
        estimator=DummyClassifier(strategy="prior"),
        estimator_features=player_rating_generator.features_out + ["minutes"],
    )

    return MatchKFoldCrossValidator(
        match_id_column_name="mid",
        date_column_name="date",
        target_column="perf",
        estimator=pipeline,
        prediction_column_name="perf_pred",
        n_splits=2,
        features=pipeline.required_features,
    )


@pytest.fixture
def scorer():
    return OrdinalLossScorer(
        pred_column="perf_pred",
        target="perf",
        classes=[0, 1],
        validation_column="is_validation",
    )


def test_basic_optimization__completes_successfully(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that basic optimization completes without errors."""
    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert result.best_params is not None
    assert isinstance(result.best_params, dict)
    assert isinstance(result.best_value, float)
    assert result.best_trial is not None
    assert result.study is not None


@pytest.mark.parametrize(
    "df_fixture",
    ["sample_player_df_pd", "sample_player_df_pl"],
)
def test_optimization__works_with_pandas_and_polars(
    player_rating_generator, cross_validator, scorer, df_fixture, request
):
    """Test that optimization works with both pandas and polars DataFrames."""
    df = request.getfixturevalue(df_fixture)

    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        n_trials=2,
        show_progress_bar=False,
    )

    result = tuner.optimize(df)

    assert result.best_params is not None
    assert isinstance(result.best_value, float)


def test_custom_search_space__merges_with_defaults(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that custom search space merges with defaults correctly."""
    custom_space = {
        "rating_change_multiplier_offense": ParamSpec(
            param_type="float",
            low=30.0,
            high=50.0,
            log=True,
        ),
    }

    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        param_search_space=custom_space,
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert "rating_change_multiplier_offense" in result.best_params
    assert 30.0 <= result.best_params["rating_change_multiplier_offense"] <= 50.0


def test_direction_minimize__returns_lower_score(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that minimize direction attempts to find lower scores."""
    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        n_trials=5,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert result.study.direction.name == "MINIMIZE"


def test_direction_maximize__returns_higher_score(
    player_rating_generator, cross_validator, sample_player_df_pd
):
    """Test that maximize direction attempts to find higher scores."""
    from spforge.scorer import MeanBiasScorer

    scorer_max = MeanBiasScorer(
        pred_column="perf_pred",
        target="perf",
        validation_column="is_validation",
    )

    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer_max,
        direction="maximize",
        n_trials=5,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert result.study.direction.name == "MAXIMIZE"


def test_deep_copy_isolation__trials_independent(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that each trial uses independent deep copy of rating generator."""
    original_gen = copy.deepcopy(player_rating_generator)

    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert len(player_rating_generator._player_off_ratings) == len(
        original_gen._player_off_ratings
    )


def test_param_spec__suggests_float_correctly():
    """Test that ParamSpec suggests float parameters correctly."""
    import optuna

    spec = ParamSpec(param_type="float", low=1.0, high=10.0, log=False)

    study = optuna.create_study()
    trial = study.ask()
    value = spec.suggest(trial, "test_param")

    assert isinstance(value, float)
    assert 1.0 <= value <= 10.0


def test_param_spec__suggests_int_correctly():
    """Test that ParamSpec suggests int parameters correctly."""
    import optuna

    spec = ParamSpec(param_type="int", low=1, high=10, step=2)

    study = optuna.create_study()
    trial = study.ask()
    value = spec.suggest(trial, "test_param")

    assert isinstance(value, int)
    assert 1 <= value <= 10


def test_param_spec__suggests_categorical_correctly():
    """Test that ParamSpec suggests categorical parameters correctly."""
    import optuna

    choices = ["option_a", "option_b", "option_c"]
    spec = ParamSpec(param_type="categorical", choices=choices)

    study = optuna.create_study()
    trial = study.ask()
    value = spec.suggest(trial, "test_param")

    assert value in choices


def test_param_spec__suggests_bool_correctly():
    """Test that ParamSpec suggests bool parameters correctly."""
    import optuna

    spec = ParamSpec(param_type="bool")

    study = optuna.create_study()
    trial = study.ask()
    value = spec.suggest(trial, "test_param")

    assert isinstance(value, bool)


def test_scorer_dict__aggregates_to_mean(player_rating_generator, sample_player_df_pd):
    """Test that dict scores from granular scorers are aggregated to mean."""
    from spforge.scorer import OrdinalLossScorer

    scorer_granular = OrdinalLossScorer(
        pred_column="perf_pred",
        target="perf",
        classes=[0, 1],
        validation_column="is_validation",
        granularity=["tid"],
    )

    pipeline = AutoPipeline(
        estimator=DummyClassifier(strategy="prior"),
        estimator_features=player_rating_generator.features_out + ["minutes"],
    )

    cv = MatchKFoldCrossValidator(
        match_id_column_name="mid",
        date_column_name="date",
        target_column="perf",
        estimator=pipeline,
        prediction_column_name="perf_pred",
        n_splits=2,
        features=pipeline.required_features,
    )

    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cv,
        scorer=scorer_granular,
        direction="minimize",
        n_trials=2,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert isinstance(result.best_value, float)


def test_invalid_direction__raises_error(player_rating_generator, cross_validator, scorer):
    """Test that invalid direction raises ValueError."""
    with pytest.raises(ValueError, match="direction must be"):
        RatingHyperparameterTuner(
            rating_generator=player_rating_generator,
            cross_validator=cross_validator,
            scorer=scorer,
            direction="invalid",
            n_trials=3,
        )


def test_storage_without_study_name__raises_error(
    player_rating_generator, cross_validator, scorer
):
    """Test that using storage without study_name raises ValueError."""
    with pytest.raises(ValueError, match="study_name is required"):
        RatingHyperparameterTuner(
            rating_generator=player_rating_generator,
            cross_validator=cross_validator,
            scorer=scorer,
            direction="minimize",
            n_trials=3,
            storage="sqlite:///test.db",
            study_name=None,
        )


def test_team_rating_generator__works_correctly():
    """Test that TeamRatingGenerator works with tuner."""
    from spforge import GameColumnNames

    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "mid": f"M{i}",
                "date": date,
                "home_team": "T1" if i % 2 == 0 else "T2",
                "away_team": "T2" if i % 2 == 0 else "T1",
                "home_perf": 0.6 + np.random.normal(0, 0.1),
                "away_perf": 0.4 + np.random.normal(0, 0.1),
            }
        )
    df = pd.DataFrame(rows)

    column_names = GameColumnNames(
        match_id="mid",
        start_date="date",
        team1_name="home_team",
        team2_name="away_team",
        performance_column_pairs={"perf": ("home_perf", "away_perf")},
    )

    rating_gen = TeamRatingGenerator(
        performance_column="perf",
        column_names=column_names,
        auto_scale_performance=True,
    )

    pipeline = AutoPipeline(
        estimator=DummyClassifier(strategy="prior"),
        estimator_features=rating_gen.features_out,
    )

    cv = MatchKFoldCrossValidator(
        match_id_column_name="mid",
        date_column_name="date",
        target_column="perf",
        estimator=pipeline,
        prediction_column_name="perf_pred",
        n_splits=2,
        features=pipeline.required_features,
    )

    scorer = OrdinalLossScorer(
        pred_column="perf_pred",
        target="perf",
        classes=[0, 1],
        validation_column="is_validation",
    )

    tuner = RatingHyperparameterTuner(
        rating_generator=rating_gen,
        cross_validator=cv,
        scorer=scorer,
        direction="minimize",
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(df)

    assert result.best_params is not None
    assert isinstance(result.best_value, float)


def test_param_spec__float_requires_bounds():
    """Test that float ParamSpec requires low and high bounds."""
    import optuna

    spec = ParamSpec(param_type="float", low=None, high=None)

    study = optuna.create_study()
    trial = study.ask()

    with pytest.raises(ValueError, match="requires low and high bounds"):
        spec.suggest(trial, "test_param")


def test_param_spec__categorical_requires_choices():
    """Test that categorical ParamSpec requires choices."""
    import optuna

    spec = ParamSpec(param_type="categorical", choices=None)

    study = optuna.create_study()
    trial = study.ask()

    with pytest.raises(ValueError, match="requires choices"):
        spec.suggest(trial, "test_param")


def test_param_ranges__overrides_bounds(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that param_ranges overrides low/high bounds while preserving param_type."""
    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        param_ranges={
            "confidence_weight": (0.2, 0.3),
        },
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert "confidence_weight" in result.best_params
    assert 0.2 <= result.best_params["confidence_weight"] <= 0.3


def test_exclude_params__removes_from_search(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that exclude_params removes parameters from search space."""
    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        exclude_params=["use_off_def_split", "confidence_weight"],
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert "use_off_def_split" not in result.best_params
    assert "confidence_weight" not in result.best_params
    assert "rating_change_multiplier_offense" in result.best_params


def test_fixed_params__applies_values_without_tuning(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that fixed_params sets values without including in search space."""
    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        fixed_params={"use_off_def_split": False},
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert "use_off_def_split" not in result.best_params


def test_param_ranges__unknown_param_raises_error(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that param_ranges with unknown param raises ValueError."""
    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        param_ranges={"nonexistent_param": (0.0, 1.0)},
        n_trials=3,
        show_progress_bar=False,
    )

    with pytest.raises(ValueError, match="unknown parameter"):
        tuner.optimize(sample_player_df_pd)


def test_param_ranges__non_numeric_param_raises_error(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test that param_ranges on non-float/int param raises ValueError."""
    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        param_ranges={"use_off_def_split": (0, 1)},
        n_trials=3,
        show_progress_bar=False,
    )

    with pytest.raises(ValueError, match="can only override float/int"):
        tuner.optimize(sample_player_df_pd)


def test_combined_api__param_ranges_exclude_fixed(
    player_rating_generator, cross_validator, scorer, sample_player_df_pd
):
    """Test using param_ranges, exclude_params, and fixed_params together."""
    tuner = RatingHyperparameterTuner(
        rating_generator=player_rating_generator,
        cross_validator=cross_validator,
        scorer=scorer,
        direction="minimize",
        param_ranges={
            "confidence_weight": (0.2, 1.0),
            "rating_change_multiplier_offense": (10.0, 150.0),
        },
        exclude_params=["start_league_quantile"],
        fixed_params={"use_off_def_split": False},
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(sample_player_df_pd)

    assert 0.2 <= result.best_params["confidence_weight"] <= 1.0
    assert 10.0 <= result.best_params["rating_change_multiplier_offense"] <= 150.0
    assert "start_league_quantile" not in result.best_params
    assert "use_off_def_split" not in result.best_params


def test_default_search_space__excludes_performance_predictor_and_team_start(
    player_rating_generator,
):
    """Test that performance_predictor and team start params are not in default search space."""
    from spforge.hyperparameter_tuning._default_search_spaces import (
        get_default_search_space,
    )

    defaults = get_default_search_space(player_rating_generator)

    assert "performance_predictor" not in defaults
    assert "start_team_rating_subtract" not in defaults
    assert "start_team_weight" not in defaults
    assert "start_min_match_count_team_rating" not in defaults


def test_full_player_rating_search_space__includes_all_params():
    """Test that full search space includes performance_predictor and team start params."""
    from spforge.hyperparameter_tuning._default_search_spaces import (
        get_full_player_rating_search_space,
    )

    full = get_full_player_rating_search_space()

    assert "performance_predictor" in full
    assert "start_team_rating_subtract" in full
    assert "start_team_weight" in full
    assert "start_min_match_count_team_rating" in full
    assert "rating_change_multiplier_offense" in full
    assert "confidence_weight" in full
