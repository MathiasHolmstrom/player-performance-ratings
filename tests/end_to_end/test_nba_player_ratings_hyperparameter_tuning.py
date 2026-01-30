"""
End-to-end test for hyperparameter tuning of PlayerRatingGenerator on NBA data.

Tests the full workflow:
1. Load real NBA player-game data
2. Setup rating generator with default parameters (baseline)
3. Run hyperparameter optimization with limited trials
4. Compare optimized vs baseline performance
5. Validate integration with FeatureGeneratorPipeline, AutoPipeline, CrossValidator, Scorer
"""

import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier

from examples import get_sub_sample_nba_data
from spforge import (
    AutoPipeline,
    ColumnNames,
    RatingHyperparameterTuner,
)
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.ratings import PlayerRatingGenerator
from spforge.scorer import OrdinalLossScorer


@pytest.fixture
def nba_column_names():
    """Column name configuration for NBA data."""
    return ColumnNames(
        player_id="player_id",
        team_id="team_id",
        match_id="game_id",
        start_date="start_date",
        update_match_id="game_id",
        participation_weight="minutes_ratio",
    )


@pytest.fixture
def nba_df():
    """Load and preprocess NBA data."""
    df = get_sub_sample_nba_data()

    df = df.assign(
        minutes_ratio=lambda x: x["minutes"] / x.groupby("game_id")["minutes"].transform("sum"),
        points_clipped=lambda x: x["points"].clip(0, 40),
    )

    df = df.sort_values(["start_date", "game_id", "team_id", "player_id"])

    return df


@pytest.mark.parametrize("dataframe_type", ["pd", "pl"])
def test_nba_player_ratings_hyperparameter_tuning__workflow_completes(
    nba_df, nba_column_names, dataframe_type
):
    """
    Test that hyperparameter tuning workflow completes successfully.

    Validates the full integration: RatingGenerator → FeatureGeneratorPipeline →
    AutoPipeline → CrossValidator → Scorer → Tuner
    """
    if dataframe_type == "pl":
        df = pl.from_pandas(nba_df)
    else:
        df = nba_df.copy()

    rating_gen = PlayerRatingGenerator(
        performance_column="points_clipped",
        column_names=nba_column_names,
        auto_scale_performance=True,
    )

    tuner = _create_tuner(
        rating_generator=rating_gen,
        column_names=nba_column_names,
        n_trials=10,
    )

    result = tuner.optimize(df)

    assert result.best_params is not None
    assert isinstance(result.best_params, dict)
    assert len(result.best_params) > 0
    assert isinstance(result.best_value, float)
    assert result.best_trial is not None
    assert result.study is not None

    expected_params = {
        "rating_change_multiplier_offense",
        "rating_change_multiplier_defense",
        "confidence_weight",
        "confidence_value_denom",
        "confidence_max_sum",
        "use_off_def_split",
        "start_league_quantile",
        "start_min_count_for_percentiles",
    }
    assert set(result.best_params.keys()) == expected_params

    optimized_rating_gen = PlayerRatingGenerator(
        performance_column="points_clipped",
        column_names=nba_column_names,
        auto_scale_performance=True,
        **result.best_params,
    )

    df_with_features = optimized_rating_gen.fit_transform(df)
    assert df_with_features is not None


def test_nba_player_ratings_hyperparameter_tuning__study_records_trials(nba_df, nba_column_names):
    """
    Test that optimization study records all trials correctly.
    """
    df = nba_df.copy()

    rating_gen = PlayerRatingGenerator(
        performance_column="points_clipped",
        column_names=nba_column_names,
        auto_scale_performance=True,
    )

    n_trials = 10
    tuner = _create_tuner(
        rating_generator=rating_gen,
        column_names=nba_column_names,
        n_trials=n_trials,
    )

    result = tuner.optimize(df)

    assert len(result.study.trials) == n_trials
    assert all(trial.value is not None for trial in result.study.trials)
    assert result.best_value in [trial.value for trial in result.study.trials]


def test_nba_player_ratings_hyperparameter_tuning__custom_search_space(nba_df, nba_column_names):
    """
    Test hyperparameter tuning with custom search space.
    """
    from spforge import ParamSpec

    df = nba_df.copy()

    rating_gen = PlayerRatingGenerator(
        performance_column="points_clipped",
        column_names=nba_column_names,
        auto_scale_performance=True,
    )

    custom_space = {
        "rating_change_multiplier_offense": ParamSpec(
            param_type="float",
            low=40.0,
            high=60.0,
            log=True,
        ),
    }

    tuner = _create_tuner(
        rating_generator=rating_gen,
        column_names=nba_column_names,
        n_trials=5,
        custom_search_space=custom_space,
    )

    result = tuner.optimize(df)

    assert 40.0 <= result.best_params["rating_change_multiplier_offense"] <= 60.0
    assert "rating_change_multiplier_defense" in result.best_params


def _create_tuner(
    rating_generator: PlayerRatingGenerator,
    column_names: ColumnNames,
    n_trials: int,
    custom_search_space: dict | None = None,
) -> RatingHyperparameterTuner:
    """
    Create hyperparameter tuner with standard configuration.

    This encapsulates the setup of AutoPipeline, CrossValidator, and Scorer
    to avoid repetition across tests.
    """
    pipeline = AutoPipeline(
        estimator=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        estimator_features=rating_generator.features_out + ["minutes_ratio"],
    )

    cv = MatchKFoldCrossValidator(
        match_id_column_name="game_id",
        date_column_name="start_date",
        target_column="points_clipped",
        estimator=pipeline,
        prediction_column_name="points_pred",
        n_splits=2,
        features=pipeline.required_features,
    )

    scorer = OrdinalLossScorer(
        pred_column="points_pred",
        target="points_clipped",
        classes=list(range(0, 41)),
        validation_column="is_validation",
    )

    return RatingHyperparameterTuner(
        rating_generator=rating_generator,
        cross_validator=cv,
        scorer=scorer,
        direction="minimize",
        param_search_space=custom_search_space,
        n_trials=n_trials,
        show_progress_bar=False,
    )
