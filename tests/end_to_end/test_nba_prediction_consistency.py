import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from examples import get_sub_sample_nba_data
from spforge import FeatureGeneratorPipeline
from spforge.autopipeline import AutoPipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.data_structures import ColumnNames
from spforge.feature_generator import LagTransformer, RollingWindowTransformer
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures


@pytest.mark.parametrize("dataframe_type", ["pl", "pd"])
def test_nba_prediction_consistency__cv_vs_future_transform(dataframe_type):
    """
    Test that validates prediction consistency between cross-validation
    and future prediction workflows using future_transform().

    Ensures that:
    1. future_transform() produces valid predictions without updating state
    2. CV and future approaches produce predictions with similar statistical properties
    3. Both approaches capture similar underlying patterns in the data
    """

    # ============================================================
    # Phase 1: Setup
    # ============================================================

    # Load NBA data
    df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)

    # Setup column names
    column_names = ColumnNames(
        team_id="team_id",
        match_id="game_id",
        start_date="start_date",
        player_id="player_id",
        participation_weight="minutes_ratio",
    )

    # Sort by temporal order
    df = df.sort(
        [
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]
    )

    # Create derived columns
    df = df.with_columns(
        [
            (pl.col("minutes") / pl.lit(48.25)).alias("minutes_ratio"),
            (pl.col("points") / pl.col("minutes")).alias("points_per_minute"),
        ]
    )

    # Handle division by zero for points_per_minute
    df = df.with_columns(
        pl.when(pl.col("minutes") == 0)
        .then(pl.lit(0))
        .otherwise(pl.col("points_per_minute"))
        .alias("points_per_minute")
    )

    # Clip points to [0, 40]
    df = df.with_columns(pl.col("points").clip(0, 40).alias("points"))

    # Convert to pandas if needed
    if dataframe_type == "pd":
        df = df.to_pandas()

    # Split by games (not rows) - use last 10 games for testing
    if dataframe_type == "pl":
        all_games = df[column_names.match_id].unique().to_list()
    else:
        all_games = df[column_names.match_id].unique().tolist()

    train_games = all_games[:-10]  # All games except last 10
    test_games = all_games[-10:]    # Last 10 games

    if dataframe_type == "pl":
        train_df = df.filter(pl.col(column_names.match_id).is_in(train_games))
        test_df = df.filter(pl.col(column_names.match_id).is_in(test_games))
    else:
        train_df = df[df[column_names.match_id].isin(train_games)].copy()
        test_df = df[df[column_names.match_id].isin(test_games)].copy()

    # Ensure sufficient test data
    print(f"\nTrain games: {len(train_games)}, Test games: {len(test_games)}")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    assert len(test_df) >= 50, f"Insufficient test data: {len(test_df)} rows"

    # ============================================================
    # Phase 2: Cross-Validation Path (on ENTIRE dataset)
    # ============================================================

    # Create feature generators for CV path
    player_rating_cv = PlayerRatingGenerator(
        performance_column="points_per_minute",
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED],
        column_names=column_names,
    )

    lag_gen_cv = LagTransformer(
        features=["points"],
        lag_length=3,
        granularity=["player_id"],
    )

    rolling_gen_cv = RollingWindowTransformer(
        features=["points"],
        window=5,
        granularity=["player_id"],
    )

    features_pipeline_cv = FeatureGeneratorPipeline(
        column_names=column_names,
        feature_generators=[player_rating_cv, lag_gen_cv, rolling_gen_cv],
    )

    # Fit and transform ENTIRE dataset for cross-validation
    df_cv_transformed = features_pipeline_cv.fit_transform(df)

    # Create AutoPipeline with LinearRegression (auto-detects and enables imputation)
    pipeline_cv = AutoPipeline(
        estimator=LinearRegression(),
        estimator_features=features_pipeline_cv.features_out,
    )

    # Create cross-validator
    cross_validator = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=pipeline_cv,
        prediction_column_name="points_cv_pred",
        target_column="points",
        n_splits=3,
        features=pipeline_cv.required_features,
    )

    # Generate validation predictions for ENTIRE dataset
    df_with_cv_preds = cross_validator.generate_validation_df(
        df=df_cv_transformed,
        add_training_predictions=False
    )

    # Extract validation predictions for last test_games only (to match future predictions)
    if dataframe_type == "pl":
        cv_test_games = df_with_cv_preds.filter(pl.col(column_names.match_id).is_in(test_games))
        # Filter to validation rows only
        cv_test_games_validation = cv_test_games.filter(pl.col("is_validation") == 1)
        cv_test_pd = cv_test_games_validation.to_pandas()
    else:
        cv_test_games = df_with_cv_preds[df_with_cv_preds[column_names.match_id].isin(test_games)]
        # Filter to validation rows only
        cv_test_games_validation = cv_test_games[cv_test_games["is_validation"] == 1].copy()
        cv_test_pd = cv_test_games_validation

    # ============================================================
    # Phase 3: Future Prediction Path (train on all except last 100)
    # ============================================================

    # Create SEPARATE feature generators for future path (avoid state pollution)
    player_rating_future = PlayerRatingGenerator(
        performance_column="points_per_minute",
        auto_scale_performance=True,
        features_out=[RatingKnownFeatures.PLAYER_RATING_DIFFERENCE_PROJECTED],
        column_names=column_names,
    )

    lag_gen_future = LagTransformer(
        features=["points"],
        lag_length=3,
        granularity=["player_id"],
    )

    rolling_gen_future = RollingWindowTransformer(
        features=["points"],
        window=5,
        granularity=["player_id"],
    )

    features_pipeline_future = FeatureGeneratorPipeline(
        column_names=column_names,
        feature_generators=[player_rating_future, lag_gen_future, rolling_gen_future],
    )

    # Train on all data except last 100 rows
    train_future_transformed = features_pipeline_future.fit_transform(train_df)

    # Create and train AutoPipeline
    pipeline_future = AutoPipeline(
        estimator=LinearRegression(),
        estimator_features=features_pipeline_future.features_out,
    )

    pipeline_future.fit(X=train_future_transformed, y=train_future_transformed["points"])

    # Use future_transform on test data (last 100 rows - key method being tested!)
    test_df_transformed = features_pipeline_future.future_transform(test_df)

    # Make predictions on last 100 rows
    if dataframe_type == "pl":
        predictions = pipeline_future.predict(test_df_transformed)
        test_df_transformed = test_df_transformed.with_columns(
            pl.lit(predictions).alias("points_future_pred")
        )
        future_test_pd = test_df_transformed.to_pandas()
    else:
        predictions = pipeline_future.predict(test_df_transformed)
        test_df_transformed["points_future_pred"] = predictions
        future_test_pd = test_df_transformed

    # ============================================================
    # Phase 4: Assertions - Comparison Metrics on Last 100 Rows
    # ============================================================

    print(f"\nCV validation rows in test games: {len(cv_test_pd)}")
    print(f"Future prediction rows in test games: {len(future_test_pd)}")

    # 1. Mean Prediction Difference (within 15% tolerance)
    # Note: Different training sets and potentially different sample sizes
    cv_mean = cv_test_pd["points_cv_pred"].mean()
    future_mean = future_test_pd["points_future_pred"].mean()
    mean_diff = abs(cv_mean - future_mean)
    tolerance_15_percent = 0.15 * cv_mean
    print(f"\nMean predictions: CV={cv_mean:.3f}, Future={future_mean:.3f}")
    print(f"Absolute difference: {mean_diff:.3f}, 15% tolerance: {tolerance_15_percent:.3f}")
    assert mean_diff <= tolerance_15_percent, f"Mean difference {mean_diff:.3f} exceeds 15% tolerance {tolerance_15_percent:.3f}"

    # 2. Standard Deviation Similarity
    cv_std = cv_test_pd["points_cv_pred"].std()
    future_std = future_test_pd["points_future_pred"].std()
    std_diff = abs(cv_std - future_std)
    print(f"Std: CV={cv_std:.3f}, Future={future_std:.3f}, Diff={std_diff:.3f}")
    assert std_diff < 2.0, f"Std difference too large: {std_diff:.3f}"

    # 3. Correlation with Actual Values
    cv_corr = cv_test_pd[["points_cv_pred", "points"]].corr().iloc[0, 1]
    future_corr = future_test_pd[["points_future_pred", "points"]].corr().iloc[0, 1]
    corr_diff = abs(cv_corr - future_corr)
    print(f"Correlation: CV={cv_corr:.3f}, Future={future_corr:.3f}, Diff={corr_diff:.3f}")
    assert cv_corr > 0.3, f"CV correlation too low: {cv_corr:.3f}"
    assert future_corr > 0.3, f"Future correlation too low: {future_corr:.3f}"

    # 4. MAE Comparison
    cv_mae = mean_absolute_error(cv_test_pd["points"], cv_test_pd["points_cv_pred"])
    future_mae = mean_absolute_error(future_test_pd["points"], future_test_pd["points_future_pred"])
    mae_diff = abs(cv_mae - future_mae)
    print(f"MAE: CV={cv_mae:.3f}, Future={future_mae:.3f}, Diff={mae_diff:.3f}")
    assert mae_diff < 2.0, f"MAE difference too large: {mae_diff:.3f}"

    # 5. Directional Consistency (High vs Low Ratings)
    player_rating_col = player_rating_cv.PLAYER_DIFF_PROJ_COL

    # CV path
    cv_high = cv_test_pd[cv_test_pd[player_rating_col] > 0]["points_cv_pred"].mean()
    cv_low = cv_test_pd[cv_test_pd[player_rating_col] < 0]["points_cv_pred"].mean()
    print(f"CV directional: High={cv_high:.3f}, Low={cv_low:.3f}")
    assert cv_high > cv_low, f"CV: High rating pred should exceed low rating pred"

    # Future path
    future_high = future_test_pd[future_test_pd[player_rating_col] > 0]["points_future_pred"].mean()
    future_low = future_test_pd[future_test_pd[player_rating_col] < 0]["points_future_pred"].mean()
    print(f"Future directional: High={future_high:.3f}, Low={future_low:.3f}")
    assert future_high > future_low, f"Future: High rating pred should exceed low rating pred"

    # 6. No NaN or Infinite Values
    assert not cv_test_pd["points_cv_pred"].isna().any(), "CV predictions contain NaN"
    assert not future_test_pd["points_future_pred"].isna().any(), "Future predictions contain NaN"
    assert not np.isinf(cv_test_pd["points_cv_pred"]).any(), "CV predictions contain inf"
    assert not np.isinf(future_test_pd["points_future_pred"]).any(), "Future predictions contain inf"

    print("\nAll assertions passed! CV and future predictions are consistent.")
