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
        all_games = df[column_names.match_id].unique(maintain_order=True).to_list()
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
    # Phase 4: Assertions - Check prediction consistency
    # ============================================================

    print(f"\nCV validation rows in test games: {len(cv_test_pd)}")
    print(f"Future prediction rows in test games: {len(future_test_pd)}")

    # Need to align rows between CV and future predictions (inner join on game_id + player_id)
    if dataframe_type == "pl":
        merged = cv_test_pd.merge(
            future_test_pd[[column_names.match_id, column_names.player_id, "points_future_pred"]],
            on=[column_names.match_id, column_names.player_id],
            how="inner"
        )
    else:
        merged = cv_test_pd.merge(
            future_test_pd[[column_names.match_id, column_names.player_id, "points_future_pred"]],
            on=[column_names.match_id, column_names.player_id],
            how="inner"
        )

    print(f"Aligned rows for comparison: {len(merged)}")

    # Calculate mean prediction value
    mean_prediction_value = (merged["points_cv_pred"].mean() + merged["points_future_pred"].mean()) / 2

    # Calculate MAE between the two prediction sets
    mae_between_predictions = np.abs(merged["points_cv_pred"] - merged["points_future_pred"]).mean()

    # 3% tolerance based on mean prediction value
    tolerance_3_percent = 0.03 * mean_prediction_value

    print(f"\nMean prediction value: {mean_prediction_value:.3f}")
    print(f"MAE between predictions: {mae_between_predictions:.3f}")
    print(f"3% tolerance: {tolerance_3_percent:.3f}")
    print(f"Ratio: {mae_between_predictions / tolerance_3_percent:.2f}x tolerance")

    assert mae_between_predictions <= tolerance_3_percent, \
        f"MAE between predictions {mae_between_predictions:.3f} exceeds 3% tolerance {tolerance_3_percent:.3f}"
