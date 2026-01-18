import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

from examples import get_sub_sample_lol_data
from spforge import AutoPipeline, ColumnNames, FeatureGeneratorPipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.distributions import NegativeBinomialEstimator
from spforge.feature_generator import LagTransformer, RollingWindowTransformer
from spforge.performance_transformers._performance_manager import ColumnWeight
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures


def test_lol_feature_engineering_and_distribution_end_to_end():
    column_names = ColumnNames(
        team_id="teamname",
        match_id="gameid",
        start_date="date",
        player_id="playername",
        league="league",
        position="position",
    )

    df = get_sub_sample_lol_data(as_pandas=True)
    df = (
        df.loc[lambda x: x.position != "team"]
        .assign(team_count=df.groupby("gameid")["teamname"].transform("nunique"))
        .loc[lambda x: x.team_count == 2]
        .assign(player_count=df.groupby(["gameid", "teamname"])["playername"].transform("nunique"))
        .loc[lambda x: x.player_count == 5]
    )
    df = df.assign(team_count=df.groupby("gameid")["teamname"].transform("nunique")).loc[
        lambda x: x.team_count == 2
    ]
    df = df.drop_duplicates(subset=["gameid", "playername", "teamname"])

    most_recent_10_games = df[column_names.match_id].unique()[-10:]
    historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)].copy()
    future_df = (
        df[df[column_names.match_id].isin(most_recent_10_games)].drop(columns=["result"]).copy()
    )

    rating_generator_player_kills = PlayerRatingGenerator(
        features_out=[RatingKnownFeatures.PLAYER_RATING],
        performance_column="kills",
        auto_scale_performance=True,
        performance_weights=[ColumnWeight(name="kills", weight=1)],
        column_names=column_names,
    )
    rating_generator_result = PlayerRatingGenerator(
        features_out=[RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED],
        performance_column="result",
        non_predictor_features_out=[RatingKnownFeatures.PLAYER_RATING],
        column_names=column_names,
    )

    lag_generators = [
        LagTransformer(
            features=["kills", "deaths", "result"],
            lag_length=3,
            granularity=["playername"],
        ),
        RollingWindowTransformer(
            features=["kills", "deaths", "result"],
            window=20,
            min_periods=1,
            granularity=["playername"],
        ),
    ]

    features_generator = FeatureGeneratorPipeline(
        feature_generators=[
            rating_generator_player_kills,
            rating_generator_result,
            *lag_generators,
        ],
        column_names=column_names,
    )

    historical_df = features_generator.fit_transform(historical_df)

    game_winner_model = AutoPipeline(
        estimator=LogisticRegression(max_iter=1000),
        impute_missing_values=True,
        scale_features=False,
        estimator_features=rating_generator_result.features_out,
    )

    cv_game_winner = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=game_winner_model,
        prediction_column_name="result_oof",
        n_splits=3,
        target_column="result",
    )

    historical_df = cv_game_winner.generate_validation_df(historical_df)
    assert "result_oof" in historical_df.columns

    def _prob1(x):
        if isinstance(x, (list, np.ndarray)):
            return float(x[1]) if len(x) > 1 else float(x[0])
        return float(x)

    historical_df["result_prob1"] = historical_df["result_oof"].apply(_prob1)

    player_kills_features = ["result_prob1", *features_generator.features_out]

    player_kills_model = AutoPipeline(
        estimator=LGBMRegressor(verbose=-100),
        impute_missing_values=True,
        scale_features=False,
        estimator_features=player_kills_features,
    )

    cv_player_kills = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=player_kills_model,
        prediction_column_name="kills_oof",
        n_splits=3,
        target_column="kills",
    )
    cv_player_kills.features = player_kills_features
    cv_player_kills.target = "kills"

    historical_df = cv_player_kills.generate_validation_df(historical_df)
    assert "kills_oof" in historical_df.columns

    future_df = features_generator.future_transform(future_df)

    game_winner_model.fit(X=historical_df, y=historical_df["result"])

    proba = game_winner_model.predict_proba(future_df)
    future_df["result_prob1"] = proba[:, 1]

    player_kills_model.fit(X=historical_df[player_kills_features], y=historical_df["kills"])
    historical_kills_pred = player_kills_model.predict(X=historical_df[player_kills_features])
    historical_df["kills_pred"] = historical_kills_pred

    future_kills_pred = player_kills_model.predict(X=future_df[player_kills_features])
    future_df["kills_pred"] = future_kills_pred

    probability_predictor = NegativeBinomialEstimator(
        point_estimate_pred_column="kills_pred",
        max_value=15,
    )
    probability_predictor.fit(X=historical_df[["kills_pred"]], y=historical_df["kills"])

    probabilities = probability_predictor.predict_proba(future_df[["kills_pred"]])

    n_samples = len(future_df)
    n_classes = 16  # max_value - min_value + 1 = 15 - 0 + 1 = 16
    assert probabilities.shape == (
        n_samples,
        n_classes,
    ), f"Probabilities shape should be ({n_samples}, {n_classes}), got {probabilities.shape}"

    neg_mask = probabilities < 0

    if neg_mask.any():
        idx = np.argwhere(neg_mask)

        # Show first few offenders
        bad = idx[:10]
        raise AssertionError(
            f"Found negative probabilities at indices (row, class): {bad.tolist()}\n"
            f"Values: {[probabilities[i, j] for i, j in bad]}"
        )
    prob_sums = probabilities.sum(axis=1)

    if not np.allclose(prob_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(prob_sums, 1.0, atol=1e-6))[0]

        rows = bad[:10].tolist()

        details = []
        for i in rows:
            row = probabilities[i]
            details.append(
                {
                    "row": int(i),
                    "sum": float(prob_sums[i]),
                    "has_nan": bool(np.isnan(row).any()),
                    "has_inf": bool(np.isinf(row).any()),
                    "min": float(np.nanmin(row)),
                    "max": float(np.nanmax(row)),
                    "values": row.tolist(),
                }
            )

        raise AssertionError(
            "Probabilities should sum to 1.0 for each sample.\n" f"Bad rows (first 10): {details}"
        )

    # Assert 3: Calculate expected value using dot product: E[X] = sum(probabilities * classes)
    classes = np.arange(0, n_classes)  # [0, 1, 2, ..., 15]
    expected_kills = (
        probabilities @ classes
    )  # Dot product: each row of probabilities @ classes vector
    future_df["expected_kills"] = expected_kills

    # Verify dot product calculation is correct
    assert (
        len(expected_kills) == n_samples
    ), "Expected kills should have same length as number of samples"
    assert expected_kills.dtype in [np.float64, np.float32], "Expected kills should be float"

    # Assert 4: Expected kills should be within valid range [0, max_value]
    assert (
        expected_kills >= 0
    ).all(), f"Expected kills must be >= 0, got min={expected_kills.min():.2f}"
    assert (
        expected_kills <= 15
    ).all(), f"Expected kills must be <= 15, got max={expected_kills.max():.2f}"

    # Assert 5: Expected kills should correlate positively with point estimates
    correlation = future_df["kills_pred"].corr(future_df["expected_kills"])
    assert not np.isnan(
        correlation
    ), "Correlation between kills_pred and expected_kills should not be NaN"
    assert (
        correlation > 0.3
    ), f"Expected kills should be positively correlated with kills_pred, got correlation={correlation:.3f}"

    # Assert 6: Expected kills should be reasonably close to point estimates
    # (allowing for distribution shape differences)
    diff = np.abs(future_df["expected_kills"] - future_df["kills_pred"])
    assert diff.mean() < 5.0, (
        f"Expected kills should be reasonably close to point estimates, "
        f"mean absolute difference={diff.mean():.2f} is too large"
    )

    # Assert 7: Use historical performance to validate model behavior
    # Players with higher historical average kills should have higher expected kills
    historical_avg_kills = (
        historical_df.groupby(column_names.player_id)["kills"].mean().sort_values(ascending=False)
    )

    # Get top and bottom performers (at least 3 each for statistical significance)
    n_top = min(5, len(historical_avg_kills))
    n_bottom = min(5, len(historical_avg_kills))
    top_players = historical_avg_kills.head(n_top).index.tolist()
    bottom_players = historical_avg_kills.tail(n_bottom).index.tolist()

    # Filter to players that exist in both historical and future data
    future_df_with_history = future_df[
        future_df[column_names.player_id].isin(historical_avg_kills.index)
    ]

    if len(future_df_with_history) > 0:
        top_player_mask = future_df_with_history[column_names.player_id].isin(top_players)
        bottom_player_mask = future_df_with_history[column_names.player_id].isin(bottom_players)

        if top_player_mask.sum() > 0 and bottom_player_mask.sum() > 0:
            top_avg_expected = future_df_with_history.loc[top_player_mask, "expected_kills"].mean()
            bottom_avg_expected = future_df_with_history.loc[
                bottom_player_mask, "expected_kills"
            ].mean()

            # Assert 8: Top historical performers should have higher expected kills
            assert top_avg_expected > bottom_avg_expected, (
                f"Top {n_top} performers (historical avg kills: {historical_avg_kills[top_players].mean():.2f}, "
                f"expected kills: {top_avg_expected:.2f}) should have higher expected kills than "
                f"bottom {n_bottom} performers (historical avg kills: {historical_avg_kills[bottom_players].mean():.2f}, "
                f"expected kills: {bottom_avg_expected:.2f})"
            )

            # Assert 9: Verify the difference is meaningful (not just noise)
            difference = top_avg_expected - bottom_avg_expected
            assert (
                difference > 0.5
            ), f"Difference between top and bottom performers ({difference:.2f}) should be meaningful"

    # Assert 10: Verify probabilities have reasonable spread (not all mass on one class)
    max_probs = probabilities.max(axis=1)
    assert (
        max_probs.mean() < 0.95
    ), f"Probabilities should have some spread, but mean max probability is {max_probs.mean():.3f}"

    # Assert 11: Verify that higher point estimates lead to higher expected values
    # Sort by kills_pred and check that expected_kills generally increases
    sorted_df = future_df.sort_values("kills_pred")
    if len(sorted_df) > 10:
        # Compare first quartile vs last quartile
        q1_expected = sorted_df.head(len(sorted_df) // 4)["expected_kills"].mean()
        q4_expected = sorted_df.tail(len(sorted_df) // 4)["expected_kills"].mean()
        assert q4_expected > q1_expected, (
            f"Players with higher kills_pred should have higher expected_kills. "
            f"Q1 (low kills_pred) expected={q1_expected:.2f}, Q4 (high kills_pred) expected={q4_expected:.2f}"
        )
