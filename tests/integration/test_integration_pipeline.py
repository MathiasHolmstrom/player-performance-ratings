import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

from examples import get_sub_sample_lol_data
from spforge import ColumnNames, FeatureGeneratorPipeline, Pipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.performance_transformers._performance_manager import ColumnWeight
from spforge.estimator import NegativeBinomialEstimator
from spforge.ratings import RatingKnownFeatures, PlayerRatingGenerator
from spforge.feature_generator import RollingWindowTransformer, LagTransformer


def test_nba_feature_engineering_and_distribution_end_to_end():
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
        .assign(
            player_count=df.groupby(["gameid", "teamname"])["playername"].transform(
                "nunique"
            )
        )
        .loc[lambda x: x.player_count == 5]
    )
    df = df.assign(team_count=df.groupby("gameid")["teamname"].transform("nunique")).loc[
        lambda x: x.team_count == 2
    ]
    df = df.drop_duplicates(subset=["gameid", "playername", "teamname"])

    most_recent_10_games = df[column_names.match_id].unique()[-10:]
    historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)].copy()
    future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(
        columns=["result"]
    ).copy()

    rating_generator_player_kills = PlayerRatingGenerator(
        features_out=[RatingKnownFeatures.PLAYER_RATING],
        performance_column="performance_kills",
        auto_scale_performance=True,
        performance_weights=[ColumnWeight(name="kills", weight=1)],
    )
    rating_generator_result = PlayerRatingGenerator(
        features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        performance_column="result",
        non_predictor_features_out=[RatingKnownFeatures.PLAYER_RATING],
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
        column_names=column_names,
        transformers=[rating_generator_player_kills, rating_generator_result, *lag_generators],
    )

    historical_df = features_generator.fit_transform(historical_df)

    game_winner_model = Pipeline(
        estimator=LogisticRegression(max_iter=1000),
        target="result",
        features=list(rating_generator_result.features_out),
        pred_column="result_pred",
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=False,
    )

    cv_game_winner = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=game_winner_model,                # <- sklearn-compatible estimator (your Pipeline wrapper is fine)
        prediction_column_name="result_oof",         # <- single output col
        n_splits=3,
    )
    cv_game_winner.features = game_winner_model.features
    cv_game_winner.target = game_winner_model.target

    historical_df = cv_game_winner.generate_validation_df(historical_df)
    assert "result_oof" in historical_df.columns

    # CV stores:
    # - binary classification => float prob of class 1
    # - multiclass => list[float] per row
    def _prob1(x):
        if isinstance(x, list):
            return float(x[1]) if len(x) > 1 else float(x[0])
        return float(x)

    historical_df["result_prob1"] = historical_df["result_oof"].apply(_prob1)

    # -------------------------
    # Player kills (regressor)
    # -------------------------
    player_kills_features = ["result_prob1", *features_generator.features_out]

    player_kills_model = Pipeline(
        estimator=LGBMRegressor(verbose=-100),
        target="kills",
        features=list(player_kills_features),
        pred_column="kills_pred",
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
        scale_features=False,
    )

    cv_player_kills = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=player_kills_model,
        prediction_column_name="kills_oof",
        n_splits=3,
    )
    cv_player_kills.features = player_kills_model.features
    cv_player_kills.target = player_kills_model.target

    historical_df = cv_player_kills.generate_validation_df(historical_df)
    assert "kills_oof" in historical_df.columns

    future_df = features_generator.future_transform(future_df)

    game_winner_model.fit(historical_df)
    proba = game_winner_model.predict_proba(future_df[game_winner_model.features])
    future_df["result_prob1"] = proba[:, 1]
    player_kills_model.fit(historical_df)
    kills_pred = player_kills_model.predict(future_df[player_kills_model.features])
    future_df[player_kills_model.pred_column] = kills_pred


    probability_predictor = NegativeBinomialEstimator(
        point_estimate_pred_column=player_kills_model.pred_column,
        max_value=15,
    )
    probability_predictor.fit(historical_df, historical_df['kills'])
    future_df = probability_predictor.predict(future_df)

    assert probability_predictor.pred_column in future_df.columns
