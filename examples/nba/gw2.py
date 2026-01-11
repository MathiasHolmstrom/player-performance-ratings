import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from spforge import ColumnNames, Pipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.estimator import GroupByPredictor, SklearnPredictor
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures
from spforge.scorer import SklearnScorer
from spforge.transformers import LagTransformer, RollingWindowTransformer

df = pd.read_parquet("data/game_player_subsample.parquet")
df["participation_weight"] = df["minutes"] / df["game_minutes"].mean()
# Reloads everything from scratch to ensure we are not using previously added columns
df = pd.read_parquet("data/game_player_subsample.parquet")
df["participation_weight"] = df["minutes"] / df["game_minutes"].mean()
df["score_difference"] = df["score"] - df["score_opponent"]
column_names_participation_weight = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_id",
    participation_weight="participation_weight",
)

lag_player_transformer = LagTransformer(
    features=["plus_minus"], granularity=["player_id"], lag_length=4
)

df["score_difference"] = df["score"] - df["score_opponent"]

lag_team_transformer = LagTransformer(
    features=["score_difference"], granularity=["team_id"], lag_length=4
)
df = lag_player_transformer.transform_historical(df, column_names=column_names_participation_weight)
df = lag_team_transformer.transform_historical(df, column_names=column_names_participation_weight)
rating_generator_plus_minus_participation_weight = PlayerRatingGenerator(
    performance_column="plus_minus",
    non_predictor_known_features_out=[
        RatingKnownFeatures.PLAYER_RATING,
        RatingKnownFeatures.TEAM_RATING_PROJECTED,
        RatingKnownFeatures.OPPONENT_RATING_PROJECTED,
    ],
    auto_scale_performance=True,
    column_names=column_names_participation_weight,
    suffix="_plus_minus_pw",
)

game_winner_pipeline = Pipeline(
    lag_transformers=[lag_player_transformer, lag_team_transformer],
    rating_generators=rating_generator_plus_minus_participation_weight,
    predictor=GroupByPredictor(
        game_id_colum=column_names_participation_weight.match_id,
        team_id_column=column_names_participation_weight.team_id,
        predictor=SklearnPredictor(
            features=[
                "location"
            ],  # only feaure we manually specify. Other lag and rating generator features added automatically
            target="won",
            estimator=LogisticRegression(),
        ),
        impute_missing_values=True,
        one_hot_encode_cat_features=True,  # one-hot-encodes any categorical features
    ),
    column_names=column_names_participation_weight,
)

cross_validator = MatchKFoldCrossValidator(
    match_id_column_name="game_id",
    date_column_name="start_date",
    estimator=game_winner_pipeline,
)

log_loss_scorer = SklearnScorer(
    pred_column=game_winner_pipeline.pred_column, scorer_function=log_loss, target="won"
)
df = cross_validator.generate_validation_df(df, add_train_prediction=True)
print(
    f"logloss score Pipeline {cross_validator.cross_validation_score(df, scorer=log_loss_scorer )}"
)

rolling_mean_transformer = RollingWindowTransformer(
    window=15,
    features=["minutes"],
    granularity=["player_id"],
    update_column="game_id",
)
df = rolling_mean_transformer.transform_historical(df)
df["projected_participation_weight"] = df[rolling_mean_transformer.features_out[0]] / 48
df["projected_participation_weight"] = df["projected_participation_weight"].fillna(
    df["projected_participation_weight"].mean()
)
df.tail()

column_names_projected_participation_weight = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_id",
    participation_weight="participation_weight",
    projected_participation_weight="projected_participation_weight",
)


rating_generator_plus_minus_projected_participation_weight = PlayerRatingGenerator(
    performance_column="plus_minus",
    non_predictor_known_features_out=[
        RatingKnownFeatures.PLAYER_RATING,
        RatingKnownFeatures.TEAM_RATING_PROJECTED,
        RatingKnownFeatures.OPPONENT_RATING_PROJECTED,
    ],
    auto_scale_performance=True,
    column_names=column_names_participation_weight,
    suffix="_plus_minus_pw",
)


game_winner_pipeline = Pipeline(
    lag_transformers=[lag_player_transformer, lag_team_transformer],
    rating_generators=rating_generator_plus_minus_participation_weight,
    predictor=GroupByPredictor(
        game_id_colum=column_names_participation_weight.match_id,
        team_id_column=column_names_participation_weight.team_id,
        predictor=SklearnPredictor(
            features=["location"],
            target="won",
            estimator=LogisticRegression(),
        ),
        impute_missing_values=True,
        one_hot_encode_cat_features=True,  # one-hot-encodes any categorical features
    ),
    column_names=column_names_projected_participation_weight,
)


cross_validator = MatchKFoldCrossValidator(
    match_id_column_name="game_id",
    date_column_name="start_date",
    estimator=game_winner_pipeline,
)
log_loss_scorer = SklearnScorer(
    pred_column=game_winner_pipeline.pred_column, scorer_function=log_loss, target="won"
)
df = cross_validator.generate_validation_df(df, add_train_prediction=True)
print(cross_validator.cross_validation_score(df, scorer=log_loss_scorer))
