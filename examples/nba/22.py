import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from spforge import ColumnNames, Pipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.estimator import SklearnPredictor
from spforge.scorer import SklearnScorer
from spforge.transformers import LagTransformer, RollingWindowTransformer

df = pd.read_parquet("data/game_player_subsample.parquet")
df["points_per_minute"] = df["points"] / df["minutes"]
df.head()

rm_transformer_window20 = RollingWindowTransformer(
    features=["points", "points_per_minute", "minutes"],
    granularity=["player_id"],
    window=20,
    update_column="game_id",
)
rm_transformer_window10 = RollingWindowTransformer(
    features=["points", "points_per_minute", "minutes"],
    granularity=["player_id"],
    window=10,
    update_column="game_id",
)
rm_transformer_window5 = RollingWindowTransformer(
    features=["points", "points_per_minute", "minutes"],
    granularity=["player_id"],
    window=5,
    update_column="game_id",
)
rm_transformer_window40 = RollingWindowTransformer(
    features=["points", "points_per_minute", "minutes"],
    granularity=["player_id"],
    window=40,
    update_column="game_id",
)

lag_transformer = LagTransformer(
    features=["points", "points_per_minute", "minutes"],
    granularity=["player_id"],
    lag_length=5,
    update_column="game_id",
)


column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_id",
)
predictor = SklearnPredictor(
    estimator=LGBMRegressor(max_depth=4, verbose=-100),
    target="points",
    features=["team_id_opponent"],
    convert_cat_features_to_cat_dtype=True,
)
pipeline = Pipeline(
    lag_transformers=[
        lag_transformer,
        rm_transformer_window5,
        rm_transformer_window10,
        rm_transformer_window20,
        rm_transformer_window40,
    ],
    predictor=predictor,
    column_names=column_names,
)

cross_validator_cat_feats = MatchKFoldCrossValidator(
    match_id_column_name="game_id",
    date_column_name="start_date",
    estimator=pipeline,
)
mean_absolute_scorer = SklearnScorer(
    pred_column=pipeline.pred_column,
    scorer_function=mean_absolute_error,
    target="points",
)
df = cross_validator_cat_feats.generate_validation_df(df, add_train_prediction=True)
cross_validator_cat_feats.cross_validation_score(df, scorer=mean_absolute_scorer)
