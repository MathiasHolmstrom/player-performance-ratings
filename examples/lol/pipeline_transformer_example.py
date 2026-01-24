import polars as pl
from lightgbm import LGBMRegressor

from examples import get_sub_sample_lol_data
from spforge import AutoPipeline, ColumnNames, FeatureGeneratorPipeline
from spforge.distributions import NegativeBinomialEstimator
from spforge.feature_generator import LagTransformer, RollingWindowTransformer
from spforge.transformers import EstimatorTransformer

column_names = ColumnNames(
    team_id="teamname",
    match_id="gameid",
    start_date="date",
    player_id="player_uid",
    league="league",
    position="position",
)

df = get_sub_sample_lol_data(as_pandas=False, as_polars=True)
df = (
    df.with_columns(
        pl.concat_str([pl.col("playername"), pl.col("teamname")], separator="__").alias(
            column_names.player_id
        )
    )
    .filter(pl.col(column_names.position) != "team")
    .with_columns(
        pl.col(column_names.team_id)
        .n_unique()
        .over(column_names.match_id)
        .alias("team_count"),
        pl.col(column_names.player_id)
        .n_unique()
        .over([column_names.match_id, column_names.team_id])
        .alias("player_count"),
    )
    .filter((pl.col("team_count") == 2) & (pl.col("player_count") == 5))
    .drop(["team_count", "player_count"])
    .unique(subset=[column_names.match_id, column_names.player_id, column_names.team_id])
    .sort(
        [
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]
    )
)

most_recent_10_games = (
    df.select(pl.col(column_names.match_id))
    .unique(maintain_order=True)
    .tail(10)
    .get_column(column_names.match_id)
    .to_list()
)
historical_df = df.filter(~pl.col(column_names.match_id).is_in(most_recent_10_games))
future_df = df.filter(pl.col(column_names.match_id).is_in(most_recent_10_games)).drop("kills")

lag_transformers = [
    LagTransformer(features=["kills", "deaths"], lag_length=3, granularity=["player_uid"]),
    RollingWindowTransformer(
        features=["kills", "deaths"],
        window=20,
        min_periods=1,
        granularity=["player_uid"],
    ),
]

features_generator = FeatureGeneratorPipeline(
    column_names=column_names,
    feature_generators=lag_transformers,
)

historical_df = features_generator.fit_transform(historical_df).to_pandas()
future_df = features_generator.future_transform(future_df).to_pandas()

point_estimate_transformer = EstimatorTransformer(
    prediction_column_name="kills_estimate",
    estimator=LGBMRegressor(verbose=-100, random_state=42),
    features=features_generator.features_out,
)

probability_estimator = NegativeBinomialEstimator(
    max_value=15,
    point_estimate_pred_column="kills_estimate",
    r_specific_granularity=[column_names.player_id],
    predicted_r_weight=1,
    column_names=column_names,
)

pipeline = AutoPipeline(
    estimator=probability_estimator,
    estimator_features=features_generator.features_out,
    predictor_transformers=[point_estimate_transformer],
)

pipeline.fit(X=historical_df, y=historical_df["kills"])

future_point_estimates = pipeline.predict(future_df)
future_probabilities = pipeline.predict_proba(future_df)
future_df["kills_pred"] = future_point_estimates

print(future_df.head(5))
print(f"Probability matrix shape: {future_probabilities.shape}")
print(f"First row probabilities (0-15 kills): {future_probabilities[0]}")
