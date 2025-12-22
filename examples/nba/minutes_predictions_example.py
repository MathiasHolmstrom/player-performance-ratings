import polars as pl
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, LGBMClassifier

from examples import get_sub_sample_nba_data
from spforge.cross_validator import MatchKFoldCrossValidator

from spforge.pipeline import Pipeline
from spforge.predictor import (
    SklearnPredictor,
    NegativeBinomialPredictor,
    DistributionManagerPredictor,
)

from spforge.data_structures import ColumnNames

from spforge.transformers import (
    RollingWindowTransformer,
    LagTransformer,
)

df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)
# df = df.filter(pl.col('minutes')>0)
column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_id",
)
df = df.sort(
    [
        column_names.start_date,
        column_names.match_id,
        column_names.team_id,
        column_names.player_id,
    ]
)

df = df.with_columns(pl.col("points").clip(0, 40).alias("points"))

predictor = DistributionManagerPredictor(
    point_predictor=SklearnPredictor(
        estimator=LGBMRegressor(verbose=-100, random_state=42),
        features=["location"],
        target="minutes",
        convert_cat_features_to_cat_dtype=True,
        weight_by_date=True,
        date_column=column_names.start_date,
    ),
    distribution_predictor=NegativeBinomialPredictor(
        max_value=40,
        target="points",
        point_estimate_pred_column="points_estimate",
        # predict_granularity=["game_id", "team_id"],
        r_specific_granularity=["player_id"],
        predicted_r_weight=1,
        column_names=column_names,
    ),
)

pipeline = Pipeline(
    lag_transformers=[
        RollingWindowTransformer(
            features=["points"], window=15, granularity=["player_id"]
        ),
        LagTransformer(features=["points"], lag_length=3, granularity=["player_id"]),
    ],
    predictor=predictor,
    column_names=column_names,
)
