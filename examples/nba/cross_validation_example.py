import polars as pl
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from examples import get_sub_sample_nba_data
from spforge import FeatureGeneratorPipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.data_structures import ColumnNames
from spforge.estimator import (
    DistributionManagerPredictor,
    NegativeBinomialEstimator,
)
from spforge.feature_generator import (
    LagTransformer,
    RollingWindowTransformer,
)
from spforge.pipeline import Pipeline
from spforge.scorer import Filter, Operator, OrdinalLossScorer, SklearnScorer

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

features_generator = FeatureGeneratorPipeline(
    column_names=column_names,
    feature_generators=[
        RollingWindowTransformer(features=["points"], window=15, granularity=["player_id"]),
        LagTransformer(features=["points"], lag_length=3, granularity=["player_id"]),
    ],
)
df = features_generator.fit_transform(df)
predictor = DistributionManagerPredictor(
    point_predictor=SklearnPredictor(
        estimator=LGBMRegressor(verbose=-100, random_state=42),
        features=features_generator.features_out,
        target="points",
        pred_column="points_estimate",
        weight_by_date=True,
        date_column=column_names.start_date,
    ),
    distribution_predictor=NegativeBinomialEstimator(
        max_value=40,
        target="points",
        point_estimate_pred_column="points_estimate",
        r_specific_granularity=["player_id"],
        predicted_r_weight=1,
        column_names=column_names,
    ),
)


pipeline = Pipeline(
    convert_cat_features_to_cat_dtype=True,
    predictor=predictor,
)

cross_validator = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    estimator=pipeline,
)
validation_df = cross_validator.generate_validation_df(df=df, return_features=True)

mean_absolute_scorer = SklearnScorer(
    pred_column=predictor.point_predictor.pred_column,
    target=predictor.target,
    scorer_function=mean_absolute_error,
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
)

mae_score = cross_validator.cross_validation_score(
    validation_df=validation_df, scorer=mean_absolute_scorer
)
print(f"MAE {mae_score}")

ordinal_scorer = OrdinalLossScorer(
    pred_column=predictor.pred_column,
    target=predictor.target,
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
    classes=range(0, predictor.distribution_predictor.max_value + 1),
)

ordinal_loss_score = cross_validator.cross_validation_score(
    validation_df=validation_df,
    scorer=ordinal_scorer,
)
print(f"Ordinal Loss {ordinal_loss_score}")
