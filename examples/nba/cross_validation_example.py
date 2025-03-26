import polars as pl
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, LGBMClassifier

from examples import get_sub_sample_nba_data
from spforge.cross_validator import MatchKFoldCrossValidator

from spforge.pipeline import Pipeline
from spforge.predictor import SklearnPredictor

from spforge.data_structures import ColumnNames
from spforge.predictor.classifier import NegativeBinomialPredictor
from spforge.predictor.predictor import DistributionPredictor
from spforge.scorer import SklearnScorer, OrdinalLossScorer
from spforge.scorer import Filter, Operator
from spforge.transformers import (
    RollingMeanTransformer,
    LagTransformer,
)

df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)
# df = df.filter(pl.col('minutes')>0)
column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_name",
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

predictor = DistributionPredictor(
    point_predictor=SklearnPredictor(
        estimator=LGBMRegressor(verbose=-100, random_state=42),
        features=["location"],
        target="points",
        convert_cat_features_to_cat_dtype=True,
        pred_column="points_estimate",
    ),
    distribution_predictor=NegativeBinomialPredictor(
        max_value=40, target="points", point_estimate_pred_column="points_estimate"
    ),
)

pipeline = Pipeline(
    lag_transformers=[
        RollingMeanTransformer(
            features=["points"], window=15, granularity=["player_id"]
        ),
        LagTransformer(features=["points"], lag_length=3, granularity=["player_id"]),
    ],
    predictor=predictor,
    column_names=column_names,
)

cross_validator = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    predictor=pipeline,
)
validation_df = cross_validator.generate_validation_df(
    df=df, column_names=column_names, return_features=True
)

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
)

ordinal_loss_score = cross_validator.cross_validation_score(
    validation_df=validation_df, scorer=ordinal_scorer
)
print(f"Ordinal Loss {ordinal_loss_score}")

lgbm_classifier_predictor = SklearnPredictor(
    estimator=LGBMClassifier(verbose=-100, random_state=42, max_depth=2),
    features=[
        *pipeline.lag_transformers[0].features_out,
        "location",
        predictor.point_predictor.pred_column,
    ],
    target=predictor.target,
    pred_column="lgbm_classifier_point_estimate",
    convert_cat_features_to_cat_dtype=True,
    multiclass_output_as_struct=True,
)

lgbm_classifier_cross_validator = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    predictor=lgbm_classifier_predictor,
)

validation_df = lgbm_classifier_cross_validator.generate_validation_df(
    df=validation_df, column_names=column_names
)

ordinal_scorer_lgbm_classifier = OrdinalLossScorer(
    pred_column=lgbm_classifier_predictor.pred_column,
    target=predictor.target,
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
)


lgbm_classifier_ordinal_loss_score = (
    lgbm_classifier_cross_validator.cross_validation_score(
        validation_df=validation_df, scorer=ordinal_scorer_lgbm_classifier
    )
)
print(f"Ordinal Loss Lgbm Classifier {lgbm_classifier_ordinal_loss_score}")
