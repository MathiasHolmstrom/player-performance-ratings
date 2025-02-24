import polars as pl
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

from examples import get_sub_sample_nba_data
from player_performance_ratings.cross_validator import MatchKFoldCrossValidator

from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.predictor import Predictor

from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.scorer import SklearnScorer
from player_performance_ratings.scorer.score import Filter, Operator
from player_performance_ratings.transformers.lag_generators import (
    RollingMeanTransformer,
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

predictor = Predictor(
    estimator=LGBMRegressor(verbose=-100, random_state=42),
    convert_to_cat_feats_to_cat_dtype=True,
    estimator_features=["location"],
    target="points",
)

pipeline = Pipeline(
    # filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
    lag_generators=[RollingMeanTransformer(features=["points"], window=15)],
    predictor=predictor,
    column_names=column_names,
)


scorer = SklearnScorer(
    pred_column=predictor.pred_column,
    target=predictor.target,
    scorer_function=mean_absolute_error,
    validation_column="is_validation",
    filters=[Filter(column_name="minutes", value=0, operator=Operator.GREATER_THAN)],
)

cross_validator = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    predictor=pipeline,
)
validation_df = cross_validator.generate_validation_df(df=df, column_names=column_names)
score = cross_validator.cross_validation_score(
    validation_df=validation_df, scorer=scorer
)
print(score, len(validation_df))
