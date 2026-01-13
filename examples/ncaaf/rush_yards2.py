import narwhals.stable.v2 as nw
import numpy as np

import polars as pl

import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from narwhals.stable.v2.typing import IntoFrameT
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from examples.ncaaf.add_team_div import add_division_and_conference
from examples.ncaaf.score_ex import run_score
from spforge import ColumnNames, AutoPipeline, FeatureGeneratorPipeline
from spforge.base_feature_generator import FeatureGenerator
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.estimator._distribution import StudentTDistributionEstimator
from spforge.estimator.sklearn_estimator import ConditionalEstimator
from spforge.feature_generator._net_over_predicted import NetOverPredictedFeatureGenerator

from spforge.ratings import TeamRatingGenerator, RatingKnownFeatures
from spforge.scorer import SklearnScorer

from spforge.transformers import EstimatorTransformer

kfdf = pd.read_csv(r'C:\Users\m.holmstrom\Downloads\New_Query_2025_09_30_10_17am (2).csv')
df = pd.read_csv(r"C:\Users\m.holmstrom\Downloads\New_Query_2025_09_30_10_17am (1).csv")
df = df[df['play_type'] == 'rush']
# model = LGBMRegressor(max_depth=3, verbose=-100)
# features = ['down', 'yardline_100', 'qtr', 'ydstogo']
# model.fit(df[features], df['rush_yards'])
# df['raw_predicted_rush_yards'] = model.predict(df[features])
# df['nop_rush_yards'] = df['rush_yards'] - df['raw_predicted_rush_yards']
#
df = df.merge(kfdf[['sr_play_id', 'rush_ncaaf_expectancy']], on='sr_play_id')
# df['player_name'] = df['description'].str.replace(r"\s+", " ", regex=True).str.split().str[0]
# df = df[~df['rush_yards'].isna()]
df.loc[df['posteam'] == df['home_team'], 'is_home'] = 1
df.loc[df['posteam'] != df['home_team'], 'is_home'] = 0
df['game_date'] = pd.to_datetime(df['game_date'])
df.loc[df['is_home'] == 1, 'defteam'] = df['away_team']
df.loc[df['is_home'] == 0, 'defteam'] = df['home_team']
df['defteam2'] = df['defteam']
df['posteam2'] = df['posteam']
df = df.assign(team_count=df.groupby("sr_game_id")["defteam"].transform("nunique")).loc[
    lambda x: x.team_count == 2
]
df = add_division_and_conference(df=df, team_col_name='posteam')
df = add_division_and_conference(df=df, team_col_name='defteam', prefix='def_')
# u = df[df['sr_game_id'] == '42f08caa-beab-4404-8ad9-20b48b9c9157']
df['division_matchup'] = (df['division'] + df['def_division'])
#
team_col_names = ColumnNames(
    player_id='posteam',
    team_id='posteam',
    match_id='sr_game_id',
    start_date='game_date',
    league='division',
    # participation_weight='rush_attempts_ratio'
)


class GroupbyRatingTransformer(FeatureGenerator):

    def __init__(self, feature_generator: FeatureGenerator):
        super().__init__(features_out=feature_generator.features_out)
        self.transformer = feature_generator
        self._group_by = ['sr_game_id', 'posteam', 'is_home', 'game_date', 'division', 'division_matchup']

    @nw.narwhalify
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        grouped = df.group_by(self._group_by).agg(
            nw.col('net_over_predicted_rush_yards').mean().alias('net_over_predicted_rush_yards'),
            nw.len().alias('rush_attempts')
        )
        grouped = grouped.with_columns(
            (nw.col('rush_attempts') / 40).clip(upper_bound=1.0).alias('rush_attempts_ratio')
        )
        assert grouped.group_by('sr_game_id').agg(nw.col('posteam').n_unique())['posteam'].n_unique() == 1
        grouped = nw.from_native(self.transformer.fit_transform(grouped))
        return df.join(grouped.select([*self._group_by, *self.transformer.features_out]),
                       on=self._group_by)

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        grouped = df.group_by(self._group_by).agg(
            nw.col('net_over_predicted_rush_yards').mean().alias('net_over_predicted_rush_yards'),
            nw.len().alias('rush_attempts')
        )
        grouped = grouped.with_columns(
            (nw.col('rush_attempts')/40).clip(upper_bound=1.0).alias('rush_attempts_ratio')
        )
        grouped = nw.from_native(self.transformer.transform(grouped))
        return df.join(grouped.select([*self._group_by, *self.transformer.features_out]),
                       on=self._group_by)

    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        grouped = df.group_by(self._group_by).agg(
            nw.col('net_over_predicted_rush_yards').mean().alias('net_over_predicted_rush_yards'),
            nw.len().alias('rush_attempts')
        )
        grouped = nw.from_native(self.transformer.future_transform(grouped))
        return df.join(grouped.select([*self._group_by, *self.transformer.features_out]),
                       on=self._group_by)


generic_feature_names = ['is_home', 'down', 'yardline_100', 'ydstogo']

nop = NetOverPredictedFeatureGenerator(
    target_name='rush_yards',
    net_over_predicted_col='net_over_predicted_rush_yards',
    features=generic_feature_names,
    estimator=AutoPipeline(
        estimator=LGBMRegressor(max_depth=3, verbose=-100, random_state=42),
        feature_names=generic_feature_names,

    )
)

rating_generator = TeamRatingGenerator(
    performance_column=nop.features_out[0],
    auto_scale_performance=True,
    confidence_value_denom=30,
    column_names=team_col_names,
    start_league_ratings={
        'FBS': 1100,
        'FCS': 900
    }
)

grp = GroupbyRatingTransformer(
    feature_generator=rating_generator
)
features_generator = FeatureGeneratorPipeline(
    feature_generators=[
        nop, grp
    ],
    column_names=team_col_names
)
df = pl.DataFrame(df)
df = features_generator.fit_transform(df)
grouped = df.group_by(['sr_game_id', 'posteam', 'game_date', 'division', 'is_home', 'division_matchup']).agg(
    pl.col(features_generator.features_out).mean(), pl.col('rush_ncaaf_expectancy').mean(), pl.col('rush_yards').mean())
mean_rush_yards = grouped['rush_yards'].mean()
grouped = grouped.with_columns(
    (pl.lit(mean_rush_yards) + pl.col('rush_ncaaf_expectancy')).alias('rush_yards_kf_pred')
)
mean_home_kf_prediction = grouped.filter(pl.col('is_home') == 1)['rush_ncaaf_expectancy'].mean()
mean_away_kf_prediction = grouped.filter(pl.col('is_home') == 0)['rush_ncaaf_expectancy'].mean()



adj = (mean_home_kf_prediction - mean_away_kf_prediction) * 0.5
grouped = grouped.with_columns(
    pl.when(
        pl.col('is_home') == 1
    ).then(
        pl.col('rush_yards_kf_pred') - pl.lit(adj)
    ).otherwise(
        pl.col('rush_yards_kf_pred') + pl.lit(adj)
    ).alias('rush_yards_kf_pred_no_home')
)
model = make_pipeline(
    SplineTransformer(n_knots=5, degree=2),
    LinearRegression()
)

cross_validator_lgm = MatchKFoldCrossValidator(
    # estimator=LGBMRegressor(max_depth=2, n_estimators=00, random_state=42, learning_rate=0.02),
    estimator=AutoPipeline(
        estimator=model,
        feature_names=rating_generator.features_out + ['division_matchup']
    ),
    date_column_name=team_col_names.start_date,
    match_id_column_name=team_col_names.match_id,
    target_column='rush_yards',
    prediction_column_name='rush_yards_lgbm_prediction_rating',
    features=rating_generator.features_out + ['division_matchup']
)
grouped = cross_validator_lgm.generate_validation_df(grouped)
print('rating MAE score', SklearnScorer(
    target=cross_validator_lgm.target_column,
    pred_column=cross_validator_lgm.prediction_column_name,
    scorer_function=mean_absolute_error,
).score(grouped))

print('rating MAE kf', SklearnScorer(
    target=cross_validator_lgm.target_column,
    pred_column='rush_yards_kf_pred_no_home',
    scorer_function=mean_absolute_error
).score(grouped))

min_value = -5
max_value = 35

lgbm_classifier_predictor = AutoPipeline(
    min_target=min_value,
    max_target=max_value,
    estimator=LGBMClassifier(
        verbose=-100,
        max_depth=4
    ),
    feature_names=generic_feature_names + ['rush_ncaaf_expectancy']
)

cross_validator_lgm = MatchKFoldCrossValidator(
    estimator=lgbm_classifier_predictor,
    date_column_name=team_col_names.start_date,
    match_id_column_name=team_col_names.match_id,
    target_column='rush_yards',
    prediction_column_name='rush_yards_probabilities_lgm'
)
df = cross_validator_lgm.generate_validation_df(df)
condition_estimator = AutoPipeline(
    min_target=min_value,
    max_target=max_value,
    estimator=ConditionalEstimator(
        gate_estimator=LGBMClassifier(max_depth=4, verbose=-100),
        gate_distance_col='ydstogo',
        outcome_0_estimator=LGBMClassifier(max_depth=4, verbose=-100),
        outcome_1_estimator=AutoPipeline(
            estimator=LGBMClassifier(max_depth=4, verbose=-100),
            min_target=-30,
            max_target=0,
            feature_names=generic_feature_names + ['rush_ncaaf_expectancy'],
        ),
        outcome_0_value=0,
        outcome_1_value=1,
    ),
    feature_names=generic_feature_names + ['rush_ncaaf_expectancy'],
)
cross_validator_condition = MatchKFoldCrossValidator(
    estimator=condition_estimator,
    date_column_name=team_col_names.start_date,
    match_id_column_name=team_col_names.match_id,
    target_column='rush_yards',
    prediction_column_name='rush_yards_probabilities_cond'
)
df = cross_validator_condition.generate_validation_df(df)

distribution_pipeline = AutoPipeline(
    predictor_transformers=[
        EstimatorTransformer(estimator=LGBMRegressor(max_depth=4, verbose=-100),
                             prediction_column_name='point_prediction'
                             ),
    ],
    estimator=StudentTDistributionEstimator(
        sigma_conditioning_columns=("point_prediction", "yardline_100"),
        sigma_bins_per_column=(3, 2),
        point_estimate_pred_column='point_prediction',
        max_value=max_value,
        min_value=min_value,
        target='rush_yards',
        df=3
    ),
    feature_names=generic_feature_names + ['rush_ncaaf_expectancy']

)

cross_validator_dist = MatchKFoldCrossValidator(
    estimator=distribution_pipeline,
    date_column_name=team_col_names.start_date,
    match_id_column_name=team_col_names.match_id,
    target_column='rush_yards',
    prediction_column_name='rush_yards_probabilities_dist'
)
df = cross_validator_dist.generate_validation_df(df)

df.write_parquet('rush_yards_cv.parquet')
run_score(df)
