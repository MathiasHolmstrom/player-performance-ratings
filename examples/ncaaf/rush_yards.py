import narwhals.stable.v2 as nw

import polars as pl

import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from narwhals.stable.v2.typing import IntoFrameT

from spforge import ColumnNames, AutoPipeline, FeaturesGenerator
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.predictor import SklearnPredictor
from spforge.predictor._distribution import StudentTDistributionPredictor, DistributionManagerPredictor
from spforge.ratings import TeamRatingGenerator, RatingKnownFeatures
from spforge.ratings._base import RatingGenerator
from spforge.scorer._score import PWMSE
from spforge.transformers import NetOverPredictedTransformer
from spforge.transformers.base_transformer import BaseTransformer

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
# u = df[df['sr_game_id'] == '42f08caa-beab-4404-8ad9-20b48b9c9157']
#
team_col_names = ColumnNames(
    player_id='posteam',
    team_id='posteam',
    match_id='sr_game_id',
    start_date='game_date',
)


class GroupbyRatingTransformer(BaseTransformer):

    def __init__(self, transformer: RatingGenerator):
        super().__init__(features=transformer.features, features_out=transformer.features_out)
        self.transformer = transformer
        self._group_by = ['sr_game_id', 'posteam', 'is_home', 'game_date']

    @nw.narwhalify
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        grouped = df.group_by(self._group_by).agg(
            nw.col('net_over_predicted_rush_yars_pred_raw').mean().alias('net_over_predicted_rush_yars_pred_raw'),
            nw.len().alias('rush_attempts')
        )
        assert grouped.group_by('sr_game_id').agg(nw.col('posteam').n_unique())['posteam'].n_unique() == 1
        grouped = nw.from_native(self.transformer.fit_transform(grouped, column_names=column_names))
        return df.join(grouped.select([*self._group_by, *self.transformer.features_out]),
                       on=self._group_by)

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        grouped = df.group_by(self._group_by).agg(
            nw.col('net_over_predicted_rush_yars_pred_raw').mean().alias('net_over_predicted_rush_yars_pred_raw'),
            nw.len().alias('rush_attempts')
        )
        grouped = nw.from_native(self.transformer.transform(grouped))
        return df.join(grouped.select([*self._group_by, *self.transformer.features_out]),
                       on=self._group_by)

    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        grouped = df.group_by(self._group_by).agg(
            nw.col('net_over_predicted_rush_yars_pred_raw').mean().alias('net_over_predicted_rush_yars_pred_raw'),
            nw.len().alias('rush_attempts')
        )
        grouped = nw.from_native(self.transformer.future_transform(grouped))
        return df.join(grouped.select([*self._group_by, *self.transformer.features_out]),
                       on=self._group_by)


nop = NetOverPredictedTransformer(
    target_name='rush_yards',
    net_over_predicted_col='',
    estimator=AutoPipeline(
        estimator=LGBMRegressor(max_depth=4, verbose=-100),
        feature_names=['is_home', 'down', 'yardline_100', 'ydstogo'],

    )
)

rating_generator = TeamRatingGenerator(
    features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED, RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED, RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED],
    performance_column=nop.features_out[1],
    auto_scale_performance=True,
    column_names=team_col_names
)

grp = GroupbyRatingTransformer(
    transformer=rating_generator
)
features_generator = FeaturesGenerator(
    transformers=[
        nop, grp
    ],
    column_names=team_col_names
)
df = pl.DataFrame(df)
df = features_generator.fit_transform(df)

point_predictor = AutoPipeline(
    convert_cat_features_to_cat_dtype=True,
    predictor=SklearnPredictor(
        estimator=LGBMRegressor(max_depth=4, verbose=-100),
        target='rush_yards',
        features=[*rating_generator.features_out, 'is_home', 'yardline_100', 'ydstogo', 'down']
    )
)
distribution_predictor = StudentTDistributionPredictor(
    point_estimate_pred_column=point_predictor.pred_column,
    max_value=50,
    min_value=-5,
    target=point_predictor.target,
    df=8
)
dist_manager = DistributionManagerPredictor(
    point_predictor=point_predictor,
    distribution_predictor=distribution_predictor
)
lgbm_classifier_predictor = AutoPipeline(
    one_hot_encode_cat_features=True,
    predictor=SklearnPredictor(
    pred_column='rush_yards_probabilities',
    estimator=LGBMClassifier(
        verbose=-100,
        max_depth=4
    ),
    features=point_predictor.features,
    target=point_predictor.target,
),
    min_target=-5,
    max_target=35
)

cross_validator = MatchKFoldCrossValidator(
    predictor=dist_manager,
    date_column_name=team_col_names.start_date,
    match_id_column_name=team_col_names.match_id
)
df = cross_validator.generate_validation_df(df).to_pandas()

pwmse = PWMSE(
    labels=list(range(distribution_predictor.min_value, distribution_predictor.max_value + 1)),
    pred_column=distribution_predictor.pred_column,
    target=dist_manager.target
)
score = pwmse.score(df)
print(score)
