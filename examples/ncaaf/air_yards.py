from typing import Sequence, Mapping, Callable

import narwhals.stable.v2 as nw
import numpy as np

import polars as pl

import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from narwhals.stable.v2.typing import IntoFrameT
from sklearn import clone
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

from spforge import ColumnNames, AutoPipeline, FeaturesGenerator
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.predictor import SklearnPredictor
from spforge.predictor._distribution import StudentTDistributionPredictor, DistributionManagerPredictor
from spforge.ratings import TeamRatingGenerator, RatingKnownFeatures
from spforge.ratings._base import RatingGenerator
from spforge.scorer._score import PWMSE, SklearnScorer
from spforge.transformers import NetOverPredictedTransformer
from spforge.transformers.base_transformer import BaseTransformer

kfdf = pd.read_csv(r'C:\Users\m.holmstrom\Downloads\New_Query_2025_12_04_11_24am.csv')
df = pd.read_csv(r"C:\Users\m.holmstrom\Downloads\New_Query_2025_09_30_10_17am (1).csv")
df = df[df['play_type'] == 'pass']

df = df.merge(kfdf[['sr_play_id', 'pass_yards_attempted_ncaaf_expectancy']], on='sr_play_id')
GENERIC_FEATS =['is_home', 'down', 'yardline_100', 'ydstogo']
TARGET = 'air_yards'
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
    def __init__(
        self,
        transformer: "RatingGenerator",
        *,
        group_by: list[str],
        mean_cols: list[str],
        static_cols: list[str] = (),
        validate_static: bool = True,
    ):
        super().__init__(
            features=transformer.features,
            features_out=transformer.features_out,
        )
        self.transformer = transformer
        self._group_by = group_by
        self._mean_cols = mean_cols
        self._static_cols = static_cols
        self._validate_static = validate_static

    def _group_and_agg(self, df):
        exprs = []

        for c in self._mean_cols:
            exprs.append(nw.col(c).mean().alias(c))

        for c in self._static_cols:
            exprs.append(nw.col(c).first().alias(c))

        grouped = df.group_by(self._group_by).agg(*exprs)

        if self._validate_static and self._static_cols:
            chk = df.group_by(self._group_by).agg(
                *[nw.col(c).n_unique().alias(c) for c in self._static_cols]
            )
            for c in self._static_cols:
                assert chk.filter(nw.col(c) != 1).height == 0, (
                    f"Column '{c}' is not constant within group"
                )

        return grouped

    @nw.narwhalify
    def fit_transform(
        self,
        df: IntoFrameT,
        column_names: "ColumnNames | None" = None,
    ) -> IntoFrameT:
        grouped = self._group_and_agg(df)
        grouped = nw.from_native(
            self.transformer.fit_transform(grouped, column_names=column_names)
        )
        return df.join(
            grouped.select([*self._group_by, *self.transformer.features_out]),
            on=self._group_by,
        )

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        grouped = self._group_and_agg(df)
        grouped = nw.from_native(self.transformer.transform(grouped))
        return df.join(
            grouped.select([*self._group_by, *self.transformer.features_out]),
            on=self._group_by,
        )

    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        grouped = self._group_and_agg(df)
        grouped = nw.from_native(self.transformer.future_transform(grouped))
        return df.join(
            grouped.select([*self._group_by, *self.transformer.features_out]),
            on=self._group_by,
        )



nop = NetOverPredictedTransformer(
    predictor=AutoPipeline(
        convert_cat_features_to_cat_dtype=True,
        predictor=SklearnPredictor(
            features=GENERIC_FEATS,
            target=TARGET,
            pred_column=f'{TARGET}_prediction_raw',
            estimator=LGBMRegressor(max_depth=3, verbose=-100, random_state=42)
        )
    )
)

rating_generator = TeamRatingGenerator(
    features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
    rating_change_multiplier_defense=10,
    rating_change_multiplier_offense=10,
    performance_column=nop.features_out[1],
    auto_scale_performance=True,
    column_names=team_col_names
)

grp = GroupbyRatingTransformer(
    transformer=rating_generator,
    group_by=['sr_game_id', 'posteam', 'is_home', 'game_date'],
    mean_cols=nop.features_out
)
features_generator = FeaturesGenerator(
    transformers=[
        nop, grp
    ],
    column_names=team_col_names
)
df = pl.DataFrame(df)
df = features_generator.fit_transform(df)


class DMLOutcomeRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-style regressor that outputs outcome predictions y_hat,
    but estimates the effect of treatment via DML (PLR) using a nonlinear basis Phi(T).

    Expected input X contains the treatment column at position `treatment_index`.
    All other columns are controls Xc.

    Prediction:
        y_hat = l_hat(Xc) + (Phi(T) - Phi(m_hat(Xc))) @ theta
    """

    def __init__(
        self,
        model_y,
        model_t,
        treatment_index: int,
        t_transformer=None,           # e.g. SplineTransformer(...)
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.treatment_index = treatment_index
        self.t_transformer = t_transformer
        self.n_splits = n_splits
        self.random_state = random_state

        self.theta_ = None
        self.model_y_full_ = None
        self.model_t_full_ = None
        self.t_transformer_ = None

    def _split_XT(self, X):
        X = np.asarray(X)
        T = X[:, self.treatment_index].reshape(-1)
        Xc = np.delete(X, self.treatment_index, axis=1)
        return Xc, T

    def fit(self, X, y, sample_weight=None):
        Xc, T = self._split_XT(X)
        Y = np.asarray(y).reshape(-1)

        # Default: nonlinear spline basis for T
        if self.t_transformer is None:
            self.t_transformer_ = SplineTransformer(
                n_knots=6, degree=3, knots="quantile",
                extrapolation="linear", include_bias=False
            )
        else:
            self.t_transformer_ = clone(self.t_transformer)

        # Fit transformer on T (safe; uses only T)
        Phi = self.t_transformer_.fit_transform(T.reshape(-1, 1))  # (n, k)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        y_res = np.empty_like(Y, dtype=float)
        Phi_res = np.empty_like(Phi, dtype=float)

        for tr, te in kf.split(Xc):
            my = clone(self.model_y)
            mt = clone(self.model_t)

            if sample_weight is not None:
                sw_tr = np.asarray(sample_weight)[tr]
                try:
                    my.fit(Xc[tr], Y[tr], sample_weight=sw_tr)
                except TypeError:
                    my.fit(Xc[tr], Y[tr])
                try:
                    mt.fit(Xc[tr], T[tr], sample_weight=sw_tr)
                except TypeError:
                    mt.fit(Xc[tr], T[tr])
            else:
                my.fit(Xc[tr], Y[tr])
                mt.fit(Xc[tr], T[tr])

            y_hat = my.predict(Xc[te])
            t_hat = mt.predict(Xc[te])

            y_res[te] = Y[te] - y_hat

            Phi_hat = self.t_transformer_.transform(np.asarray(t_hat).reshape(-1, 1))
            Phi_res[te] = Phi[te] - Phi_hat

        lr = LinearRegression(fit_intercept=False)
        lr.fit(Phi_res, y_res)
        self.theta_ = lr.coef_.reshape(-1)

        # Refit nuisance models on full data for prediction
        self.model_y_full_ = clone(self.model_y).fit(Xc, Y)
        self.model_t_full_ = clone(self.model_t).fit(Xc, T)

        return self

    def predict(self, X):
        if self.theta_ is None:
            raise ValueError("DMLOutcomeRegressor not fitted yet.")

        Xc, T = self._split_XT(X)

        l_hat = self.model_y_full_.predict(Xc)
        t_hat = self.model_t_full_.predict(Xc)

        Phi = self.t_transformer_.transform(np.asarray(T).reshape(-1, 1))
        Phi_hat = self.t_transformer_.transform(np.asarray(t_hat).reshape(-1, 1))

        # y_hat = l_hat + (Phi(T) - Phi(m_hat(X))) @ theta
        return l_hat + (Phi - Phi_hat) @ self.theta_

TREATMENT_COL = rating_generator.features_out[0]  # change to your actual column name
ALL_FEATS = [*GENERIC_FEATS, *rating_generator.features_out]
treat_idx = ALL_FEATS.index(TREATMENT_COL)

dml_estimator = DMLOutcomeRegressor(
    model_y=LGBMRegressor(max_depth=4, verbose=-100, random_state=42),
    model_t=LGBMRegressor(max_depth=4, verbose=-100, random_state=42),
    treatment_index=treat_idx,
    t_transformer=SplineTransformer(
        n_knots=6, degree=3, knots="quantile",
        extrapolation="linear", include_bias=False
    ),
    n_splits=5,
    random_state=42,
)
point_predictor = AutoPipeline(
    convert_cat_features_to_cat_dtype=True,
    predictor=SklearnPredictor(
        # estimator=LGBMRegressor(max_depth=4, verbose=-100, random_state=42),
        estimator=dml_estimator,
        target=TARGET,
        features=[*GENERIC_FEATS, *rating_generator.features_out]
    )
)
distribution_predictor = StudentTDistributionPredictor(
    point_estimate_pred_column=point_predictor.pred_column,
    max_value=50,
    min_value=-5,
    target=point_predictor.target,
    df=10
)
dist_manager = DistributionManagerPredictor(
    point_predictor=point_predictor,
    distribution_predictor=distribution_predictor
)
lgbm_classifier_predictor = AutoPipeline(
    one_hot_encode_cat_features=True,
    predictor=SklearnPredictor(
    pred_column=f'{TARGET}_probabilities',
    estimator=LGBMClassifier(
        verbose=-100,
        max_depth=3
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
    # predictor=lgbm_classifier_predictor,
    match_id_column_name=team_col_names.match_id
)
df = cross_validator.generate_validation_df(df)

pwmse = PWMSE(
    labels=list(range(distribution_predictor.min_value, distribution_predictor.max_value + 1)),
    #labels=list(range(lgbm_classifier_predictor.min_target, lgbm_classifier_predictor.max_target + 1)),
    pred_column=distribution_predictor.pred_column,
    target=dist_manager.target
)
score = pwmse.score(df)
print(score)

pwmse = SklearnScorer(
    scorer_function=mean_pinball_loss,
    #labels=list(range(lgbm_classifier_predictor.min_target, lgbm_classifier_predictor.max_target + 1)),
    params={'alpha': 0.8},
    pred_column=nop.predictor.pred_column,
    target=dist_manager.target
)
score = pwmse.score(df)
print(score)


pwmse = SklearnScorer(
    scorer_function=mean_pinball_loss,
    #labels=list(range(lgbm_classifier_predictor.min_target, lgbm_classifier_predictor.max_target + 1)),
    params={'alpha':0.8},
    pred_column=point_predictor.pred_column,
    target=dist_manager.target
)
score = pwmse.score(df)
print(score)
