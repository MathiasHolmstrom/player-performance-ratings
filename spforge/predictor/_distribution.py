from typing import Optional

import numpy as np
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom
from sklearn.preprocessing import StandardScaler

from spforge import ColumnNames
from spforge.predictor_transformer._simple_transformer import SimpleTransformer
from spforge.scorer import OrdinalLossScorer, apply_filters, Filter

from spforge.predictor import (
    BasePredictor,
)
from spforge.scorer import BaseScorer


def neg_binom_log_likelihood(r, actual_points, predicted_points):
    if r <= 0:
        return np.inf

    term1 = gammaln(actual_points + r)
    term2 = gammaln(r)
    term3 = gammaln(actual_points + 1)
    term4 = r * np.log(r / (r + predicted_points))
    term5 = actual_points * np.log(predicted_points / (r + predicted_points))

    if np.any(np.isnan(term4)) or np.any(np.isnan(term5)):
        raise ValueError("NaN detected in log-likelihood terms")

    log_likelihood = term1 - term2 - term3 + term4 + term5

    if np.isnan(np.sum(log_likelihood)):
        print(f"NaN detected in log-likelihood for r={r}")
        raise ValueError("NaN detected in log-likelihood terms")

    return -np.sum(log_likelihood)


class NegativeBinomialPredictor(BasePredictor):

    def __init__(
        self,
        point_estimate_pred_column: str,
        max_value: int,
        target: str,
        min_value: int = 0,
        pred_column: Optional[str] = None,
        predicted_r_iterations: int = 6,
        predict_granularity: Optional[list[str]] = None,
        multiclass_output_as_struct: bool = True,
        r_specific_granularity: Optional[list[str]] = None,
        r_rolling_mean_window: int = 40,
        predicted_r_weight: float = 0.7,
        column_names: Optional[ColumnNames] = None,
    ):
        self.point_estimate_pred_column = point_estimate_pred_column
        pred_column = pred_column or f"{target}_probabilities"
        self.multiclass_output_as_struct = multiclass_output_as_struct
        self.r_specific_granularity = r_specific_granularity
        self.predicted_r_weight  =predicted_r_weight
        self.r_rolling_mean_window = r_rolling_mean_window
        self.column_names = column_names
        self.predict_granularity = predict_granularity

        super().__init__(
            target=target,
            features=[],
            pred_column=pred_column,
            multiclass_output_as_struct=multiclass_output_as_struct,
        )

        if self.r_specific_granularity and not self.column_names:
            raise ValueError(
                "r_specific_granularity is set but column names is not provided."
            )

        self.max_value = max_value
        self.min_value = min_value
        self._multipliers = list(range(2, 2 + predicted_r_iterations * 2, 2))
        self._target_scaler = StandardScaler()
        self._best_multiplier = None
        self._mean_r = None
        self._scores = []

    @nw.narwhalify
    def train(self, df: FrameT, features: Optional[list[str]] = None) -> None:

        positive_predicted_rows = df.filter(nw.col(self.point_estimate_pred_column) > 0)
        if self.predict_granularity:
            positive_predicted_rows = positive_predicted_rows.group_by(
                self.predict_granularity
            ).agg(
                [
                    nw.col([self.point_estimate_pred_column, self.target]).mean(),
                    nw.col(self.column_names.start_date).median(),
                ]
            )
        result = minimize(
            neg_binom_log_likelihood,
            x0=np.array([1.0]),  # Initial guess
            args=(
                positive_predicted_rows[self.target].to_numpy(),
                positive_predicted_rows[self.point_estimate_pred_column].to_numpy(),
            ),
            method="L-BFGS-B",
            bounds=[(0.01, None)],  # Ensure r stays positive
        )
        self._mean_r = float(result.x[0])
        if self.r_specific_granularity:
            gran_grp = self._grp_to_r_granularity(df)
            gran_grp = self._add_predicted_r(gran_grp)
            self.historical_df = gran_grp.select(
                [
                    "__predicted_r",
                    *self.r_specific_granularity,
                    self.column_names.match_id,
                    self.column_names.start_date,
                    self.target,
                    self.point_estimate_pred_column,
                ]
            ).to_native()

    @nw.narwhalify
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:
        input_cols = df.columns
        if self.r_specific_granularity:
            pred_gran_grp = self._grp_to_r_granularity(df)
            gran_grp = nw.concat(
                [
                    nw.from_native(self.historical_df),
                    pred_gran_grp.select(
                        [
                            self.column_names.match_id,
                            *self.r_specific_granularity,
                            self.target,
                            self.point_estimate_pred_column,
                        ]
                    ),
                ],
                how="diagonal",
            ).unique([self.column_names.match_id, *self.r_specific_granularity])

            gran_grp = self._add_predicted_r(gran_grp)
            pre_join_df_row_count = len(df)
            pred_df = df.join(
                gran_grp.select(
                    [
                        self.column_names.match_id,
                        *self.r_specific_granularity,
                        "__predicted_r",
                    ]
                ),
                on=[self.column_names.match_id, *self.r_specific_granularity],
                how="left",
            )
            pred_df = pred_df.with_columns(
                ((nw.col('__predicted_r')*self.predicted_r_weight+nw.lit(self._mean_r)*(1-self.predicted_r_weight))).alias('__predicted_r')
            )

            assert len(pred_df) == pre_join_df_row_count
            df = self._add_probabilities(df=pred_df).select([*input_cols, self.pred_column])
            assert len(df) == pre_join_df_row_count
            return df

        else:
            df = df.with_columns(nw.lit(self._mean_r).alias("__predicted_r"))
            return self._add_probabilities(df=df).select(
                [*input_cols, self.pred_column]
            )

    def _grp_to_r_granularity(self, df: FrameT) -> FrameT:
        return (
            df.group_by([self.column_names.match_id, *self.r_specific_granularity])
            .agg(
                [
                    nw.col([self.point_estimate_pred_column, self.target]).mean(),
                    nw.col(self.column_names.start_date).median(),
                ]
            )
            .sort(self.column_names.start_date)
        )

    def _add_predicted_r(self, df: FrameT) -> FrameT:

        df = df.with_columns(
            [
                nw.col(self.point_estimate_pred_column)
                .rolling_mean(window_size=self.r_rolling_mean_window, min_samples=5)
                .over(self.r_specific_granularity)
                .alias("mu_roll"),
                nw.col(self.target)
                .rolling_var(window_size=self.r_rolling_mean_window, min_samples=5)
                .over(self.r_specific_granularity)
                .alias("var_y_roll"),
            ]
        )

        return df.with_columns(
            [
                nw.when(nw.col("var_y_roll") > nw.col("mu_roll"))
                .then(
                    (nw.col("mu_roll") ** 2)
                    / (nw.col("var_y_roll") - nw.col("mu_roll"))
                )
                .otherwise(nw.lit(self._mean_r))
                .alias("__predicted_r")
            ]
        )

    def _add_probabilities(self, df: FrameT) -> FrameT:

        all_outcome_probs = []
        classes_ = []

        if self.predict_granularity:
            if self.column_names and self.column_names.projected_participation_weight:
                df = (
                    df.with_columns(
                        (
                            nw.col(self.column_names.projected_participation_weight)
                            * nw.col("__predicted_r")
                        ).alias("raw_weighted__predicted_r")
                    )
                    .with_columns(
                        nw.col(self.column_names.projected_participation_weight)
                        .over(self.predict_granularity)
                        .alias("sum_projected_participation_weight")
                    )
                    .with_columns(
                        (
                            nw.col("raw_weighted__predicted_r")
                            / nw.col("sum_projected_participation_weight")
                        ).alias("__predicted_r")
                    )
                )

            pred_grp = df.group_by(self.predict_granularity).agg(
                [
                    nw.col("__predicted_r").mean().alias("__predicted_r"),
                    nw.col(self.point_estimate_pred_column)
                    .median()
                    .alias(self.point_estimate_pred_column),
                ]
            )

        else:
            pred_grp = df

        pred_grp = pred_grp.with_columns(
            (
                nw.col("__predicted_r")
                / (nw.col("__predicted_r") + nw.col(self.point_estimate_pred_column))
            ).alias("__predicted_p")
        )

        for row in pred_grp.iter_rows(named=True):
            point_range = np.arange(self.min_value, self.max_value + 1)
            r = row["__predicted_r"]
            p = row["__predicted_p"]
            prob = nbinom.pmf(point_range, r, p)
            if self.multiclass_output_as_struct:
                outcome_probs = {
                    str(i + self.min_value): float(prob[i]) for i in range(len(prob))
                }
            else:
                outcome_probs = prob
                classes_.append(np.argmax(prob))
            all_outcome_probs.append(outcome_probs)

        if not self.multiclass_output_as_struct:
            pred_grp = pred_grp.with_columns(
                nw.new_series(
                    name="classes",
                    values=classes_,
                    native_namespace=nw.get_native_namespace(df),
                )
            )
        else:
            pred_grp = pred_grp.with_columns(
                nw.new_series(
                    name=self.pred_column,
                    values=all_outcome_probs,
                    native_namespace=nw.get_native_namespace(df),
                )
            )
        if self.predict_granularity:
            return df.join(
                pred_grp.select([*self.predict_granularity, *self._pred_columns_added]),
                on=self.predict_granularity,
                how="left",
            )
        return pred_grp


class DistributionPredictor(BasePredictor):

    def __init__(
        self,
        point_predictor: BasePredictor,
        distribution_predictor: BasePredictor,
        filters: Optional[list[Filter]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        multiclass_output_as_struct: bool = False,
    ):
        self.point_predictor = point_predictor
        self.distribution_predictor = distribution_predictor

        super().__init__(
            target=point_predictor.target,
            pred_column=distribution_predictor.pred_column,
            features=point_predictor.features,
            multiclass_output_as_struct=multiclass_output_as_struct,
            post_predict_transformers=post_predict_transformers,
            filters=filters,
        )
        self._pred_columns_added = [
            point_predictor.pred_column,
            distribution_predictor.pred_column,
        ]

    @nw.narwhalify
    def train(self, df: FrameT, features: Optional[list[str]] = None) -> None:
        self._features = features or self.features

        df = apply_filters(df=df, filters=self.filters)
        self.point_predictor.train(df=df, features=features)
        df = nw.from_native(self.point_predictor.predict(df))
        self.distribution_predictor.train(df, features)

    @nw.narwhalify
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:
        if self.point_predictor.pred_column not in df.columns:
            df = nw.from_native(self.point_predictor.predict(df))
        df = self.distribution_predictor.predict(df)
        for post_predict_transformer in self.post_predict_transformers:
            df = nw.from_native(post_predict_transformer.transform(df))
        return df
