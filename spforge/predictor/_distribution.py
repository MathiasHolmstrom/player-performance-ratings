import copy
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
from spforge.scorer import apply_filters, Filter

from spforge.predictor import (
    BasePredictor,
)
from spforge.scorer import BaseScorer
from spforge.transformers import RollingWindowTransformer


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
        predicted_r_weight: float = 1.0,
        column_names: Optional[ColumnNames] = None,
    ):
        self.point_estimate_pred_column = point_estimate_pred_column
        pred_column = pred_column or f"{target}_probabilities"
        self.multiclass_output_as_struct = multiclass_output_as_struct
        self.r_specific_granularity = r_specific_granularity
        self.predicted_r_weight = predicted_r_weight
        self.r_rolling_mean_window = r_rolling_mean_window
        self.column_names = column_names
        self.predict_granularity = predict_granularity
        if self.r_specific_granularity:
            self._rolling_mean: RollingWindowTransformer = RollingWindowTransformer(
                aggregation="mean",
                features=[self.point_estimate_pred_column],
                window=self.r_rolling_mean_window,
                granularity=self.r_specific_granularity,
                are_estimator_features=False,
                update_column=self.column_names.update_match_id,
                min_periods=5,
                unique_constraint=[
                    self.column_names.match_id,
                    *self.r_specific_granularity,
                ],
            )
            self._rolling_var: RollingWindowTransformer = RollingWindowTransformer(
                aggregation="var",
                features=[self.point_estimate_pred_column],
                window=self.r_rolling_mean_window,
                granularity=self.r_specific_granularity,
                are_estimator_features=False,
                update_column=self.column_names.update_match_id,
                min_periods=5,
                unique_constraint=[
                    self.column_names.match_id,
                    *self.r_specific_granularity,
                ],
            )
        else:
            self._rolling_mean = None
            self._rolling_var = None

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
        self._r_estimates = {}

        self._historical_game_ids = []

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
            gran_grp = self._grp_to_r_granularity(df, is_train=True)
            column_names_lag = copy.deepcopy(self.column_names)
            if self.column_names.player_id not in self.r_specific_granularity:
                column_names_lag.player_id = None
            gran_grp = nw.from_native(
                self._rolling_mean.transform_historical(
                    gran_grp, column_names=column_names_lag
                )
            )
            gran_grp = nw.from_native(
                self._rolling_var.transform_historical(
                    gran_grp, column_names=column_names_lag
                )
            )

            def define_quantile_bins(
                df: FrameT, column: str, quantiles: list[float]
            ) -> list[float]:
                return [
                    df[column].quantile(q, interpolation="nearest") for q in quantiles
                ]

            mu_bins = define_quantile_bins(
                gran_grp, self._rolling_mean.features_out[0], [0.2, 0.4, 0.6, 0.8]
            )

            for mu_idx, mu_bin in enumerate(mu_bins):
                mu_group = gran_grp.filter(
                    (nw.col(self._rolling_mean.features_out[0]) >= mu_bin)
                )
                if mu_idx < len(mu_bins) - 1:
                    next_mu_bin = mu_bins[mu_idx + 1]
                    mu_group = mu_group.filter(
                        (nw.col(self._rolling_mean.features_out[0]) < next_mu_bin)
                    )

                var_bins = define_quantile_bins(
                    mu_group, self._rolling_var.features_out[0], [0.2, 0.4, 0.6, 0.8]
                )

                for var_idx, var_bin in enumerate(var_bins):
                    final_group = mu_group.filter(
                        nw.col(self._rolling_var.features_out[0]) >= var_bin
                    )
                    if var_idx < len(var_bins) - 1:
                        next_var_bin = var_bins[var_idx + 1]
                        final_group = final_group.filter(
                            (nw.col(self._rolling_var.features_out[0]) < next_var_bin)
                        )

                    result = minimize(
                        neg_binom_log_likelihood,
                        x0=np.array([1.0]),
                        args=(
                            final_group[self.target].to_numpy(),
                            final_group[self.point_estimate_pred_column].to_numpy(),
                        ),
                        method="L-BFGS-B",
                        bounds=[(0.01, None)],
                    )
                    self._r_estimates[(mu_bin, var_bin)] = float(result.x[0])

            self._historical_game_ids = (
                gran_grp[self.column_names.match_id].unique().to_list()
            )

    @nw.narwhalify
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:
        input_cols = df.columns
        if self.r_specific_granularity:
            gran_grp = self._grp_to_r_granularity(df, is_train=False)
            historical_gran_group = gran_grp.filter(
                nw.col(self.column_names.match_id).is_in(self._historical_game_ids)
            )
            future_gran_group = gran_grp.filter(
                ~nw.col(self.column_names.match_id).is_in(self._historical_game_ids)
            )
            if len(historical_gran_group) > 0:
                historical_gran_group = nw.from_native(
                    self._rolling_mean.transform_historical(historical_gran_group)
                )
                historical_gran_group = nw.from_native(
                    self._rolling_var.transform_historical(historical_gran_group)
                )
            if len(future_gran_group) > 0:
                future_gran_group = nw.from_native(
                    self._rolling_mean.transform_future(future_gran_group)
                )
                future_gran_group = nw.from_native(
                    self._rolling_var.transform_future(future_gran_group)
                )

            if len(historical_gran_group) > 0 and len(future_gran_group) > 0:
                gran_grp = nw.concat([historical_gran_group, future_gran_group]).unique(
                    [self.column_names.update_match_id, *self.r_specific_granularity],
                    maintain_order=True,
                )
            elif len(historical_gran_group) == 0:
                gran_grp = future_gran_group
            else:
                gran_grp = historical_gran_group

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
                (
                    (
                        nw.col("__predicted_r") * self.predicted_r_weight
                        + nw.lit(self._mean_r) * (1 - self.predicted_r_weight)
                    )
                ).alias("__predicted_r")
            )

            assert len(pred_df) == pre_join_df_row_count
            df = self._add_probabilities(df=pred_df).select(
                [*input_cols, self.pred_column]
            )
            assert len(df) == pre_join_df_row_count
            return df

        else:
            df = self._add_predicted_r(df)
            return self._add_probabilities(df=df).select(
                [*input_cols, self.pred_column]
            )

    def _grp_to_r_granularity(self, df: FrameT, is_train: bool) -> FrameT:

        aggregation = (
            [
                nw.col([self.point_estimate_pred_column, self.target]).mean(),
                nw.col(self.column_names.start_date).median(),
            ]
            if is_train
            else [
                nw.col(self.point_estimate_pred_column).mean(),
                nw.col(self.column_names.start_date).median(),
            ]
        )

        if df.schema[self.column_names.start_date] not in (nw.Date, nw.Datetime):
            df = df.with_columns(nw.col(self.column_names.start_date).str.to_datetime())
        return (
            df.group_by(
                list(
                    set(
                        [
                            self.column_names.match_id,
                            *self.r_specific_granularity,
                            self.column_names.team_id,
                            self.column_names.update_match_id,
                        ]
                    )
                )
            )
            .agg(aggregation)
            .sort(self.column_names.start_date)
        )

    def _add_predicted_r(self, df: FrameT) -> FrameT:

        df = df.with_columns(nw.lit(self._mean_r).alias("__predicted_r"))
        for (mu_bin, var_bin), predicted_r in self._r_estimates.items():
            next_higher_mu_bins = [m[0] for m in self._r_estimates if m[0] > mu_bin]
            next_higher_var_bins = [
                m[1] for m in self._r_estimates if m[1] > var_bin and m[0] == mu_bin
            ]

            if len(next_higher_mu_bins) == 0:
                next_higher_mu_bin = 9999
            else:
                next_higher_mu_bin = min(next_higher_mu_bins)
            if len(next_higher_var_bins) == 0:
                next_higher_var_bin = 9999
            else:

                next_higher_var_bin = min(next_higher_var_bins)

            df = df.with_columns(
                nw.when(
                    (nw.col(self._rolling_mean.features_out[0]) < next_higher_mu_bin)
                    & (nw.col(self._rolling_var.features_out[0]) < next_higher_var_bin)
                    & (nw.col(self._rolling_mean.features_out[0]) >= mu_bin)
                    & (nw.col(self._rolling_var.features_out[0]) >= var_bin)
                )
                .then(nw.lit(predicted_r))
                .otherwise(nw.col("__predicted_r"))
                .alias("__predicted_r")
            )
        return df

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
