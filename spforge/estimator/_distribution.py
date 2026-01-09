import copy
from typing import Sequence

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
from narwhals.typing import IntoFrameT
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom, norm, t as student_t, poisson
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from spforge import ColumnNames
from spforge.feature_generator._rolling_window import RollingWindowTransformer
from spforge.scorer import Filter, apply_filters


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


class NegativeBinomialEstimator(BaseEstimator):

    def __init__(
        self,
        point_estimate_pred_column: str,
        max_value: int,
        min_value: int = 0,
        predicted_r_iterations: int = 6,
        predict_granularity: list[str] | None = None,
        r_specific_granularity: list[str] | None = None,
        r_rolling_mean_window: int = 40,
        predicted_r_weight: float = 1.0,
        column_names: ColumnNames | None = None,
    ):
        self.point_estimate_pred_column = point_estimate_pred_column
        self.max_value = max_value
        self.min_value = min_value
        self.predicted_r_iterations = predicted_r_iterations
        self.predict_granularity = predict_granularity
        self.r_specific_granularity = r_specific_granularity
        self.r_rolling_mean_window = r_rolling_mean_window
        self.predicted_r_weight = predicted_r_weight
        self.column_names = column_names

        if self.r_specific_granularity and not self.column_names:
            raise ValueError("r_specific_granularity is set but column names is not provided.")
        if self.r_specific_granularity:
            self._rolling_mean: RollingWindowTransformer = RollingWindowTransformer(
                aggregation="mean",
                features=[point_estimate_pred_column],
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
                features=[point_estimate_pred_column],
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

        self._multipliers = list(range(2, 2 + predicted_r_iterations * 2, 2))
        self._target_scaler = StandardScaler()
        self._best_multiplier = None
        self._mean_r = None
        self._scores = []
        self._r_estimates = {}
        self.target = "__target"
        self._historical_game_ids = []
        self.classes_ = np.arange(min_value, max_value + 1)

        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y, sample_weight: np.ndarray | None = None):
        """
        Fit the negative binomial distribution predictor.

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :param y: Target Series
        :param sample_weight: Optional sample weights (unused for distribution predictors)
        """

        y_values = (
            y.to_list() if hasattr(y, "to_list") else (y.values if hasattr(y, "values") else y)
        )
        df = X.with_columns(
            nw.new_series(name=self.target, values=y_values, backend=nw.get_native_namespace(X))
        )

        if self.point_estimate_pred_column not in df.columns:
            raise ValueError(
                f"point_estimate_pred_column '{self.point_estimate_pred_column}' not found in X.columns: {df.columns}"
            )

        self._train_internal(df)

        return self

    @nw.narwhalify
    def _train_internal(self, df: IntoFrameT) -> None:
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
            x0=np.array([1.0]),
            args=(
                positive_predicted_rows[self.target].to_numpy(),
                positive_predicted_rows[self.point_estimate_pred_column].to_numpy(),
            ),
            method="L-BFGS-B",
            bounds=[(0.01, None)],
        )
        self._mean_r = float(result.x[0])
        if self.r_specific_granularity:
            gran_grp = self._grp_to_r_granularity(df, is_train=True)
            column_names_lag = copy.deepcopy(self.column_names)
            if self.column_names.player_id not in self.r_specific_granularity:
                column_names_lag.player_id = None
            gran_grp = nw.from_native(
                self._rolling_mean.fit_transform(gran_grp, column_names=column_names_lag)
            )
            gran_grp = nw.from_native(
                self._rolling_var.fit_transform(gran_grp, column_names=column_names_lag)
            )

            def define_quantile_bins(
                df: IntoFrameT, column: str, quantiles: list[float]
            ) -> list[float]:
                return [df[column].quantile(q, interpolation="nearest") for q in quantiles]

            mu_bins = define_quantile_bins(
                gran_grp, self._rolling_mean.features_out[0], [0.2, 0.4, 0.6, 0.8]
            )

            for mu_idx, mu_bin in enumerate(mu_bins):
                mu_group = gran_grp.filter(nw.col(self._rolling_mean.features_out[0]) >= mu_bin)
                if mu_idx < len(mu_bins) - 1:
                    next_mu_bin = mu_bins[mu_idx + 1]
                    mu_group = mu_group.filter(
                        nw.col(self._rolling_mean.features_out[0]) < next_mu_bin
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
                            nw.col(self._rolling_var.features_out[0]) < next_var_bin
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

            self._historical_game_ids = gran_grp[self.column_names.match_id].unique().to_list()

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        """
        Predict probability distributions.

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :return: Array of probability distributions (n_samples, n_classes)
        """
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )

        if self.point_estimate_pred_column not in X.columns:
            raise ValueError(
                f"point_estimate_pred_column '{self.point_estimate_pred_column}' not found in X.columns: {X.columns}"
            )

        if self._mean_r is None:
            raise ValueError("NegativeBinomialPredictor has not been fitted yet. Call fit() first.")

        df = X

        prob_df = self._predict_internal(df)

        prob_col = "__target_probabilities"
        prob_series = prob_df[prob_col].to_list()
        probabilities = np.array([np.array(p) for p in prob_series])

        return probabilities

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        """
        Predict point estimates (mode of distribution).

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :return: Array of predicted values
        """
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1) + self.min_value

    @nw.narwhalify
    def _predict_internal(self, df: IntoFrameT) -> IntoFrameT:
        """Internal prediction logic using Narwhals DataFrames"""
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
                    self._rolling_mean.fit_transform(historical_gran_group)
                )
                historical_gran_group = nw.from_native(
                    self._rolling_var.fit_transform(historical_gran_group)
                )
            if len(future_gran_group) > 0:
                future_gran_group = nw.from_native(self._rolling_mean.future_transform(future_gran_group))
                future_gran_group = nw.from_native(self._rolling_var.future_transform(future_gran_group))

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
            prob_col = "__target_probabilities"
            df = self._add_probabilities(df=pred_df).select([*input_cols, prob_col])
            assert len(df) == pre_join_df_row_count
            return df

        else:
            df = self._add_predicted_r(df)
            prob_col = "__target_probabilities"
            return self._add_probabilities(df=df).select([*input_cols, prob_col])

    def _grp_to_r_granularity(self, df: IntoFrameT, is_train: bool) -> IntoFrameT:

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

    def _add_predicted_r(self, df: IntoFrameT) -> IntoFrameT:

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

    def _add_probabilities(self, df: IntoFrameT) -> IntoFrameT:

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
        EPS = 1e-12
        POISSON_EPS = 0.01
        for row in pred_grp.iter_rows(named=True):
            point_range = np.arange(self.min_value, self.max_value + 1, dtype=np.int64)

            r = float(row["__predicted_r"])
            mu = float(row[self.point_estimate_pred_column])

            # hard safety
            if not np.isfinite(mu) or mu < 0:
                mu = 0.0
            if not np.isfinite(r) or r <= 0:
                r = 0.01

            use_poisson = (mu * mu / r) < (POISSON_EPS * max(mu, EPS))

            if use_poisson:
                logp = poisson.logpmf(point_range, mu)
            else:
                p = r / (r + mu)
                p = min(max(p, EPS), 1.0 - EPS)
                logp = nbinom.logpmf(point_range, r, p)

            if not np.isfinite(logp).any():
                prob = np.zeros_like(point_range, dtype=float)
                prob[0] = 1.0
            else:
                z = logsumexp(logp[np.isfinite(logp)])
                prob = np.exp(logp - z)
                prob[~np.isfinite(prob)] = 0.0

                s = prob.sum()
                if s <= 0 or not np.isfinite(s):
                    prob = np.zeros_like(point_range, dtype=float)
                    prob[0] = 1.0
                else:
                    prob /= s

            all_outcome_probs.append(prob)

        prob_col = "__target_probabilities"
        pred_grp = pred_grp.with_columns(
            nw.new_series(
                name=prob_col,
                values=all_outcome_probs,
                backend=nw.get_native_namespace(df),
            )
        )

        if self.predict_granularity:
            return df.join(
                pred_grp.select([*self.predict_granularity, prob_col]),
                on=self.predict_granularity,
                how="left",
            )
        return pred_grp


class NormalDistributionPredictor(BaseEstimator):

    def __init__(
        self,
        point_estimate_pred_column: str,
        max_value: int,
        min_value: int,
        target: str,
        sigma: float = 10.0,
    ):
        self.point_estimate_pred_column = point_estimate_pred_column
        self.max_value = max_value
        self.min_value = min_value
        self.target = target
        self.sigma = sigma
        self._classes = None
        self.classes_ = np.arange(min_value, max_value + 1)
        super().__init__()

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y, sample_weight: np.ndarray | None = None):
        """
        Fit the normal distribution predictor.

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :param y: Target Series (unused, kept for sklearn interface)
        :param sample_weight: Optional sample weights (unused)
        """
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        if self.point_estimate_pred_column not in X.columns:
            raise ValueError(
                f"point_estimate_pred_column '{self.point_estimate_pred_column}' not found in X.columns: {X.columns}"
            )
        self._classes = np.arange(self.min_value, self.max_value + 1)
        return self

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        """
        Predict probability distributions.

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :return: Array of probability distributions (n_samples, n_classes)
        """
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        if self.point_estimate_pred_column not in X.columns:
            raise ValueError(
                f"point_estimate_pred_column '{self.point_estimate_pred_column}' not found in X.columns: {X.columns}"
            )
        if self._classes is None:
            raise ValueError(
                "NormalDistributionPredictor has not been fitted yet. Call fit() first."
            )

        lower_bounds = self._classes - 0.5
        upper_bounds = self._classes + 0.5

        mu_values = X[self.point_estimate_pred_column].to_list()
        probabilities = []
        for mu in mu_values:
            cdf_upper = norm.cdf(upper_bounds, loc=mu, scale=self.sigma)
            cdf_lower = norm.cdf(lower_bounds, loc=mu, scale=self.sigma)
            probs = cdf_upper - cdf_lower
            probabilities.append(probs)

        return np.array(probabilities)

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> np.ndarray:
        """
        Predict point estimates (mode of distribution).

        :param X: DataFrame (any DataFrame type - pandas, polars, or Narwhals). Cannot be numpy array.
        :return: Array of predicted values
        """
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1) + self.min_value

class StudentTDistributionEstimator(BaseEstimator):
    """
    Discretized Student-t over integer outcomes [min_value, max_value].

    Improvements:
      - Heteroskedastic sigma: sigma can be learned as a function of one or more conditioning columns
        using quantile bins, and a robust MAD-based scale per bin.
      - Optional per-row support cap: cap the maximum feasible outcome based on a column
        (e.g. yardline_100), and renormalize.

    Params:
      - df: fixed tail parameter (not learned).
      - sigma: if provided, fixed global sigma (disables heteroskedastic fitting).
      - sigma_conditioning_columns: columns to condition sigma on (default: (point_estimate_pred_column,))
      - sigma_bins_per_column: number of quantile bins per conditioning column (same length as sigma_conditioning_columns).
      - support_cap_column: column name used to cap max_value per row (e.g., 'yardline_100'); if None, disabled.
      - support_cap_offset: integer offset applied to cap (e.g., -1).
    """

    def __init__(
            self,
            point_estimate_pred_column: str,
            max_value: int,
            min_value: int,
            target: str,
            df: float = 5.0,
            sigma: float | None = None,
            min_sigma: float = 1.0,
            min_train_rows: int = 50,
            sigma_fallback_if_no_data: float = 10.0,
            sigma_bins: int = 8,
            min_bin_rows: int = 25,
            sigma_conditioning_columns: tuple[str, ...] | None = None,
            sigma_bins_per_column: tuple[int, ...] | None = None,
            support_cap_column: str | None = None,
            support_cap_offset: int = 0,
            support_cap_min: int | None = None,
    ):
        self.point_estimate_pred_column = point_estimate_pred_column
        self.max_value = int(max_value)
        self.min_value = int(min_value)
        self.target = target

        self.df = float(df)
        if self.df <= 2.0:
            raise ValueError(f"df must be > 2.0 for finite variance; got df={self.df}")

        self._sigma_init: float | None = float(sigma) if sigma is not None else None
        self.sigma_global: float | None = None

        self.min_sigma = float(min_sigma)
        self.min_train_rows = int(min_train_rows)
        self.sigma_fallback_if_no_data = float(sigma_fallback_if_no_data)

        self.sigma_bins = int(sigma_bins)
        self.min_bin_rows = int(min_bin_rows)

        if sigma_conditioning_columns is None:
            sigma_conditioning_columns = (self.point_estimate_pred_column,)
        self.sigma_conditioning_columns = tuple(sigma_conditioning_columns)

        if sigma_bins_per_column is None:
            sigma_bins_per_column = tuple([self.sigma_bins] * len(self.sigma_conditioning_columns))
        self.sigma_bins_per_column = tuple(int(b) for b in sigma_bins_per_column)

        if len(self.sigma_bins_per_column) != len(self.sigma_conditioning_columns):
            raise ValueError(
                "sigma_bins_per_column must have same length as sigma_conditioning_columns. "
                f"Got {len(self.sigma_bins_per_column)} vs {len(self.sigma_conditioning_columns)}"
            )

        self.support_cap_column = support_cap_column
        self.support_cap_offset = int(support_cap_offset)
        self.support_cap_min = int(support_cap_min) if support_cap_min is not None else self.min_value

        self._classes: np.ndarray | None = None

        self._cond_bin_edges: list[np.ndarray] | None = None
        self._sigma_by_cell: dict[tuple[int, ...], float] | None = None

        self.classes_ = np.arange(self.min_value, self.max_value + 1)
        super().__init__()

    def _reset_state(self) -> None:
        self._classes = None
        self._cond_bin_edges = None
        self._sigma_by_cell = None
        self.sigma_global = self._sigma_init

    @staticmethod
    def _mad_scale(x: np.ndarray) -> float:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return float(1.4826 * mad)

    def _fit_sigma_global(self, resid: np.ndarray) -> float:
        sigma_hat = self._mad_scale(resid)
        if not np.isfinite(sigma_hat) or sigma_hat <= 0:
            sigma_hat = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
        return max(self.min_sigma, float(sigma_hat))

    def _compute_edges(self, x: np.ndarray, bins: int) -> np.ndarray:
        qs = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(x, qs)
        edges = np.unique(edges)
        return edges

    def _bin_index(self, edges: np.ndarray, val: float) -> int:
        idx = int(np.searchsorted(edges, val, side="right") - 1)
        return max(0, min(idx, len(edges) - 2))  # bin count = len(edges)-1

    @nw.narwhalify
    def fit(self, X, y: list[int] | np.ndarray, sample_weight: np.ndarray | None = None):
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        for c in self.sigma_conditioning_columns:
            if c not in X.columns:
                raise ValueError(f"conditioning column '{c}' not found in X.columns: {X.columns}")

        df = nw.from_native(pl.DataFrame(X))
        df = df.with_columns(
            nw.new_series(name=self.target, values=y, backend=nw.get_native_namespace(df))
        )
        self._train_internal(df)
        return self

    @nw.narwhalify
    def _train_internal(self, df: nw.DataFrame) -> None:
        self._reset_state()
        self._classes = np.arange(self.min_value, self.max_value + 1)

        if self.sigma_global is not None:
            self.sigma_global = max(self.min_sigma, float(self.sigma_global))
            return

        if self.target not in df.columns:
            raise ValueError(f"Expected '{self.target}' column to exist for sigma fitting.")

        y = df[self.target].to_numpy()
        cond_arrays = [df[c].to_numpy() for c in self.sigma_conditioning_columns]
        mu = df[self.point_estimate_pred_column].to_numpy()  # used for residual definition

        mask = np.isfinite(y) & np.isfinite(mu)
        for arr in cond_arrays:
            mask &= np.isfinite(arr)

        if mask.sum() < self.min_train_rows:
            self.sigma_global = max(self.min_sigma, self.sigma_fallback_if_no_data)
            return

        y = y[mask].astype(float)
        mu = mu[mask].astype(float)
        cond_arrays = [arr[mask].astype(float) for arr in cond_arrays]
        resid = y - mu

        sigma_global = self._fit_sigma_global(resid)

        edges_list: list[np.ndarray] = []
        bin_counts: list[int] = []
        for arr, bins in zip(cond_arrays, self.sigma_bins_per_column):
            edges = self._compute_edges(arr, bins)
            if edges.size < 3:
                self.sigma_global = sigma_global
                return
            edges_list.append(edges)
            bin_counts.append(edges.size - 1)

        cell_to_resids: dict[tuple[int, ...], list[float]] = {}
        n = resid.shape[0]
        for i in range(n):
            key = tuple(self._bin_index(edges_list[d], cond_arrays[d][i]) for d in range(len(edges_list)))
            cell_to_resids.setdefault(key, []).append(float(resid[i]))

        sigma_by_cell: dict[tuple[int, ...], float] = {}
        for key, rlist in cell_to_resids.items():
            if len(rlist) < self.min_bin_rows:
                sigma_by_cell[key] = sigma_global
            else:
                sigma_by_cell[key] = self._fit_sigma_global(np.asarray(rlist, dtype=float))

        self._cond_bin_edges = edges_list
        self._sigma_by_cell = sigma_by_cell
        self.sigma_global = None

    def _sigma_for_row(self, cond_vals: Sequence[float]) -> float:
        if self._cond_bin_edges is not None and self._sigma_by_cell is not None:
            key = tuple(self._bin_index(self._cond_bin_edges[d], float(cond_vals[d]))
                        for d in range(len(self._cond_bin_edges)))
            sigma = self._sigma_by_cell.get(key)
            if sigma is None:
                return max(self.min_sigma, float(self.sigma_fallback_if_no_data))
            return max(self.min_sigma, float(sigma))

        if self.sigma_global is not None:
            return max(self.min_sigma, float(self.sigma_global))
        return max(self.min_sigma, float(self.sigma_fallback_if_no_data))

    def _row_max_value(self, row_dict_like) -> int:
        if self.support_cap_column is None:
            return self.max_value

        cap_val = row_dict_like[self.support_cap_column]
        if cap_val is None or not np.isfinite(cap_val):
            return self.max_value

        cap_int = int(np.floor(float(cap_val))) + self.support_cap_offset
        cap_int = max(self.support_cap_min, cap_int)
        return min(self.max_value, cap_int)

    @nw.narwhalify
    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        if self._classes is None:
            raise ValueError("StudentTDistributionEstimator has not been fitted yet. Call fit() first.")

        for c in self.sigma_conditioning_columns:
            if c not in X.columns:
                raise ValueError(f"conditioning column '{c}' not found in X.columns: {X.columns}")
        if self.support_cap_column is not None and self.support_cap_column not in X.columns:
            raise ValueError(
                f"support_cap_column '{self.support_cap_column}' not found in X.columns: {X.columns}")

        classes = self._classes
        lower_bounds = classes - 0.5
        upper_bounds = classes + 0.5
        df_param = float(self.df)

        # pull needed columns as python lists (narwhals friendly)
        cond_lists = [X[c].to_list() for c in self.sigma_conditioning_columns]
        cap_list = X[self.support_cap_column].to_list() if self.support_cap_column is not None else None

        probabilities = []
        for i in range(len(cond_lists[0])):
            cond_vals = [cond_lists[d][i] for d in range(len(cond_lists))]
            sigma = self._sigma_for_row(cond_vals)

            mu_val = X[self.point_estimate_pred_column].to_list()[i]  # safe but a bit repetitive
            cdf_upper = student_t.cdf(upper_bounds, df=df_param, loc=mu_val, scale=sigma)
            cdf_lower = student_t.cdf(lower_bounds, df=df_param, loc=mu_val, scale=sigma)
            probs = np.clip(cdf_upper - cdf_lower, 0.0, 1.0)

            if cap_list is not None:
                cap_val = cap_list[i]
                if cap_val is not None and np.isfinite(cap_val):
                    row_cap = int(np.floor(float(cap_val))) + self.support_cap_offset
                    row_cap = max(self.support_cap_min, row_cap)
                    row_cap = min(self.max_value, row_cap)
                    probs = probs.copy()
                    probs[classes > row_cap] = 0.0

            s = probs.sum()
            if s > 0:
                probs = probs / s
            probabilities.append(probs)

        return np.array(probabilities)

    @nw.narwhalify
    def predict(self, X) -> np.ndarray:
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array")
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1) + self.min_value
