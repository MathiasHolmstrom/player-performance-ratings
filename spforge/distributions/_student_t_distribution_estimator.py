from collections.abc import Sequence

import narwhals.stable.v2 as nw
import numpy as np
import polars as pl
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator


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
        self.support_cap_min = (
            int(support_cap_min) if support_cap_min is not None else self.min_value
        )

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
        return max(0, min(idx, len(edges) - 2))

    @nw.narwhalify
    def fit(self, X, y: list[int] | np.ndarray, sample_weight: np.ndarray | None = None):
        if isinstance(X.to_native() if hasattr(X, "to_native") else X, np.ndarray):
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
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
        mu = df[self.point_estimate_pred_column].to_numpy()

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
        for arr, bins in zip(cond_arrays, self.sigma_bins_per_column, strict=False):
            edges = self._compute_edges(arr, bins)
            if edges.size < 3:
                self.sigma_global = sigma_global
                return
            edges_list.append(edges)
            bin_counts.append(edges.size - 1)

        cell_to_resids: dict[tuple[int, ...], list[float]] = {}
        n = resid.shape[0]
        for i in range(n):
            key = tuple(
                self._bin_index(edges_list[d], cond_arrays[d][i]) for d in range(len(edges_list))
            )
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
            key = tuple(
                self._bin_index(self._cond_bin_edges[d], float(cond_vals[d]))
                for d in range(len(self._cond_bin_edges))
            )
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
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        if self._classes is None:
            raise ValueError(
                "StudentTDistributionEstimator has not been fitted yet. Call fit() first."
            )

        for c in self.sigma_conditioning_columns:
            if c not in X.columns:
                raise ValueError(f"conditioning column '{c}' not found in X.columns: {X.columns}")
        if self.support_cap_column is not None and self.support_cap_column not in X.columns:
            raise ValueError(
                f"support_cap_column '{self.support_cap_column}' not found in X.columns: {X.columns}"
            )

        classes = self._classes
        lower_bounds = classes - 0.5
        upper_bounds = classes + 0.5
        df_param = float(self.df)

        cond_lists = [X[c].to_list() for c in self.sigma_conditioning_columns]
        cap_list = (
            X[self.support_cap_column].to_list() if self.support_cap_column is not None else None
        )

        probabilities = []
        for i in range(len(cond_lists[0])):
            cond_vals = [cond_lists[d][i] for d in range(len(cond_lists))]
            sigma = self._sigma_for_row(cond_vals)

            mu_val = X[self.point_estimate_pred_column].to_list()[i]
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
            raise TypeError(
                "X must be a DataFrame (pandas, polars, or Narwhals), not a numpy array"
            )
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1) + self.min_value
