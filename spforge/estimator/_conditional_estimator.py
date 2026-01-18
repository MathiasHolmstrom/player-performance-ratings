from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from sklearn.base import BaseEstimator, ClassifierMixin


class ConditionalEstimator(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        gate_estimator: Any,
        gate_distance_col: str,
        outcome_0_value: str | int,
        outcome_1_value: str | int,
        outcome_0_estimator: Any,
        outcome_1_estimator: Any,
        gate_distance_col_is_feature: bool = True,
    ):
        self.gate_estimator = gate_estimator
        self.gate_distance_col = gate_distance_col
        self.outcome_0_estimator = outcome_0_estimator
        self.outcome_1_estimator = outcome_1_estimator
        self.outcome_0_value = outcome_0_value
        self.outcome_1_value = outcome_1_value
        self.gate_distance_col_is_feature = gate_distance_col_is_feature

    @nw.narwhalify
    def fit(
        self, X: IntoFrameT, y: list[int] | np.ndarray, sample_weight: np.ndarray | None = None
    ):
        self.fitted_feats = (
            X.columns
            if self.gate_distance_col_is_feature
            else X.drop(self.gate_distance_col).columns
        )

        df = X.with_columns(
            nw.new_series(name="__target", values=y, backend=nw.get_native_namespace(X))
        )

        df = df.with_columns(
            nw.when(nw.col(self.gate_distance_col) > nw.col("__target"))
            .then(nw.lit(1))
            .otherwise(nw.lit(0))
            .alias("__gate_target")
        )

        y_gate = df["__gate_target"].to_numpy()
        self.gate_estimator.fit(df.select(self.fitted_feats).to_pandas(), y_gate)

        y_array = y if isinstance(y, np.ndarray) else np.array(y)
        self.classes_ = sorted(list(set(y_array)))

        df = df.with_columns(
            (nw.col("__target") - nw.col(self.gate_distance_col)).alias("__diff_gate")
        )

        df0_rows = df.filter(nw.col("__gate_target") == self.outcome_0_value)
        df1_rows = df.filter(nw.col("__gate_target") == self.outcome_1_value)

        X0 = df0_rows.select(self.fitted_feats).to_pandas()
        X1 = df1_rows.select(self.fitted_feats).to_pandas()

        y0 = df0_rows["__diff_gate"].to_numpy()
        y1 = df1_rows["__diff_gate"].to_numpy()

        self.outcome_0_estimator.fit(X0, y0)
        self.outcome_1_estimator.fit(X1, y1)

        self.outcome_0_classes_ = np.asarray(self.outcome_0_estimator.classes_, dtype=int)
        self.outcome_1_classes_ = np.asarray(self.outcome_1_estimator.classes_, dtype=int)

        return self

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        X_feats_pd = X.select(self.fitted_feats).to_pandas()
        gate_distance = X[self.gate_distance_col].to_numpy()

        n = len(gate_distance)
        classes = np.asarray(self.classes_, dtype=int)
        C = len(classes)

        gate_proba = self.gate_estimator.predict_proba(X_feats_pd)
        gate_class_to_idx = {int(c): i for i, c in enumerate(self.gate_estimator.classes_)}

        p0 = gate_proba[:, gate_class_to_idx[int(self.outcome_0_value)]]
        p1 = gate_proba[:, gate_class_to_idx[int(self.outcome_1_value)]]

        proba_diff0 = self.outcome_0_estimator.predict_proba(X_feats_pd)
        proba_diff1 = self.outcome_1_estimator.predict_proba(X_feats_pd)

        diff0_to_idx = {int(d): j for j, d in enumerate(self.outcome_0_classes_)}
        diff1_to_idx = {int(d): j for j, d in enumerate(self.outcome_1_classes_)}

        def map_diff_to_y(proba_diff: np.ndarray, diff_to_idx: dict[int, int]) -> np.ndarray:
            out = np.zeros((n, C), dtype=float)
            class_to_idx = {int(c): j for j, c in enumerate(classes)}
            min_class = int(classes[0])
            max_class = int(classes[-1])

            for i in range(n):
                gd = int(gate_distance[i])
                for diff, k in diff_to_idx.items():
                    target = gd + diff
                    prob = proba_diff[i, k]

                    if target in class_to_idx:
                        j = class_to_idx[target]
                        out[i, j] += prob
                    elif target < min_class:
                        out[i, 0] += prob
                    else:
                        out[i, C - 1] += prob

            return out

        y_proba0 = map_diff_to_y(proba_diff0, diff0_to_idx)
        y_proba1 = map_diff_to_y(proba_diff1, diff1_to_idx)

        out = (p0[:, None] * y_proba0) + (p1[:, None] * y_proba1)

        row_sums = out.sum(axis=1, keepdims=True)
        out = np.where(row_sums > 0, out / row_sums, out)

        return out

    @nw.narwhalify
    def predict(self, X: Any) -> np.ndarray:
        """Predict most likely global expert label."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array(self.classes_)[idx]
