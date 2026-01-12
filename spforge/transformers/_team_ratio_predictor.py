from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
from narwhals.typing import IntoFrameT
from sklearn.base import is_regressor

from spforge.transformers._base import PredictorTransformer


class RatioEstimatorTransformer(PredictorTransformer):
    def __init__(
        self,
        features: list[str],
        estimator: Any,
        granularity: list[str],
        ratio_column_name: str = "__ratio",
        prediction_column_name: str | None = None,
        granularity_prediction_column_name: str | None = None,
        predict_row: bool = True,
        predict_granularity: bool = True,
    ):
        self.features = features
        self.estimator = estimator
        self.granularity = granularity

        self.ratio_column_name = ratio_column_name
        self.prediction_column_name = prediction_column_name
        self.granularity_prediction_column_name = granularity_prediction_column_name

        self.predict_row = predict_row
        self.predict_granularity = predict_granularity

        if not self.features:
            raise ValueError("features must be non-empty")
        if not self.granularity:
            raise ValueError("granularity must be non-empty")
        if not is_regressor(self.estimator):
            raise TypeError(f"estimator must be a regressor, got {type(self.estimator).__name__}")

        if not self.predict_row and not self.prediction_column_name:
            raise ValueError("prediction_column_name must be provided when predict_row=False")
        if not self.predict_granularity and not self.granularity_prediction_column_name:
            raise ValueError(
                "granularity_prediction_column_name must be provided when predict_granularity=False"
            )

        self._numeric_features: list[str] = []
        self._non_numeric_features: list[str] = []
        self._is_fitted = False

    def _infer_feature_types(self, X: IntoFrameT):
        df_feats = X.select(self.features)
        numeric = list(df_feats.select(nw.selectors.numeric()).columns)
        non_numeric = [c for c in self.features if c not in set(numeric)]
        return numeric, non_numeric

    def _group_agg_exprs(self):
        exprs = []
        for c in self._numeric_features:
            exprs.append(nw.col(c).mean().alias(c))
        for c in self._non_numeric_features:
            exprs.append(nw.col(c).first().alias(c))
        return exprs

    def _extra_output_cols(self) -> list[str]:
        cols = [self.ratio_column_name]
        if self.prediction_column_name:
            cols.append(self.prediction_column_name)
        if self.granularity_prediction_column_name:
            cols.append(self.granularity_prediction_column_name)
        return cols

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any):
        self._numeric_features, self._non_numeric_features = self._infer_feature_types(X)

        if self.predict_row or self.predict_granularity:
            y_values = y.to_list() if hasattr(y, "to_list") else y

            df = X.with_columns(
                nw.new_series(
                    name="__target__",
                    values=y_values,
                    backend=nw.get_native_namespace(X),
                )
            )

            df_grp = df.group_by(self.granularity).agg(
                self._group_agg_exprs() + [nw.col("__target__").mean().alias("__target__")]
            )

            self.estimator.fit(
                df_grp.select(self.features),
                df_grp["__target__"].to_numpy(),
            )

        self._is_fitted = True
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        native = X.to_native()
        native_index = native.index if isinstance(native, pd.DataFrame) else None

        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        df = X

        if self.predict_row:
            row_pred = self.estimator.predict(df.select(self.features))
            df = df.with_columns(
                nw.new_series("__row_pred__", row_pred, backend=nw.get_native_namespace(df))
            )
        else:
            if self.prediction_column_name not in set(df.columns):
                raise ValueError(
                    f"Expected existing column {self.prediction_column_name!r} in X when predict_row=False"
                )
            df = df.with_columns(nw.col(self.prediction_column_name).alias("__row_pred__"))

        if self.predict_granularity:
            df_grp_feat = df.group_by(self.granularity).agg(self._group_agg_exprs())
            grp_pred = self.estimator.predict(df_grp_feat.select(self.features))

            df_grp_pred = df_grp_feat.with_columns(
                nw.new_series(
                    "__grp_pred__", grp_pred, backend=nw.get_native_namespace(df_grp_feat)
                )
            ).select(self.granularity + ["__grp_pred__"])

            df = df.join(df_grp_pred, on=self.granularity, how="left")
        else:
            if self.granularity_prediction_column_name not in set(df.columns):
                raise ValueError(
                    f"Expected existing column {self.granularity_prediction_column_name!r} in X when predict_granularity=False"
                )
            df = df.with_columns(
                nw.col(self.granularity_prediction_column_name).alias("__grp_pred__")
            )

        df = df.with_columns(
            (nw.col("__row_pred__") / nw.col("__grp_pred__")).alias(self.ratio_column_name)
        )

        if self.prediction_column_name and self.predict_row:
            df = df.with_columns(nw.col("__row_pred__").alias(self.prediction_column_name))

        if self.granularity_prediction_column_name and self.predict_granularity:
            df = df.with_columns(
                nw.col("__grp_pred__").alias(self.granularity_prediction_column_name)
            )

        out = df.select(self.get_feature_names_out())

        # IMPORTANT: ensure pandas index matches input index for sklearn's pandas hstack
        if native_index is not None:
            out_native = out.to_native()
            out_native.index = native_index
            return out_native

        return out

    def set_output(self, *, transform=None):
        return self

    @property
    def context_features(self) -> list[str]:
        """Returns granularity columns needed for grouping."""
        return list(self.granularity)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        out = []
        for c in self._extra_output_cols():
            if c not in set(out):
                out.append(c)
        return list(set(out))

    @property
    def context_features(self) -> list[str]:
        """Returns granularity columns needed for grouping."""
        return list(self.granularity)
