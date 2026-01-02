import polars as pl
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from typing import Optional, Any

import numpy as np
import pandas as pd
import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator

from spforge.transformers.base_transformer import BaseTransformer
from spforge.transformers import ConvertDataFrameToCategoricalTransformer

from spforge.scorer import Filter, apply_filters


class Pipeline(BaseEstimator):
    def __init__(
            self,
            estimator: Any,
            target: str,
            features: Optional[list[str]] = None,
            pred_column: Optional[str] = None,
            filters: list[Filter] | None = None,
            scale_features: bool = False,
            one_hot_encode_cat_features: bool = False,
            impute_missing_values: bool = False,
            drop_rows_where_target_is_nan: bool = False,
            min_target: int | float | None = None,
            max_target: int | float | None = None,
            categorical_features: list[str] | None = None,
            numeric_features: list[str] | None = None,
            remainder: str = "drop",
    ):
        self.estimator = estimator
        self._target = target
        self._features = features or []
        self._pred_column = pred_column or f"{target}_prediction"
        self.filters = filters or []
        self.scale_features = scale_features
        self.one_hot_encode_cat_features = one_hot_encode_cat_features
        self.impute_missing_values = impute_missing_values
        self.drop_rows_where_target_is_nan = drop_rows_where_target_is_nan
        self.min_target = min_target
        self.max_target = max_target
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.remainder = remainder
        self._sk: SkPipeline | None = None
        self._fitted_features: list[str] = []

    @property
    def target(self) -> str:
        return self._target

    @property
    def pred_column(self) -> str:
        return self._pred_column

    @property
    def features(self) -> list[str]:
        return self._features

    def _reject_numpy(self, X: Any):
        native = X.to_native() if hasattr(X, "to_native") else X
        if isinstance(native, np.ndarray):
            raise TypeError("X must be a DataFrame (pandas/polars/narwhals), not a numpy array")

    def _to_pandas(self, df: Any) -> pd.DataFrame:
        native = df.to_native() if hasattr(df, "to_native") else df
        if isinstance(native, pd.DataFrame):
            return native
        if isinstance(native, pl.DataFrame):
            return native.to_pandas()
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        return pd.DataFrame(native)

    def _infer_feature_types(self, df_pd: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
        if self.numeric_features is not None or self.categorical_features is not None:
            num = self.numeric_features or []
            cat = self.categorical_features or []
            return list(num), list(cat)

        num = []
        cat = []
        for c in features:
            s = df_pd[c]
            if pd.api.types.is_numeric_dtype(s):
                num.append(c)
            else:
                cat.append(c)
        return num, cat

    def _build_sklearn_pipeline(self, df_pd: pd.DataFrame, features: list[str]) -> SkPipeline:
        num_feats, cat_feats = self._infer_feature_types(df_pd, features)

        num_steps = []
        cat_steps = []

        if self.impute_missing_values:
            num_steps.append(("impute", SimpleImputer()))
            cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))

        if self.scale_features:
            num_steps.append(("scale", StandardScaler()))

        if self.one_hot_encode_cat_features:
            cat_steps.append(("ohe", OneHotEncoder(handle_unknown="ignore")))

        if num_feats:
            if num_steps:
                num_pipe = SkPipeline(steps=num_steps)
            else:
                num_pipe = "passthrough"
            num_tr = ("num", num_pipe, num_feats)
        else:
            num_tr = None

        if cat_feats:
            if cat_steps:
                cat_pipe = SkPipeline(steps=cat_steps)
            else:
                cat_pipe = "passthrough"
            cat_tr = ("cat", cat_pipe, cat_feats)
        else:
            cat_tr = None

        transformers = [t for t in [num_tr, cat_tr] if t is not None]
        pre = ColumnTransformer(transformers=transformers, remainder=self.remainder)

        return SkPipeline(steps=[("pre", pre), ("est", self.estimator)])

    def _narwhals_preprocess(self, df: Any) -> Any:
        if self.min_target is not None:
            df = df.with_columns(nw.col(self._target).clip(lower_bound=self.min_target))
        if self.max_target is not None:
            df = df.with_columns(nw.col(self._target).clip(upper_bound=self.max_target))

        df = apply_filters(df=df, filters=self.filters)
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)

        if self.drop_rows_where_target_is_nan:
            df = df.filter(~nw.col(self._target).is_nan())

        return df

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y=None, features: Optional[list[str]] = None,
            sample_weight: Optional[np.ndarray] = None):
        self._reject_numpy(X)

        original_native = X.to_native()
        is_sklearn_style = y is not None

        if is_sklearn_style:
            y_values = y.to_list() if hasattr(y, "to_list") else y
            df = X.with_columns(
                nw.new_series(
                    name=self._target,
                    values=y_values,
                    backend=nw.get_native_namespace(X),
                )
            )
            feats = features or list(X.columns)
        else:
            df = X
            if self._target not in df.columns:
                raise ValueError(f"Target {self._target} not in columns: {df.columns}")
            feats = features or self._features

        if not feats:
            raise ValueError("features not set. Pass features to fit() or constructor")

        df = self._narwhals_preprocess(df)
        df_pd = self._to_pandas(df)

        self._fitted_features = list(feats)
        self._sk = self._build_sklearn_pipeline(df_pd, self._fitted_features)

        X_fit = df_pd[self._fitted_features]
        y_fit = df_pd[self._target]

        if sample_weight is not None:
            self._sk.fit(X_fit, y_fit, est__sample_weight=sample_weight)
        else:
            self._sk.fit(X_fit, y_fit)

        if hasattr(self._sk[-1], "classes_"):
            self.classes_ = self._sk[-1].classes_

        return self

    @nw.narwhalify
    def predict(self, X: IntoFrameT, return_features: bool = False, **kwargs) -> IntoFrameT:
        self._reject_numpy(X)
        if self._sk is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        native_in = X.to_native()
        is_pandas_in = isinstance(native_in, pd.DataFrame)

        df = nw.from_native(X) if not hasattr(X, "to_native") else X
        df_pd = self._to_pandas(df)

        X_pred = df_pd[self._fitted_features]
        preds = self._sk.predict(X_pred)

        if is_pandas_in:
            return preds

        out = df.with_columns(
            nw.new_series(
                name=self._pred_column,
                values=preds.tolist() if isinstance(preds, np.ndarray) else list(preds),
                backend=nw.get_native_namespace(df),
            )
        )

        if return_features:
            return out

        input_cols = list(X.columns)
        cols = [c for c in input_cols if c in out.columns] + [self._pred_column]
        return out.select(cols)

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        self._reject_numpy(X)
        if self._sk is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        if not hasattr(self._sk[-1], "predict_proba"):
            raise AttributeError(f"{type(self._sk[-1]).__name__} does not support predict_proba")

        df = nw.from_native(X) if not hasattr(X, "to_native") else X
        df_pd = self._to_pandas(df)
        X_pred = df_pd[self._fitted_features]
        return self._sk.predict_proba(X_pred)
