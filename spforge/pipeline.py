from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoFrameT
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spforge.estimator.sklearn_estimator import GroupByEstimator
from spforge.scorer import Filter, apply_filters


class PreprocessorToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor: Any):
        self.preprocessor = preprocessor
        self._feature_names_out: np.ndarray | None = None

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any = None):
        y = y.to_numpy() if not isinstance(y, np.ndarray) else y
        self.preprocessor.fit(X.to_native(), y)
        if hasattr(self.preprocessor, "get_feature_names_out"):
            self._feature_names_out = self.preprocessor.get_feature_names_out()
        else:
            self._feature_names_out = None
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        out = self.preprocessor.transform(X.to_native())
        return nw.from_native(
            out,
        )

    def get_feature_names_out(self, input_features: Any = None):
        if hasattr(self.preprocessor, "get_feature_names_out"):
            return self.preprocessor.get_feature_names_out(input_features)
        if self._feature_names_out is None:
            raise AttributeError("No feature names available.")
        return self._feature_names_out

class Pipeline(BaseEstimator):
    def __init__(
        self,
        estimator: Any,
        feature_names: list[str],
        granularity: list[str] | None = None,
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
        self.feature_names = feature_names
        self.granularity = granularity
        self.estimator = estimator
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
        self._target_name: str | None = None


    def _infer_feature_types(self, df_pd: pd.DataFrame) -> tuple[list[str], list[str]]:
        feats = list(self.feature_names)
        if self.numeric_features is not None and self.categorical_features is not None:
            num = [c for c in self.numeric_features if c in feats]
            cat = [c for c in self.categorical_features if c in feats]
            return num, cat

        if self.numeric_features is not None:
            num_set = {c for c in self.numeric_features if c in feats}
            cat = [c for c in feats if c not in num_set]
            return [c for c in feats if c in num_set], cat

        if self.categorical_features is not None:
            cat_set = {c for c in self.categorical_features if c in feats}
            num = [c for c in feats if c not in cat_set]
            return num, [c for c in feats if c in cat_set]

        num = []
        cat = []
        for c in self.feature_names:
            s = df_pd[c]
            if pd.api.types.is_numeric_dtype(s):
                num.append(c)
            else:
                cat.append(c)
        return num, cat

    def _build_sklearn_pipeline(self, df_pd: pd.DataFrame) -> SkPipeline:
        num_feats, cat_feats = self._infer_feature_types(df_pd)

        gran = [c for c in (self.granularity or []) if c in self.feature_names]
        do_groupby = len(gran) > 0

        if do_groupby:
            num_feats = [c for c in num_feats if c not in gran]
            cat_feats = [c for c in cat_feats if c not in gran]

        num_steps = []
        cat_steps = []

        if self.impute_missing_values:
            if num_feats:
                num_steps.append(("impute", SimpleImputer()))
            if cat_feats:
                cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))

        if self.scale_features and num_feats:
            num_steps.append(("scale", StandardScaler()))

        if self.one_hot_encode_cat_features and cat_feats:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            cat_steps.append(("ohe", ohe))

        transformers = []
        if num_feats:
            num_pipe = SkPipeline(steps=num_steps) if num_steps else "passthrough"
            transformers.append(("num", num_pipe, num_feats))

        if cat_feats:
            cat_pipe = SkPipeline(steps=cat_steps) if cat_steps else "passthrough"
            transformers.append(("cat", cat_pipe, cat_feats))

        if do_groupby:
            transformers.append(("key", "passthrough", gran))

        pre_raw = ColumnTransformer(transformers=transformers, remainder=self.remainder)
        pre_raw.set_output(transform='polars')
        pre = PreprocessorToDataFrame(pre_raw)

        est = (
            GroupByEstimator(self.estimator, granularity=[f"key__{c}" for c in gran])
            if do_groupby
            else self.estimator
        )

        return SkPipeline(steps=[("pre", pre), ("est", est)])

    def _preprocess(self, df: nw.DataFrame, target: str) -> nw.DataFrame:
        if self.min_target is not None:
            df = df.with_columns(nw.col(target).clip(lower_bound=self.min_target))
        if self.max_target is not None:
            df = df.with_columns(nw.col(target).clip(upper_bound=self.max_target))

        df = nw.from_native(apply_filters(df=df, filters=self.filters))

        if self.drop_rows_where_target_is_nan:
            df = df.filter(~nw.col(target).is_null())

        return df

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any, sample_weight: np.ndarray | None = None):
        self._target_name = getattr(y, "name", "target")
        self._fitted_features = list(self.feature_names)

        y_values = y.to_list() if hasattr(y, "to_list") else y
        df = X.with_columns(
            nw.new_series(
                name=self._target_name,
                values=y_values,
                backend=nw.get_native_namespace(X),
            )
        )


        df = self._preprocess(df, self._target_name)

        missing = [c for c in self._fitted_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        if self._target_name not in df.columns:
            raise ValueError(f"Missing target column: {self._target_name}")

        if len(df) == 0:
            raise ValueError("DataFrame is empty after preprocessing. Cannot fit estimator.")

        self._sk = self._build_sklearn_pipeline(df)

        X_fit = df.select(self._fitted_features)
        y_fit = df[self._target_name].to_numpy()

        if sample_weight is not None:
            self._sk.fit(X_fit, y_fit, est__sample_weight=sample_weight)
        else:
            self._sk.fit(X_fit, y_fit)

        if hasattr(self._sk[-1], "classes_"):
            self.classes_ = self._sk[-1].classes_

        return self

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> IntoFrameT | np.ndarray:
        if self._sk is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        return self._sk.predict(X[self._fitted_features].to_native())

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        if self._sk is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X_pred = X[self._fitted_features]
        return self._sk.predict_proba(X_pred)
