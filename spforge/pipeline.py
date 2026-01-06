import datetime
import logging
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoFrameT
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from spforge.estimator.sklearn_estimator import GroupByEstimator
from spforge.scorer import Filter, apply_filters
from spforge.transformers import ConvertDataFrameToCategoricalTransformer

_logger = logging.getLogger(__name__)

def _dedupe_preserve_order(cols):
    return list(dict.fromkeys(cols))


def _safe_feature_names_out(transformer, input_features):
    if hasattr(transformer, "get_feature_names_out"):
        try:
            out = transformer.get_feature_names_out(input_features)
        except TypeError:
            out = transformer.get_feature_names_out()
        try:
            out = out.tolist()
        except Exception:
            pass
        out = list(out)
        if out:
            return out
    if hasattr(transformer, "features_out"):
        out = list(transformer.features_out)
        if out:
            return out
    return list(input_features)

class _OnlyOutputColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, output_cols):
        self.transformer = transformer
        self.output_cols = output_cols

    def fit(self, X, y=None, **fit_params):
        if hasattr(self.transformer, "fit"):
            self.transformer.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        cols = list(self.output_cols) if self.output_cols is not None else []
        Z = self.transformer.transform(X)

        if hasattr(Z, "columns"):
            if cols:
                missing = [c for c in cols if c not in Z.columns]
                if missing:
                    raise ValueError(f"Transformer did not produce expected columns: {missing}")
                return Z[cols]
            return Z

        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if cols and Z.shape[1] != len(cols):
            raise ValueError(f"Transformer output has {Z.shape[1]} cols but expected {len(cols)}")
        idx = getattr(X, "index", None)
        return pd.DataFrame(Z, columns=(cols if cols else None), index=idx)

    def get_feature_names_out(self, input_features=None):
        if self.output_cols is None:
            return np.asarray(input_features if input_features is not None else [], dtype=object)
        return np.asarray(list(self.output_cols), dtype=object)

class PreprocessorToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor: Any):
        self.preprocessor = preprocessor
        self._feature_names_out: np.ndarray | None = None

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any = None):
        y = y.to_numpy() if not isinstance(y, np.ndarray) else y
        self.preprocessor.fit(X.to_pandas(), y)
        if hasattr(self.preprocessor, "get_feature_names_out"):
            self._feature_names_out = self.preprocessor.get_feature_names_out()
        else:
            self._feature_names_out = None
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        out = self.preprocessor.transform(X.to_pandas())
        return nw.from_native(out)

    def get_feature_names_out(self, input_features: Any = None):
        if hasattr(self.preprocessor, "get_feature_names_out"):
            return self.preprocessor.get_feature_names_out(input_features)
        if self._feature_names_out is None:
            raise AttributeError("No feature names available.")
        return self._feature_names_out


CategoricalHandling = Literal["auto", "onehot", "ordinal", "native"]


def _is_lightgbm_estimator(obj: object) -> bool:
    mod = (getattr(type(obj), "__module__", "") or "").lower()
    name = type(obj).__name__
    if "lightgbm" in mod:
        return True
    if name.startswith("LGBM"):
        return True
    return False


def _walk_objects(root: object):
    seen: set[int] = set()
    stack: list[object] = [root]
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        yield obj

        if hasattr(obj, "estimator"):
            est = getattr(obj, "estimator")
            if est is not None:
                stack.append(est)

        steps = getattr(obj, "steps", None)
        if isinstance(steps, list):
            for _, step in steps:
                stack.append(step)

        transformers = getattr(obj, "transformers", None)
        if isinstance(transformers, list):
            for t in transformers:
                if not isinstance(t, tuple) or len(t) < 2:
                    continue
                trans = t[1]
                if trans not in ("drop", "passthrough") and trans is not None:
                    stack.append(trans)

        if isinstance(obj, (list, tuple, set)):
            stack.extend(list(obj))

        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            stack.extend(d.values())


class Pipeline(BaseEstimator):
    def __init__(
        self,
        estimator: Any,
        feature_names: list[str],
        predictor_transformers: list[Any] | None = None,
        context_feature_names: list[str] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
        scale_features: bool = False,
        categorical_handling: CategoricalHandling = "auto",
        one_hot_encode_cat_features: bool = False,
        ordinal_encode_cat_features: bool = False,
        convert_cat_features_to_cat_dtype: bool = False,
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
        self.predictor_transformers = predictor_transformers
        self.estimator = estimator
        self.filters = filters or []
        self.scale_features = scale_features
        self.context_feature_names = context_feature_names or []
        self.categorical_handling = categorical_handling

        self.one_hot_encode_cat_features = one_hot_encode_cat_features
        self.ordinal_encode_cat_features = ordinal_encode_cat_features
        self.convert_cat_features_to_cat_dtype = convert_cat_features_to_cat_dtype

        self.impute_missing_values = impute_missing_values
        self.drop_rows_where_target_is_nan = drop_rows_where_target_is_nan
        self.min_target = min_target
        self.max_target = max_target
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.remainder = remainder

        assert len([c for c in self.context_feature_names if c in self.feature_names]) == 0

        if sum(
            [
                bool(self.one_hot_encode_cat_features),
                bool(self.ordinal_encode_cat_features),
                bool(self.convert_cat_features_to_cat_dtype),
            ]
        ) > 1:
            raise ValueError(
                "Only one of one_hot_encode_cat_features, ordinal_encode_cat_features, "
                "convert_cat_features_to_cat_dtype can be True"
            )

        self.sklearn_pipeline: SkPipeline | None = None
        self._fitted_features: list[str] = []
        self._target_name: str | None = None
        self._resolved_categorical_handling: CategoricalHandling | None = None

    def _infer_feature_types(self, df: IntoFrameT) -> tuple[list[str], list[str]]:
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
        df_pd = df.to_pandas()
        for c in self.feature_names:
            s = df_pd[c]
            if pd.api.types.is_numeric_dtype(s):
                num.append(c)
                continue
            if pd.api.types.is_datetime64_any_dtype(s):
                continue
            if pd.api.types.is_object_dtype(s):
                parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().mean() > 0.9:
                    continue
            cat.append(c)
        _logger.info("Infered cat features %s", cat)
        _logger.info("Infered num features %s", num)
        return num, cat

    def _contains_lightgbm_anywhere(self) -> bool:
        roots: list[object] = [self.estimator]
        for t in self.predictor_transformers or []:
            roots.append(t)

        for root in roots:
            for obj in _walk_objects(root):
                if _is_lightgbm_estimator(obj):
                    return True
        return False

    def _resolve_categorical_handling(self) -> CategoricalHandling:
        if self.one_hot_encode_cat_features:
            return "onehot"
        if self.ordinal_encode_cat_features:
            return "ordinal"
        if self.convert_cat_features_to_cat_dtype:
            return "native"

        if self.categorical_handling != "auto":
            return self.categorical_handling

        return "native" if self._contains_lightgbm_anywhere() else "ordinal"

    def _build_sklearn_pipeline(self, df: IntoFrameT) -> SkPipeline:
        num_feats, cat_feats = self._infer_feature_types(df)

        gran = [c for c in (self.granularity or []) if c in self.feature_names]
        do_groupby = len(gran) > 0

        if do_groupby:
            num_feats = [c for c in num_feats if c not in gran]
            cat_feats = [c for c in cat_feats if c not in gran]

        self._resolved_categorical_handling = self._resolve_categorical_handling()

        num_steps = []
        cat_steps = []

        if self.impute_missing_values:
            if num_feats:
                num_steps.append(("impute", SimpleImputer()))
            if cat_feats:
                cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))

        if self.scale_features and num_feats:
            num_steps.append(("scale", StandardScaler()))

        if cat_feats:
            if self._resolved_categorical_handling == "onehot":
                try:
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                except TypeError:
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
                cat_steps.append(("ohe", ohe))
            elif self._resolved_categorical_handling == "ordinal":
                cat_steps.append(
                    (
                        "ordinal",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    )
                )
            elif self._resolved_categorical_handling == "native":
                cat_steps.append(("cat_dtype", ConvertDataFrameToCategoricalTransformer()))

        transformers = []

        if num_feats:
            num_pipe = SkPipeline(steps=num_steps) if num_steps else "passthrough"
            transformers.append(("num", num_pipe, num_feats))

        if cat_feats:
            cat_pipe = SkPipeline(steps=cat_steps) if cat_steps else "passthrough"
            transformers.append(("cat", cat_pipe, cat_feats))

        if self.context_feature_names:
            transformers.append(("ctx", "passthrough", self.context_feature_names))

        if do_groupby:
            transformers.append(("key", "passthrough", gran))

        pre_raw = ColumnTransformer(
            transformers=transformers,
            remainder=self.remainder,
            verbose_feature_names_out=False,
        )
        pre_raw.set_output(transform="pandas")
        pre = PreprocessorToDataFrame(pre_raw)

        est = (
            GroupByEstimator(self.estimator, granularity=[f"{c}" for c in gran])
            if do_groupby
            else self.estimator
        )

        steps: list[tuple[str, Any]] = [("pre", pre)]

        prev_transformer_feats_out = []

        for idx, transformer in enumerate(self.predictor_transformers or []):
            input_cols = _dedupe_preserve_order(
                self.feature_names + self.context_feature_names + prev_transformer_feats_out
            )
            feats_out = _safe_feature_names_out(transformer, input_cols)
            def _keep_cols(X, drop=set(feats_out)):
                return [c for c in X.columns if c not in drop]

            wrapped = _OnlyOutputColumns(transformer, feats_out)

            t_ct = ColumnTransformer(
                transformers=[
                    ("keep", "passthrough", _keep_cols),
                    ("features", wrapped, input_cols),
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
            t_ct.set_output(transform="pandas")
            steps.append((f"t{idx}", t_ct))

            prev_transformer_feats_out.extend(feats_out)

        steps.append(("est", est))
        return SkPipeline(steps=steps)

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
        self._fitted_features = self.feature_names + self.context_feature_names

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

        self.sklearn_pipeline = self._build_sklearn_pipeline(df)

        X_fit = df.select(self._fitted_features)
        y_fit = df[self._target_name].to_numpy()

        if sample_weight is not None:
            self.sklearn_pipeline.fit(X_fit, y_fit, est__sample_weight=sample_weight)
        else:
            self.sklearn_pipeline.fit(X_fit, y_fit)

        if hasattr(self.sklearn_pipeline[-1], "classes_"):
            self.classes_ = self.sklearn_pipeline[-1].classes_

        return self

    @nw.narwhalify
    def predict(self, X: IntoFrameT) -> IntoFrameT | np.ndarray:
        if self.sklearn_pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        return self.sklearn_pipeline.predict(X[self._fitted_features])

    @nw.narwhalify
    def predict_proba(self, X: IntoFrameT) -> np.ndarray:
        if self.sklearn_pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X_pred = X[self._fitted_features]
        return self.sklearn_pipeline.predict_proba(X_pred)
