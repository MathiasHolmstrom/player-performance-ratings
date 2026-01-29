import contextlib
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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from spforge.estimator import GroupByEstimator
from spforge.scorer import Filter, apply_filters
from spforge.transformers import PredictorTransformer
from spforge.transformers._other_transformer import ConvertDataFrameToCategoricalTransformer

_logger = logging.getLogger(__name__)

# --- Serializable Global Helpers ---


def _dedupe_preserve_order(cols):
    return list(dict.fromkeys(cols))


def _safe_feature_names_out(transformer, input_features):
    if hasattr(transformer, "get_feature_names_out"):
        try:
            out = transformer.get_feature_names_out(input_features)
        except TypeError:
            out = transformer.get_feature_names_out()
        with contextlib.suppress(Exception):
            out = out.tolist()
        out = list(out)
        if out:
            return out
    if hasattr(transformer, "features_out"):
        out = list(transformer.features_out)
        if out:
            return out
    return list(input_features)


def _to_pandas(X):
    return X.to_pandas() if hasattr(X, "to_pandas") else X


def _drop_columns_transformer(X, drop_cols):
    """Global function to replace lambda for dropping columns."""
    if not drop_cols:
        return X
    return X[[c for c in X.columns if c not in drop_cols]]


class _ColumnSelectorExcluding(BaseEstimator):
    """Class-based callable for ColumnTransformer selection to ensure picklability."""

    def __init__(self, drop_cols=frozenset()):
        self.drop_cols = drop_cols

    def __call__(self, X):
        return [c for c in X.columns if c not in self.drop_cols]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


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
        if self._feature_names_out is not None:
            return self._feature_names_out
        return np.array([], dtype=object)


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


CategoricalHandling = Literal["auto", "onehot", "ordinal", "native"]


def _is_lightgbm_estimator(obj: object) -> bool:
    mod = (getattr(type(obj), "__module__", "") or "").lower()
    name = type(obj).__name__
    if "lightgbm" in mod:
        return True
    return bool(name.startswith("LGBM"))


def _is_linear_estimator(obj: object) -> bool:
    mod = (getattr(type(obj), "__module__", "") or "").lower()
    name = type(obj).__name__
    if "logistic" in mod or "linear" in mod:
        return True
    return bool(name.startswith("Logistic"))


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
            est = obj.estimator
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


def lgbm_in_root(root) -> bool:
    return any(_is_lightgbm_estimator(obj) for obj in _walk_objects(root))


def _get_importance_estimator(estimator) -> tuple[Any, str] | None:
    """Recursively find innermost estimator with feature_importances_ or coef_."""
    if hasattr(estimator, "feature_importances_"):
        inner = _get_importance_estimator_inner(estimator)
        if inner is not None:
            return inner
        return (estimator, "feature_importances_")

    if hasattr(estimator, "coef_"):
        inner = _get_importance_estimator_inner(estimator)
        if inner is not None:
            return inner
        return (estimator, "coef_")

    return _get_importance_estimator_inner(estimator)


def _get_importance_estimator_inner(estimator) -> tuple[Any, str] | None:
    """Check wrapped estimators for importance attributes."""
    # Check estimator_ (sklearn fitted wrapper convention)
    if hasattr(estimator, "estimator_") and estimator.estimator_ is not None:
        result = _get_importance_estimator(estimator.estimator_)
        if result is not None:
            return result

    # Check _est (GroupByEstimator convention)
    if hasattr(estimator, "_est") and estimator._est is not None:
        result = _get_importance_estimator(estimator._est)
        if result is not None:
            return result

    return None


class AutoPipeline(BaseEstimator):
    def __init__(
        self,
        estimator: Any,
        estimator_features: list[str],
        predictor_transformers: list[PredictorTransformer] | None = None,
        granularity: list[str] | None = None,
        aggregation_weight: str | None = None,
        filters: list[Filter] | None = None,
        scale_features: bool = False,
        categorical_handling: CategoricalHandling = "auto",
        impute_missing_values: bool = False,
        drop_rows_where_target_is_nan: bool = True,
        min_target: int | float | None = None,
        max_target: int | float | None = None,
        categorical_features: list[str] | None = None,
        numeric_features: list[str] | None = None,
        remainder: str = "drop",
    ):
        self.estimator_features = estimator_features
        self.feature_names = estimator_features  # Internal compat
        self.granularity = granularity or []
        self.aggregation_weight = aggregation_weight
        self.predictor_transformers = predictor_transformers
        self.estimator = estimator
        self.filters = filters or []
        self.scale_features = scale_features
        self.categorical_handling = categorical_handling

        self.impute_missing_values = impute_missing_values
        self.drop_rows_where_target_is_nan = drop_rows_where_target_is_nan
        self.min_target = min_target
        self.max_target = max_target
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.remainder = remainder
        self._cat_feats = []
        self._filter_feature_names: list[str] = []

        # Auto-compute context features
        self.context_feature_names = self._compute_context_features()
        # _predictor_transformer_context is set by _compute_context_features()
        self.context_predictor_transformer_feature_names = self._predictor_transformer_context

        self.sklearn_pipeline: SkPipeline | None = None
        self._fitted_features: list[str] = []
        self._target_name: str | None = None
        self._resolved_categorical_handling: CategoricalHandling | None = None

    def _compute_context_features(self) -> list[str]:
        """Auto-compute context features from estimator and granularity.

        Note: Context from predictor_transformers is tracked separately in
        context_predictor_transformer_feature_names and is dropped before
        the final estimator. Filter columns are tracked separately and are
        dropped before the final estimator.
        """
        from spforge.transformers._base import PredictorTransformer

        context = []

        # Collect from predictor transformers (to be dropped before final estimator)
        self._predictor_transformer_context = []
        for transformer in self.predictor_transformers or []:
            if isinstance(transformer, PredictorTransformer):
                for feat in transformer.context_features:
                    if feat not in self._predictor_transformer_context:
                        self._predictor_transformer_context.append(feat)

        # Collect from final estimator (passed to final estimator)
        if hasattr(self.estimator, "context_features"):
            ctx = self.estimator.context_features
            if ctx:
                context.extend(ctx)
        else:
            # Legacy fallback for estimators without context_features property
            if hasattr(self.estimator, "date_column") and self.estimator.date_column:
                context.append(self.estimator.date_column)

            if (
                hasattr(self.estimator, "r_specific_granularity")
                and self.estimator.r_specific_granularity
            ):
                context.extend(self.estimator.r_specific_granularity)

            if hasattr(self.estimator, "column_names") and self.estimator.column_names:
                cn = self.estimator.column_names
                if hasattr(cn, "match_id") and cn.match_id:
                    context.append(cn.match_id)
                if hasattr(cn, "start_date") and cn.start_date:
                    context.append(cn.start_date)
                if hasattr(cn, "team_id") and cn.team_id:
                    context.append(cn.team_id)
                if hasattr(cn, "player_id") and cn.player_id:
                    context.append(cn.player_id)

        # Add granularity columns
        context.extend(self.granularity)

        # Add aggregation weight column
        if self.aggregation_weight:
            context.append(self.aggregation_weight)

        # Add filter columns
        self._filter_feature_names = []
        for f in self.filters:
            if f.column_name not in self._filter_feature_names:
                self._filter_feature_names.append(f.column_name)

        # Dedupe while preserving order, excluding estimator_features
        seen = set()
        deduped = []
        estimator_features_set = set(self.estimator_features)
        for c in context:
            if c not in seen and c not in estimator_features_set:
                seen.add(c)
                deduped.append(c)

        return deduped

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

    def _contains_linear_anywhere(self) -> bool:
        roots: list[object] = [self.estimator]
        for t in self.predictor_transformers or []:
            roots.append(t)

        for root in roots:
            for obj in _walk_objects(root):
                if _is_linear_estimator(obj):
                    return True
        return False

    def _resolve_categorical_handling(self) -> CategoricalHandling:
        if self.categorical_handling != "auto":
            return self.categorical_handling

        if self._contains_linear_anywhere():
            return "onehot"
        return "native" if self._contains_lightgbm_anywhere() else "ordinal"

    def _build_sklearn_pipeline(self, df: IntoFrameT) -> SkPipeline:
        num_feats, cat_feats = self._infer_feature_types(df)
        self._cat_feats = cat_feats

        do_groupby = len(self.granularity) > 0

        if do_groupby:
            num_feats = [c for c in num_feats if c not in self.granularity]
            cat_feats = [c for c in cat_feats if c not in self.granularity]

        self._resolved_categorical_handling = self._resolve_categorical_handling()

        num_steps = []
        cat_steps = []

        # Auto-enable imputation for linear estimators (they don't handle NaN)
        if self.impute_missing_values or self._contains_linear_anywhere():
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
                cat_steps.append(
                    (
                        "cat_dtype",
                        ConvertDataFrameToCategoricalTransformer(),
                    )
                )

        transformers = []

        if num_feats:
            num_pipe = SkPipeline(steps=num_steps) if num_steps else "passthrough"
            transformers.append(("num", num_pipe, num_feats))

        if cat_feats:
            cat_pipe = SkPipeline(steps=cat_steps) if cat_steps else "passthrough"
            transformers.append(("cat", cat_pipe, cat_feats))

        ctx_cols = _dedupe_preserve_order(
            (self.context_feature_names or [])
            + (self.context_predictor_transformer_feature_names or [])
        )

        if do_groupby:
            key_cols = list(self.granularity or [])
            key_set = set(key_cols)
            ctx_cols = [c for c in ctx_cols if c not in key_set]  # exclude overlap
            transformers.append(("key", "passthrough", key_cols))

        if ctx_cols:
            transformers.append(("ctx", "passthrough", ctx_cols))

        pre_raw = ColumnTransformer(
            transformers=transformers,
            remainder=self.remainder,
            verbose_feature_names_out=False,
        )
        pre_raw.set_output(transform="pandas")
        pre = PreprocessorToDataFrame(pre_raw)

        est = (
            GroupByEstimator(
                self.estimator,
                granularity=[f"{c}" for c in self.granularity],
                aggregation_weight=self.aggregation_weight,
            )
            if do_groupby
            else self.estimator
        )

        steps: list[tuple[str, Any]] = [("pre", pre)]

        prev_transformer_feats_out = []

        feature_names_set = set(self.feature_names)
        context_feature_names_set = set(self.context_feature_names)

        context_pred_feats = list(self.context_predictor_transformer_feature_names or [])
        drop_ctx = [
            c
            for c in context_pred_feats
            if (c not in feature_names_set) and (c not in context_feature_names_set)
        ]
        drop_ctx_set = set(drop_ctx)

        context_feature_names = list(set(self.context_feature_names + context_pred_feats))

        for idx, transformer in enumerate(self.predictor_transformers or []):
            input_cols = _dedupe_preserve_order(
                self.feature_names + context_feature_names + prev_transformer_feats_out
            )
            feats_out = list(_safe_feature_names_out(transformer, input_cols))
            if len(feats_out) != len(set(feats_out)):
                raise ValueError(
                    f"Duplicate names in feats_out for transformer {transformer}: {feats_out}"
                )

            # Compute columns to keep statically (all except feats_out)
            feats_out_set = set(feats_out)
            keep_cols = [c for c in input_cols if c not in feats_out_set]

            wrapped = _OnlyOutputColumns(transformer, feats_out)

            t_ct = ColumnTransformer(
                transformers=[
                    ("keep", "passthrough", keep_cols),
                    ("features", wrapped, input_cols),
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
            t_ct.set_output(transform="pandas")
            steps.append((f"t{idx}", t_ct))

            prev_transformer_feats_out.extend(feats_out)

        # Use FunctionTransformer with global function for serializability
        drop_filter_cols = set(self._filter_feature_names)
        drop_cols = drop_ctx_set | drop_filter_cols
        final = FunctionTransformer(
            _drop_columns_transformer, validate=False, kw_args={"drop_cols": drop_cols}
        )
        steps.append(("final", final))

        steps.append(("est", est))
        return SkPipeline(steps=steps)

    def _preprocess(self, df: nw.DataFrame, target: str) -> nw.DataFrame:
        if self.min_target is not None:
            df = df.with_columns(nw.col(target).clip(lower_bound=self.min_target))
        if self.max_target is not None:
            df = df.with_columns(nw.col(target).clip(upper_bound=self.max_target))

        df = nw.from_native(apply_filters(df=df, filters=self.filters))

        if self.drop_rows_where_target_is_nan:
            pre_row_count = len(df)
            df = df.filter(~nw.col(target).is_null())
            if pre_row_count != len(df):
                _logger.info("dropped %d rows with target nan", len(df) - pre_row_count)

        return df

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any, sample_weight: np.ndarray | None = None):
        self._target_name = getattr(y, "name", "target")
        self._fitted_features = list(
            set(
                self.feature_names
                + self.context_feature_names
                + self.context_predictor_transformer_feature_names
                + self._filter_feature_names
                + self.granularity
            )
        )

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

    @property
    def required_features(self) -> list[str]:
        """All features required by this pipeline for fit/predict.

        This includes:
        - estimator_features: features for the final estimator
        - Features from each predictor_transformer
        - Context features auto-detected from transformers and estimator
        - Granularity and filter columns

        Use this property when passing features to cross-validator:
            cross_validator = MatchKFoldCrossValidator(
                estimator=pipeline,
                features=pipeline.required_features,
                ...
            )

        Returns:
            Complete list of all columns needed by the pipeline.
        """

        all_features = list(self.estimator_features)

        # Add features from each predictor transformer
        for transformer in self.predictor_transformers or []:
            if hasattr(transformer, "features") and transformer.features:
                for feat in transformer.features:
                    if feat not in all_features:
                        all_features.append(feat)

        # Add context features
        for ctx in self.context_feature_names:
            if ctx not in all_features:
                all_features.append(ctx)

        # Add filter columns (needed for fit-time filtering)
        for col in self._filter_feature_names:
            if col not in all_features:
                all_features.append(col)

        return all_features

    def _get_estimator_feature_names(self) -> list[str]:
        """Get feature names as seen by the final estimator after all transformations."""
        pre_out = list(self.sklearn_pipeline.named_steps["pre"].get_feature_names_out())

        # Remove context columns dropped by "final" step
        final_step = self.sklearn_pipeline.named_steps["final"]
        drop_cols = final_step.kw_args.get("drop_cols", set()) if final_step.kw_args else set()
        features = [f for f in pre_out if f not in drop_cols]

        # Remove granularity columns (dropped by GroupByEstimator)
        granularity_set = set(self.granularity)
        features = [f for f in features if f not in granularity_set]

        # Remove context features (used by wrapper estimators, not inner model)
        context_set = set(self.context_feature_names)
        features = [f for f in features if f not in context_set]

        # Remove filter columns (used only for fit-time filtering)
        filter_set = set(self._filter_feature_names)
        features = [f for f in features if f not in filter_set]

        return features

    def _resolve_importance_feature_names(self, estimator, n_features: int) -> list[str]:
        names = None
        if hasattr(estimator, "feature_names_in_") and estimator.feature_names_in_ is not None:
            names = list(estimator.feature_names_in_)
        elif hasattr(estimator, "feature_name_") and estimator.feature_name_ is not None:
            names = list(estimator.feature_name_)
        elif hasattr(estimator, "feature_names_") and estimator.feature_names_ is not None:
            names = list(estimator.feature_names_)
        if names is None:
            names = self._get_estimator_feature_names()
        if len(names) != n_features:
            raise ValueError(
                f"Feature names length ({len(names)}) does not match importances length ({n_features})."
            )
        return names

    @property
    def feature_importances_(self) -> pd.DataFrame:
        """Get feature importances from the fitted estimator.

        Returns a DataFrame with columns ["feature", "importance"] sorted by
        absolute importance descending. Works with tree-based models
        (feature_importances_) and linear models (coef_).
        """
        if self.sklearn_pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        est = self.sklearn_pipeline.named_steps["est"]
        result = _get_importance_estimator(est)

        if result is None:
            raise RuntimeError(
                "Estimator does not support feature importances. "
                "Requires feature_importances_ or coef_ attribute."
            )

        inner_est, attr_name = result
        raw = getattr(inner_est, attr_name)

        if attr_name == "coef_":
            # Linear models: use absolute value of coefficients
            if raw.ndim == 2:
                # Multi-class: average absolute values across classes
                importances = np.abs(raw).mean(axis=0)
            else:
                importances = np.abs(raw)
        else:
            importances = raw

        feature_names = self._get_estimator_feature_names()

        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        df = df.sort_values("importance", ascending=False, key=abs).reset_index(drop=True)
        return df

    @property
    def feature_importance_names(self) -> dict[str, float]:
        """Map deepest estimator feature names to importances."""
        if self.sklearn_pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        est = self.sklearn_pipeline.named_steps["est"]
        result = _get_importance_estimator(est)

        if result is None:
            raise RuntimeError(
                "Estimator does not support feature importances. "
                "Requires feature_importances_ or coef_ attribute."
            )

        inner_est, attr_name = result
        raw = getattr(inner_est, attr_name)

        if attr_name == "coef_":
            if raw.ndim == 2:
                importances = np.abs(raw).mean(axis=0)
            else:
                importances = np.abs(raw)
        else:
            importances = raw

        importances = np.asarray(importances)
        feature_names = self._resolve_importance_feature_names(inner_est, len(importances))
        return dict(zip(feature_names, importances.tolist()))
