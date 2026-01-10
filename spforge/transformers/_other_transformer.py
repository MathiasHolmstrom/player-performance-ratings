from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoFrameT
from sklearn.base import BaseEstimator, TransformerMixin


class GroupByReducer(BaseEstimator, TransformerMixin):
    def __init__(self, granularity: list[str]):
        self.granularity = granularity

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: Any = None):
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        if not self.granularity:
            return X

        df = X
        keys = [c for c in self.granularity if c in df.columns]
        if not keys:
            raise ValueError("Could not find granularity columns in dataframe %s", self.granularity)

        non_keys = [c for c in df.columns if c not in keys]
        num_cols = [c for c in non_keys if pd.api.types.is_numeric_dtype(df[c])]
        other_cols = [c for c in non_keys if c not in num_cols]

        aggs: list[nw.Expr] = []

        for c in num_cols:
            aggs.append(nw.col(c).mean().alias(c))

        for c in other_cols:
            aggs.append(nw.col(c).first().alias(c))

        out = df.group_by(keys).agg(aggs)
        return out

    @nw.narwhalify
    def reduce_y(
        self,
        X: IntoFrameT,
        y: Any,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray | Any, np.ndarray | None]:
        if not self.granularity:
            return y, sample_weight

        keys = [c for c in self.granularity if c in X.columns]

        if not keys:
            return y, sample_weight

        df = X.with_columns(nw.new_series(values=y, name="__y", backend=nw.get_native_namespace(X)))
        if sample_weight is not None:
            df = df.with_columns(nw.lit(sample_weight).alias("__sw"))

        y_is_numeric = df.select(nw.col("__y")).schema["__y"].is_numeric()

        if y_is_numeric:
            agg_exprs = [nw.col("__y").mean().alias("__y")]
        else:
            agg_exprs = [nw.col("__y").first().alias("__y")]

        if sample_weight is not None:
            agg_exprs.append(nw.col("__sw").sum().alias("__sw"))

        out = df.group_by(keys).agg(agg_exprs)

        y_out = out["__y"].to_numpy()

        if sample_weight is None:
            return y_out, None

        sw_out = out["__sw"].to_numpy()
        return y_out, sw_out


class ConvertDataFrameToCategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Converts a specified list of columns to categorical dtype
    """

    @nw.narwhalify
    def fit(self, df: IntoFrameT, y: Any):
        self._feature_names_in = df.columns
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        return X.with_columns(
            nw.col(feature).cast(nw.Categorical).alias(feature) for feature in X.columns
        )

    def set_output(self, *, transform=None):
        pass

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)

        return np.array(list(input_features), dtype=object)
