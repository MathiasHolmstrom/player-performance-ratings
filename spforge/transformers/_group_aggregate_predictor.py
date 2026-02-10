from __future__ import annotations

import re

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge.transformers._base import PredictorTransformer


class GroupAggregatePredictorTransformer(PredictorTransformer):
    """Create group aggregate features from a row-level prediction column.

    This transformer is intended for two-stage modeling workflows:
    1) upstream model creates row-level predictions (`pred_col`)
    2) this transformer creates group-level aggregate/share features
    3) downstream model consumes both row and aggregate features

    Base outputs (always):
    - ``<prefix>_pred_total``: sum of ``pred_col`` within each group
    - ``<prefix>_pred_share_group``: row prediction share of group total

    General segment mode (recommended):
    - Set ``segment_col`` to create segment-level totals.
    - For each segment value ``v`` in ``segment_values`` (or learned during ``fit``):
      - ``<prefix>_segment_<token(v)>_total``
    - Also creates:
      - ``<prefix>_pred_share_segment``: row prediction share of its own segment total

    Legacy role mode (backward compatible):
    - Set ``role_col`` and ``role_true_label``.
    - Outputs:
      - ``<prefix>_role_total``
      - ``<prefix>_non_role_total``
      - ``<prefix>_pred_share_role``

    Notes:
    - ``segment_col`` and ``role_col`` are mutually exclusive.
    - Column names are deterministic. If ``segment_values`` is omitted, values are
      learned during ``fit`` and reused in ``transform``.
    """

    def __init__(
        self,
        pred_col: str,
        group_cols: list[str],
        segment_col: str | None = None,
        segment_values: list[bool | int | str] | None = None,
        role_col: str | None = None,
        role_true_label: bool | int | str = True,
        prefix: str = "agg",
    ):
        self.pred_col = pred_col
        self.group_cols = group_cols
        self.segment_col = segment_col
        self.segment_values = segment_values
        self.role_col = role_col
        self.role_true_label = role_true_label
        self.prefix = prefix

        if not self.pred_col:
            raise ValueError("pred_col must be non-empty")
        if not self.group_cols:
            raise ValueError("group_cols must be non-empty")
        if not self.prefix:
            raise ValueError("prefix must be non-empty")
        if self.segment_col is not None and self.role_col is not None:
            raise ValueError("segment_col and role_col are mutually exclusive")
        if self.segment_values is not None and self.segment_col is None:
            raise ValueError("segment_values requires segment_col")

        self._segment_values_: list[bool | int | str] | None = None
        self._segment_tokens_: dict[bool | int | str, str] = {}
        self._is_fitted = False

    def _tokenize_segment_value(self, value: bool | int | str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = text.strip("_")
        return text or "empty"

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y=None):
        required = [self.pred_col, *self.group_cols]
        if self.segment_col is not None:
            required.append(self.segment_col)
        if self.role_col is not None:
            required.append(self.role_col)

        missing = [c for c in required if c not in set(X.columns)]
        if missing:
            raise ValueError(f"Missing required columns for fit: {sorted(set(missing))}")

        if self.segment_col is not None:
            if self.segment_values is not None:
                self._segment_values_ = list(self.segment_values)
            else:
                segment_series = X.select(self.segment_col).to_pandas()[self.segment_col]
                inferred_values = [v for v in segment_series.dropna().unique().tolist()]
                self._segment_values_ = inferred_values

            token_to_value: dict[str, bool | int | str] = {}
            self._segment_tokens_ = {}
            for value in self._segment_values_:
                token = self._tokenize_segment_value(value)
                if token in token_to_value and token_to_value[token] != value:
                    raise ValueError(
                        "segment_values produce colliding output names after normalization: "
                        f"{token_to_value[token]!r} and {value!r} -> {token!r}"
                    )
                token_to_value[token] = value
                self._segment_tokens_[value] = token

        self._is_fitted = True
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        native = X.to_native()
        native_index = getattr(native, "index", None)

        required = [self.pred_col, *self.group_cols]
        if self.segment_col is not None:
            required.append(self.segment_col)
        if self.role_col is not None:
            required.append(self.role_col)
        missing = [c for c in required if c not in set(X.columns)]
        if missing:
            raise ValueError(f"Missing required columns for transform: {sorted(set(missing))}")

        df = X.with_columns(
            nw.col(self.pred_col).cast(nw.Float64).alias("__pred__"),
            nw.col(self.pred_col).cast(nw.Float64).sum().over(self.group_cols).alias("__total__"),
        )

        exprs = [
            nw.col("__total__").alias(f"{self.prefix}_pred_total"),
            nw.when(nw.col("__total__") != 0.0)
            .then(nw.col("__pred__") / nw.col("__total__"))
            .otherwise(0.0)
            .alias(f"{self.prefix}_pred_share_group"),
        ]

        if self.segment_col is not None:
            segment_total_cols: list[tuple[bool | int | str, str]] = []
            for value in self._segment_values_ or []:
                token = self._segment_tokens_[value]
                masked_col = f"__seg_masked_{token}__"
                total_col = f"__seg_total_{token}__"
                df = df.with_columns(
                    nw.when(nw.col(self.segment_col) == value)
                    .then(nw.col("__pred__"))
                    .otherwise(0.0)
                    .alias(masked_col)
                ).with_columns(nw.col(masked_col).sum().over(self.group_cols).alias(total_col))
                exprs.append(nw.col(total_col).alias(f"{self.prefix}_segment_{token}_total"))
                segment_total_cols.append((value, total_col))

            if segment_total_cols:
                denominator_expr = None
                for value, total_col in segment_total_cols:
                    piece = (
                        nw.when(nw.col(self.segment_col) == value)
                        .then(nw.col(total_col))
                        .otherwise(0.0)
                    )
                    denominator_expr = piece if denominator_expr is None else denominator_expr + piece
            else:
                denominator_expr = nw.lit(0.0)

            exprs.append(
                nw.when(denominator_expr != 0.0)
                .then(nw.col("__pred__") / denominator_expr)
                .otherwise(0.0)
                .alias(f"{self.prefix}_pred_share_segment")
            )

        if self.role_col is not None:
            df = df.with_columns(
                nw.when(nw.col(self.role_col) == self.role_true_label)
                .then(nw.col("__pred__"))
                .otherwise(0.0)
                .alias("__role_pred__"),
                nw.when(nw.col(self.role_col) != self.role_true_label)
                .then(nw.col("__pred__"))
                .otherwise(0.0)
                .alias("__non_role_pred__"),
            ).with_columns(
                nw.col("__role_pred__").sum().over(self.group_cols).alias("__role_total__"),
                nw.col("__non_role_pred__").sum().over(self.group_cols).alias("__non_role_total__"),
            )

            role_denominator = (
                nw.when(nw.col(self.role_col) == self.role_true_label)
                .then(nw.col("__role_total__"))
                .otherwise(nw.col("__non_role_total__"))
            )
            exprs.extend(
                [
                    nw.col("__role_total__").alias(f"{self.prefix}_role_total"),
                    nw.col("__non_role_total__").alias(f"{self.prefix}_non_role_total"),
                    nw.when(role_denominator != 0.0)
                    .then(nw.col("__pred__") / role_denominator)
                    .otherwise(0.0)
                    .alias(f"{self.prefix}_pred_share_role"),
                ]
            )

        out = df.select(exprs)
        if native_index is not None and hasattr(out, "to_native"):
            out_native = out.to_native()
            if hasattr(out_native, "index"):
                out_native.index = native_index
            return out_native
        return out

    def get_feature_names_out(self, input_features=None) -> list[str]:
        out = [f"{self.prefix}_pred_total", f"{self.prefix}_pred_share_group"]
        if self.segment_col is not None:
            values = self._segment_values_ if self._segment_values_ is not None else self.segment_values
            for value in values or []:
                token = (
                    self._segment_tokens_[value]
                    if value in self._segment_tokens_
                    else self._tokenize_segment_value(value)
                )
                out.append(f"{self.prefix}_segment_{token}_total")
            out.append(f"{self.prefix}_pred_share_segment")
        if self.role_col is not None:
            out.extend(
                [
                    f"{self.prefix}_role_total",
                    f"{self.prefix}_non_role_total",
                    f"{self.prefix}_pred_share_role",
                ]
            )
        return out

    @property
    def context_features(self) -> list[str]:
        out = list(self.group_cols)
        if self.segment_col is not None:
            out.append(self.segment_col)
        if self.role_col is not None:
            out.append(self.role_col)
        return out
