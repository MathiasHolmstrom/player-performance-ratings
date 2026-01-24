from itertools import chain

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge.base_feature_generator import FeatureGenerator
from spforge.data_structures import ColumnNames


class FeatureGeneratorPipeline(FeatureGenerator):
    """
    Pipeline of rating_generators, lag_generators and transformers to be applied to a dataframe
    For historical data use fit_transform
    For future data use transform.
    """

    def __init__(
        self,
        feature_generators: list[FeatureGenerator],
        column_names: ColumnNames,
        auto_aggregate_to_team: bool = False,
    ):
        _features_out = list(chain.from_iterable(t.features_out for t in feature_generators))
        super().__init__(features_out=_features_out)
        self.column_names = column_names
        self.feature_generators = feature_generators
        self.auto_aggregate_to_team = bool(auto_aggregate_to_team)

    def _aggregate_to_team_level(
        self, df: nw.DataFrame, preferred_weight_col: str | None, fallback_weight_col: str | None
    ) -> nw.DataFrame:
        if not self.column_names:
            raise ValueError("column_names must be set for auto_aggregate_to_team")
        cn = self.column_names

        group_cols = [cn.match_id, cn.team_id, cn.start_date]
        if (
            cn.update_match_id
            and cn.update_match_id != cn.match_id
            and cn.update_match_id in df.columns
        ):
            group_cols.append(cn.update_match_id)

        missing_cols = [c for c in group_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for auto_aggregate_to_team: {missing_cols}")

        agg_cols = [c for c in self.features_out if c in df.columns]
        if not agg_cols:
            return df.select(group_cols).unique().sort(group_cols)

        fallback_feature_map: dict[str, str] = {}
        for generator in self.feature_generators:
            features = getattr(generator, "features", None)
            prefix = getattr(generator, "prefix", None)
            iterations = getattr(generator, "iterations", None)
            if not features or prefix is None or iterations is None:
                continue
            for feature in features:
                for iteration in iterations:
                    output_col = f"{prefix}_{feature}{iteration}"
                    if output_col in df.columns and feature in df.columns:
                        fallback_feature_map[output_col] = feature

        has_pref = preferred_weight_col and preferred_weight_col in df.columns
        has_fallback = fallback_weight_col and fallback_weight_col in df.columns

        weight_expr = None
        if has_pref and has_fallback:
            weight_expr = (
                nw.when(~nw.col(preferred_weight_col).is_null())
                .then(nw.col(preferred_weight_col))
                .otherwise(nw.col(fallback_weight_col))
            )
        elif has_pref:
            weight_expr = nw.col(preferred_weight_col)
        elif has_fallback:
            weight_expr = nw.col(fallback_weight_col)

        if weight_expr is None:
            aggs = []
            for col in agg_cols:
                fallback = fallback_feature_map.get(col)
                value_expr = (
                    nw.col(col).fill_null(nw.col(fallback))
                    if fallback and fallback in df.columns
                    else nw.col(col)
                )
                aggs.append(value_expr.mean().alias(col))
        else:
            aggs = []
            for col in agg_cols:
                fallback = fallback_feature_map.get(col)
                value_expr = (
                    nw.col(col).fill_null(nw.col(fallback))
                    if fallback and fallback in df.columns
                    else nw.col(col)
                )
                denom = weight_expr.sum()
                aggs.append(
                    nw.when((~denom.is_null()) & (denom != 0))
                    .then((value_expr * weight_expr).sum() / denom)
                    .otherwise(value_expr.mean())
                    .alias(col)
                )
        return df.group_by(group_cols).agg(aggs).sort(group_cols)

    @nw.narwhalify
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        """
        Fit and transform the pipeline on historical data
        :param df: Either polars or Pandas dataframe
        """
        column_names = column_names or self.column_names
        self.column_names = column_names

        expected_feats_added = []
        dup_feats = []
        feats_not_added = []

        for transformer in self.feature_generators:
            pre_row_count = len(df)
            native_df = df.to_native()
            df = nw.from_native(transformer.fit_transform(native_df, column_names=column_names))
            assert len(df) == pre_row_count
            for f in transformer.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(transformer.features_out)

        if self.auto_aggregate_to_team:
            return self._aggregate_to_team_level(
                df, column_names.participation_weight, column_names.projected_participation_weight
            )

        return df

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        """
        Transform the pipeline on future data
        :param df: Either polars or Pandas dataframe
        """
        expected_feats_added = []
        dup_feats = []
        feats_not_added = []

        for transformer in self.feature_generators:
            pre_row_count = len(df)
            native_df = df.to_native()
            df = nw.from_native(transformer.transform(native_df))
            assert len(df) == pre_row_count
            for f in transformer.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(transformer.features_out)

        if self.auto_aggregate_to_team:
            return self._aggregate_to_team_level(
                df,
                self.column_names.projected_participation_weight,
                self.column_names.participation_weight,
            )

        return df

    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        expected_feats_added = []
        dup_feats = []
        feats_not_added = []

        for transformer in self.feature_generators:
            pre_row_count = len(df)
            if hasattr(transformer, "future_transform") and callable(transformer.future_transform):
                native_df = df.to_native()
                df = nw.from_native(transformer.future_transform(native_df))
            else:
                native_df = df.to_native()
                df = nw.from_native(transformer.transform(native_df))
            assert len(df) == pre_row_count
            for f in transformer.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(transformer.features_out)

        if self.auto_aggregate_to_team:
            return self._aggregate_to_team_level(
                df,
                self.column_names.projected_participation_weight,
                self.column_names.participation_weight,
            )

        return df
