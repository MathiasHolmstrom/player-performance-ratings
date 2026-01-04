from itertools import chain

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge import ColumnNames
from spforge.feature_generator._base import LagGenerator
from spforge.ratings._base import RatingGenerator


class FeatureGeneratorPipeline:
    """
    Pipeline of rating_generators, lag_generators and transformers to be applied to a dataframe
    For historical data use fit_transform
    For future data use transform.
    """

    def __init__(self, column_names: ColumnNames, feature_generators: list[LagGenerator | RatingGenerator]):
        features_out = list(chain.from_iterable(t.features_out for t in feature_generators))
        features = list(chain.from_iterable(t.features for t in feature_generators))
        self.features_out = features_out
        self.features = features
        self.feature_generators = feature_generators
        self.column_names = column_names

    @nw.narwhalify
    def fit_transform(self, df: IntoFrameT) -> IntoFrameT:
        """
        Fit and transform the pipeline on historical data
        :param df: Either polars or Pandas dataframe
        """

        expected_feats_added = []
        dup_feats = []
        feats_not_added = []

        for transformer in self.feature_generators:
            pre_row_count = len(df)
            df = nw.from_native(transformer.fit_transform(df, column_names=self.column_names))
            assert len(df) == pre_row_count
            for f in transformer.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(transformer.features_out)

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
            df = nw.from_native(transformer.transform(df))
            assert len(df) == pre_row_count
            for f in transformer.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(transformer.features_out)

        return df

    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        expected_feats_added = []
        dup_feats = []
        feats_not_added = []

        for transformer in self.feature_generators:
            pre_row_count = len(df)
            if hasattr(transformer, "future_transform") and callable(transformer.future_transform):
                df = nw.from_native(transformer.future_transform(df))
            else:
                df = nw.from_native(transformer.transform(df))
            assert len(df) == pre_row_count
            for f in transformer.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(transformer.features_out)

        return df
