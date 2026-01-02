from narwhals.typing import IntoFrameT, IntoFrameT

import narwhals.stable.v2 as nw
import numpy as np

from spforge.transformers.base_transformer import BaseTransformer


class ConvertDataFrameToCategoricalTransformer(BaseTransformer):
    """
    Converts a specified list of columns to categorical dtype
    """

    def __init__(self, features: list[str]):
        super().__init__(features=features, features_out=features)

    @nw.narwhalify
    def fit_transform(self, df: IntoFrameT, column_names=None) -> IntoFrameT:
        return self.transform(df)

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        df = df.with_columns(
            nw.col(feature).cast(nw.Categorical) for feature in self.features
        )
        return df


