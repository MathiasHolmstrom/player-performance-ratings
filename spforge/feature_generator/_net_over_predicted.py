from typing import Any

import narwhals as nw
from narwhals.stable.v1.typing import IntoFrameT
from sklearn.base import BaseEstimator

from spforge.base_feature_generator import FeatureGenerator
from spforge.data_structures import ColumnNames
from spforge.transformers import NetOverPredictedTransformer


class NetOverPredictedFeatureGenerator(FeatureGenerator):

    def __init__(
        self,
        estimator: BaseEstimator | Any,
        features: list[str],
        target_name: str,
        net_over_predicted_col: str,
        pred_column: str | None = None,
    ):
        _features_out = (
            [net_over_predicted_col, pred_column] if pred_column else [net_over_predicted_col]
        )
        super().__init__(features_out=_features_out)
        self.nop_transformer = NetOverPredictedTransformer(
            target_name=target_name,
            net_over_predicted_col=net_over_predicted_col,
            features=features,
            estimator=estimator,
        )

    @nw.narwhalify
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        transformed = nw.from_native(
            self.nop_transformer.fit_transform(df, df[self.nop_transformer.target_name])
        )
        return self._add_transformed(df, transformed)

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        transformed = self.nop_transformer.transform(df, df[self.nop_transformer.target_name])
        return self._add_transformed(df, transformed)

    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        transformed = self.nop_transformer.transform(df, df[self.nop_transformer.target_name])
        return self._add_transformed(df, transformed)

    def _add_transformed(self, df: IntoFrameT, transformed: IntoFrameT) -> IntoFrameT:
        for f in self.features_out:
            df = df.with_columns(
                nw.new_series(
                    name=f, values=transformed[f].to_list(), backend=nw.get_native_namespace(df)
                )
            )
        return df
