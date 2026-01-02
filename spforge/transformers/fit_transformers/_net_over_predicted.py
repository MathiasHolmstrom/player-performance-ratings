from typing import Optional


from spforge.predictor import BasePredictor

from spforge import ColumnNames

from spforge.transformers.base_transformer import (
    BaseTransformer,
)
import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT, IntoFrameT

from spforge.transformers.lag_transformers import BaseLagTransformer



class NetOverPredictedTransformer(BaseTransformer):
    def __init__(
        self,
        predictor: BasePredictor,
        features: list[str] = None,
        net_over_predicted_col: Optional[str] = None,
        are_estimator_features: bool = False,
    ):
        super().__init__(
            features=features,
            are_estimator_features=are_estimator_features,
            features_out=[],
        )

        self.predictor = predictor
        self.column_names = None

        self.net_over_predicted_col = (
            net_over_predicted_col
            if net_over_predicted_col is not None
            else f"net_over_predicted_{self.predictor.pred_column}"
        )

        if not self.net_over_predicted_col:
            raise ValueError("net_over_predicted_col must not be empty")

        self._features_out = [
            self.predictor.pred_column,
            self.net_over_predicted_col,
        ]

        if not hasattr(self, "_predictor_features_out"):
            self._predictor_features_out = []
        self._predictor_features_out.extend(self._features_out.copy())

        if self._are_estimator_features:
            self._predictor_features_out.append(self.predictor.pred_column)
            self.features_out.append(self.predictor.pred_column)

    @nw.narwhalify
    def fit_transform(
        self,
        df: IntoFrameT,
        column_names: Optional[ColumnNames] = None,
    ) -> IntoFrameT:
        ori_cols = df.columns
        self.column_names = column_names
        self.predictor.train(df, features=self.features)

        df = self._transform(df)

        return df.select(list(set(ori_cols + self.features_out)))

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        return self._transform(df)


    def _transform(self, df: IntoFrameT) -> IntoFrameT:
        ori_cols = df.columns
        df = nw.from_native(self.predictor.predict(df))

        df = df.with_columns(
            (nw.col(self.predictor.target) - nw.col(self.predictor.pred_column)).alias(
                self.net_over_predicted_col
            )
        )

        return df.select(
            list(set(ori_cols + self.features_out + self._predictor_features_out))
        )


