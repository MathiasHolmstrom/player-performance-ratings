from typing import Any

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT
from sklearn.base import BaseEstimator, TransformerMixin, is_regressor



class NetOverPredictedTransformer(TransformerMixin):
    def __init__(
        self,
        estimator: BaseEstimator | Any,
        features: list[str],
        target_name: str,
        net_over_predicted_col: str,
        pred_column: str | None = None,
    ):
        self.features_out = [net_over_predicted_col, pred_column] if pred_column else [net_over_predicted_col]
        self.features_out = features
        self.target_name = target_name
        self.estimator = estimator
        self.pred_column = pred_column or '__pred'
        self.net_over_predicted_col = net_over_predicted_col
        assert is_regressor(estimator)


    @nw.narwhalify
    def fit(
        self,
        X: IntoFrameT,
        y,
    )  :
        self.estimator.fit(X, y)
        return self


    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        ori_cols = X.columns
        predictions = self.estimator.predict(X)
        X = X.with_columns(
            nw.new_series(name=self.pred_column, values=predictions, backend=nw.get_native_namespace(X))
        )
        X = X.with_columns(
            (nw.col(self.target_name) - nw.col(self.pred_column)).alias(self.net_over_predicted_col)
        )
        return X.select(ori_cols + self.features_out)

