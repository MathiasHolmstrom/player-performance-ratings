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
        passthrough: bool = False,
        pred_column: str | None = None,
    ):
        self.features_out = [net_over_predicted_col, pred_column] if pred_column else [net_over_predicted_col]
        self.features_out = features
        self.target_name = target_name
        self.estimator = estimator
        self.pred_column = pred_column or '__pred'
        self.net_over_predicted_col = net_over_predicted_col
        self.passthrough = passthrough
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
        predictions = self.estimator.predict(X)
        X = X.with_columns(
            nw.new_series(name=self.pred_column, values=predictions, backend=nw.get_native_namespace(X))
        )
        X = X.with_columns(
            (nw.col(self.target_name) - nw.col(self.pred_column)).alias(self.net_over_predicted_col)
        )
        return X.select(self.get_feature_names_out())

    def set_output(self, *, transform=None):
        pass

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [self.net_over_predicted_col]
