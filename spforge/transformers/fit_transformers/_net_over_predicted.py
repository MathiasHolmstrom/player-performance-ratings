from typing import Optional


from spforge.predictor import BasePredictor

from spforge import ColumnNames

from spforge.transformers.base_transformer import (
    BaseTransformer,
)
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from spforge.transformers.lag_transformers import BaseLagTransformer


class NetOverPredictedTransformer(BaseTransformer):

    def __init__(
        self,
        predictor: BasePredictor,
        features: list[str] = None,
        lag_transformers: Optional[list[BaseLagTransformer]] = None,
        prefix: str = "net_over_predicted_",
        are_estimator_features: bool = False,
    ):
        super().__init__(
            features=features,
            are_estimator_features=are_estimator_features,
            features_out=[],
        )
        self.prefix = prefix
        self.predictor = predictor
        self._features_out = [self.predictor.pred_column]
        self.lag_transformers = lag_transformers or []
        self.column_names = None
        new_feature_name = self.prefix + self.predictor.pred_column
        self._features_out.append(new_feature_name)
        for lag_generator in self.lag_transformers:
            if not lag_generator.features:
                lag_generator.features = [new_feature_name]
                for iteration in lag_generator.iterations:
                    lag_generator._features_out = [
                        f"{lag_generator.prefix}{iteration}_{new_feature_name}"
                    ]
                    self.features_out.extend(lag_generator._features_out.copy())
                    self._predictor_features_out.extend(
                        lag_generator._features_out.copy()
                    )

        if self._are_estimator_features:
            self._predictor_features_out.append(self.predictor.pred_column)
            self.features_out.append(self.predictor.pred_column)
        if self.prefix is "":
            raise ValueError("Prefix must not be empty")

    @nw.narwhalify
    def fit_transform(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        ori_cols = df.columns
        self.column_names = column_names
        self.predictor.train(df, features=self.features)
        df = self._transform(df)
        for lag_generator in self.lag_transformers:
            df = lag_generator.transform_historical(df, column_names=self.column_names)
        return df.select(list(set(ori_cols + self.features_out)))

    @nw.narwhalify
    def transform(self, df: FrameT, cross_validate: bool = False) -> IntoFrameT:
        if cross_validate:
            df = self._transform(df)

        if cross_validate:
            for lag_generator in self.lag_transformers:
                df = lag_generator.transform_historical(df)
        else:
            for lag_generator in self.lag_transformers:
                df = lag_generator.transform_future(df)

        return df

    def _transform(self, df: FrameT) -> FrameT:
        ori_cols = df.columns
        df = nw.from_native(self.predictor.predict(df))
        new_feature_name = self.prefix + self.predictor.pred_column
        df = df.with_columns(
            (nw.col(self.predictor.target) - nw.col(self.predictor.pred_column)).alias(
                new_feature_name
            )
        )

        return df.select(
            list(set(ori_cols + self.features_out + self._predictor_features_out))
        )

    def reset(self) -> "BaseTransformer":
        for lag_generator in self.lag_transformers:
            lag_generator.reset()
        return self
