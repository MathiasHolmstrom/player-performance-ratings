from typing import Optional

import narwhals as nw
from narwhals.dataframe import FrameT
from narwhals.typing import IntoFrameT

from ._base import BasePredictor
from ..predictor_transformer._simple_transformer import SimpleTransformer


class OperatorsPredictor(BasePredictor):

    def __init__(
        self, transformers: list[SimpleTransformer], pred_column: str, target: str
    ):

        self.transformers = transformers
        super().__init__(
            pred_column=pred_column,
            target=target,
            features=[],
        )

    @nw.narwhalify
    def train(self, df: FrameT, features: Optional[list[str]] = None) -> None:
        pass

    @nw.narwhalify
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:
        for transformer in self.transformers:
            df = transformer.transform(df)
        assert (
            self.pred_column in df.columns
        ), f"Pred column {self.pred_column} not found in transformed DataFrame"
        return df
