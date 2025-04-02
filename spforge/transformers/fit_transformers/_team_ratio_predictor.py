from typing import Optional

from spforge.predictor import BasePredictor

from spforge import ColumnNames

from spforge.transformers.base_transformer import (
    BaseTransformer,
)
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from spforge.transformers.lag_transformers import BaseLagTransformer


class RatioTeamPredictorTransformer(BaseTransformer):
    """
    Transformer that trains and uses the output of a predictor and divides it by the sum of the predictions for all the players within the team
    If team_total_prediction_column is passed in, it will also multiply the ratio by the team_total_prediction_column
    This is useful to provide a normalized point-estimate for a player for the given feature.

    """

    def __init__(
        self,
        features: list[str],
        predictor: BasePredictor,
        team_total_prediction_column: Optional[str] = None,
        lag_transformer: Optional[list[BaseLagTransformer]] = None,
        prefix: str = "_ratio_team",
    ):
        """
        :param features: The features to use for the predictor
        :param predictor: The predictor to use to add add new prediction-columns to the dataset
        :param team_total_prediction_column: If passed, The column to multiply the ratio by.
        :param lag_transformer: Additional lag-generators (such as rolling-mean) can be performed after the ratio is calculated is passed
        :param prefix: The prefix to use for the new columns
        """

        self.predictor = predictor
        self.team_total_prediction_column = team_total_prediction_column
        self.prefix = prefix
        self.lag_transformers = lag_transformer or []
        super().__init__(
            features=features,
            features_out=[self.predictor.target + prefix, self.predictor._pred_column],
        )

        if self.team_total_prediction_column:
            self._features_out.append(
                self.predictor.target + prefix + "_team_total_multiplied"
            )
        for lag_transformer in self.lag_transformers:
            self._features_out.extend(lag_transformer.features_out)
            self.predictor_features_out.extend(lag_transformer.predictor_features_out)

    @nw.narwhalify
    def fit_transform(
        self, df: FrameT, column_names: Optional[ColumnNames]
    ) -> IntoFrameT:
        ori_cols = df.columns
        self.column_names = column_names
        self.predictor.train(df=df, features=self.features)
        transformed_df = nw.from_native(self._transform(df))
        for lag_generator in self.lag_transformers:
            transformed_df = nw.from_native(
                lag_generator.transform_historical(
                    transformed_df, column_names=self.column_names
                )
            )

        return transformed_df.select(list(set(ori_cols + self.features_out)))

    @nw.narwhalify
    def transform(self, df: FrameT, cross_validate: bool = False) -> IntoFrameT:
        df = self._transform(df)
        for lag_transformer in self.lag_transformers:
            if cross_validate:
                df = lag_transformer.transform_historical(df)
            else:
                df = lag_transformer.transform_future(df)
        return df

    def _transform(self, df: FrameT) -> IntoFrameT:
        input_features = df.columns
        df = nw.from_native(self.predictor.predict(df=df))

        df = df.with_columns(
            [
                nw.col(self.predictor.pred_column)
                .sum()
                .over([self.column_names.match_id, self.column_names.team_id])
                .alias(f"{self.predictor.pred_column}_sum")
            ]
        )

        df = df.with_columns(
            [
                (
                    nw.col(self.predictor.pred_column)
                    / nw.col(f"{self.predictor.pred_column}_sum")
                ).alias(self._features_out[0])
            ]
        )

        if self.team_total_prediction_column:
            df = df.with_columns(
                [
                    (
                        nw.col(self._features_out[0])
                        * nw.col(self.team_total_prediction_column)
                    ).alias(
                        f"{self.predictor.target}{self.prefix}_team_total_multiplied"
                    )
                ]
            )

        return df.select(list(set(input_features + self.features_out)))

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def reset(self) -> "BaseTransformer":
        for lag_generator in self.lag_transformers:
            lag_generator.reset()
        return self
