from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
from narwhals.typing import IntoFrameT
from sklearn.base import TransformerMixin

if TYPE_CHECKING:
    from spforge.pipeline import Pipeline

from spforge import ColumnNames
from spforge.feature_generator._base import LagGenerator



class RatioTeamPredictorTransformer(TransformerMixin):
    """
    Transformer that trains and uses the output of an estimator and divides it by the sum of the predictions for all the players within the team
    If team_total_prediction_column is passed in, it will also multiply the ratio by the team_total_prediction_column
    This is useful to provide a normalized point-estimate for a player for the given feature.

    """

    def __init__(
        self,
        features: list[str],
        estimator: "Pipeline",
        team_total_prediction_column: str | None = None,
        lag_transformer: list[LagGenerator] | None = None,
        prefix: str = "_ratio_team",
        target: str | None = None,
        pred_column: str = "prediction",
    ):
        """
        :param features: The features to track (for BaseTransformer), not used by estimator
        :param estimator: The estimator (sklearn-compatible) to use to add new prediction-columns to the dataset
        :param team_total_prediction_column: If passed, The column to multiply the ratio by.
        :param lag_transformer: Additional lag-generators (such as rolling-mean) can be performed after the ratio is calculated is passed
        :param prefix: The prefix to use for the new columns
        :param target: Target column name (inferred from estimator after fit if not provided)
        :param pred_column: Prediction column name (defaults to "prediction")
        """

        self.estimator = estimator
        self.team_total_prediction_column = team_total_prediction_column
        self.prefix = prefix
        self.lag_transformers = lag_transformer or []
        self._target_name = target
        self._pred_column = pred_column
        self.features_out = [target or "target" + prefix, pred_column]
        self.features = features

        target_name = target or "target"
        if self.team_total_prediction_column:
            self.features_out.append(target_name + prefix + "_team_total_multiplied")


    @nw.narwhalify
    def fit(self, df: IntoFrameT, column_names: ColumnNames | None) :
        self.column_names = column_names

        if self._target_name is None:
            all_cols = list(df.columns)
            if len(all_cols) < 2:
                raise ValueError(
                    "DataFrame must have at least 2 columns (feature + target) when target is not provided"
                )
            self._target_name = all_cols[-1]

        self.estimator.fit(X=df[self.features], y=df[self._target_name])

        if hasattr(self.estimator, "_target_name") and self.estimator._target_name:
            self._target_name = self.estimator._target_name

        return self


    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        return self._transform(df)


    def _transform(self, df: IntoFrameT) -> IntoFrameT:
        input_features = df.columns
        pred_result = self.estimator.predict(X=df[self.features])

        if isinstance(pred_result, np.ndarray):
            df_native = df.to_native()
            if isinstance(df_native, pd.DataFrame):
                df_result = df_native.copy()
                df_result[self._pred_column] = pred_result
                df = nw.from_native(df_result)
            else:
                df_result = df_native.with_columns(pl.Series(self._pred_column, pred_result))
                df = nw.from_native(df_result)
        else:
            df = pred_result

        if self._pred_column not in df.columns:
            raise ValueError(
                f"Prediction column '{self._pred_column}' not found in DataFrame. Available columns: {df.columns}"
            )

        df = df.with_columns(
            [
                nw.col(self._pred_column)
                .sum()
                .over([self.column_names.match_id, self.column_names.team_id])
                .alias(f"{self._pred_column}_sum")
            ]
        )

        df = df.with_columns(
            [
                (nw.col(self._pred_column) / nw.col(f"{self._pred_column}_sum")).alias(
                    self._features_out[0]
                )
            ]
        )

        if self.team_total_prediction_column:
            df = df.with_columns(
                [
                    (
                        nw.col(self._features_out[0]) * nw.col(self.team_total_prediction_column)
                    ).alias(f"{self._target_name}{self.prefix}_team_total_multiplied")
                ]
            )

        return df.select(list(set(input_features + self._features_out)))

