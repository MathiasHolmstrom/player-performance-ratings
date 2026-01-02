import copy
from datetime import datetime
from typing import Optional

import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge.scorer import BaseScorer
from spforge.cross_validator._base import CrossValidator


class MatchKFoldCrossValidator(CrossValidator):
    """
    Cross-validation by splitting on match order (derived from match_id transitions),
    training on past matches and validating on future matches.

    - Always keeps all original columns.
    - Adds exactly one prediction column:
        * regression / label-predict: prediction values
        * classifier with predict_proba:
            - binary: p(class 1)
            - multiclass: list[float] per row with probs for each class in estimator.classes_
    """

    def __init__(
        self,
        match_id_column_name: str,
        date_column_name: str,
        estimator,
        prediction_column_name: str,
        scorer: Optional[BaseScorer] = None,
        min_validation_date: Optional[str] = None,
        n_splits: int = 3,
    ):
        super().__init__(
            scorer=scorer,
            min_validation_date=min_validation_date,
            estimator=estimator,
        )
        self.match_id_column_name = match_id_column_name
        self.date_column_name = date_column_name
        self.n_splits = n_splits
        self.min_validation_date = min_validation_date
        self.prediction_column_name = prediction_column_name

    @staticmethod
    def _to_native_pandas(df: IntoFrameT) -> "pd.DataFrame":
        import pandas as pd
        import polars as pl

        native = df.to_native() if hasattr(df, "to_native") else df
        if isinstance(native, pd.DataFrame):
            return native
        if isinstance(native, pl.DataFrame):
            return native.to_pandas()
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        return pd.DataFrame(native)

    @staticmethod
    def _is_numpy_array(x) -> bool:
        import numpy as np

        return isinstance(x, np.ndarray)

    def _fit_estimator(self, est, train_df: IntoFrameT):
        X_pd = self._to_native_pandas(train_df.select(self.features))
        y = train_df[self.target].to_list()
        est.fit(X_pd, y)

    def _predict_smart(self, est, df: IntoFrameT) -> IntoFrameT:
        import numpy as np

        X_pd = self._to_native_pandas(df.select(self.features))
        ns = nw.get_native_namespace(df)

        proba = None
        if hasattr(est, "predict_proba"):
            try:
                proba = est.predict_proba(X_pd)
            except AttributeError:
                proba = None

        if proba is not None:
            if not isinstance(proba, np.ndarray):
                proba = np.asarray(proba)

            if proba.ndim != 2:
                raise ValueError(f"predict_proba must return 2D array, got shape={proba.shape}")

            n_classes = proba.shape[1]

            if n_classes == 2:
                values = proba[:, 1].tolist()  # binary: P(class idx 1)
            else:
                values = [row.tolist() for row in proba]  # multiclass: store full vector per row

            return df.with_columns(
                nw.new_series(
                    name=self.prediction_column_name,
                    values=values,
                    backend=ns,
                )
            )

        preds = est.predict(X_pd)
        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        return df.with_columns(
            nw.new_series(
                name=self.prediction_column_name,
                values=preds.tolist(),
                backend=ns,
            )
        )

    @nw.narwhalify
    def generate_validation_df(
        self,
        df: IntoFrameT,
        return_features: bool = True,
        add_train_prediction: bool = False,
    ) -> IntoFrameT:
        # keep all original columns; just add prediction + validation flag
        if "__cv_row_index" not in df.columns:
            df = df.with_row_index("__cv_row_index")

        if self.prediction_column_name in df.columns:
            df = df.drop(self.prediction_column_name)

        df = df.sort([self.date_column_name, self.match_id_column_name])

        if self.validation_column_name in df.columns:
            df = df.drop(self.validation_column_name)

        if not self.min_validation_date:
            unique_dates = df[self.date_column_name].unique(maintain_order=True)
            median_number = len(unique_dates) // 2
            self.min_validation_date = unique_dates[median_number]

        df = df.with_columns(
            (
                nw.col(self.match_id_column_name)
                != nw.col(self.match_id_column_name).shift(1)
            )
            .cum_sum()
            .fill_null(0)
            .alias("__cv_match_number")
        )
        if df["__cv_match_number"].min() == 0:
            df = df.with_columns(nw.col("__cv_match_number") + 1)

        if isinstance(self.min_validation_date, str) and df.schema.get(
            self.date_column_name
        ) in (nw.Date, nw.Datetime):
            min_validation_date = datetime.strptime(self.min_validation_date, "%Y-%m-%d")
        else:
            min_validation_date = self.min_validation_date

        min_validation_match_number = (
            df.filter(nw.col(self.date_column_name) >= nw.lit(min_validation_date))
            .select(nw.col("__cv_match_number").min())
            .head(1)
            .item()
        )

        max_match_number = df.select(nw.col("__cv_match_number").max()).row(0)[0]
        train_cut_off_match_number = min_validation_match_number
        step_matches = (max_match_number - min_validation_match_number) / self.n_splits

        train_df = df.filter(nw.col("__cv_match_number") < train_cut_off_match_number)
        if len(train_df) == 0:
            raise ValueError(
                f"train_df is empty. train_cut_off_match_number: {train_cut_off_match_number}. "
                f"Select a lower validation_match value."
            )

        validation_df = df.filter(
            (nw.col("__cv_match_number") >= train_cut_off_match_number)
            & (nw.col("__cv_match_number") <= train_cut_off_match_number + step_matches)
        )

        validation_dfs = []

        for idx in range(self.n_splits):
            est = copy.deepcopy(self.estimator)
            self._fit_estimator(est, train_df)

            if idx == 0 and add_train_prediction:
                train_pred = self._predict_smart(est, train_df.drop("__cv_match_number"))
                train_pred = train_pred.with_columns(nw.lit(0).alias(self.validation_column_name))
                validation_dfs.append(train_pred)

            val_pred = self._predict_smart(est, validation_df.drop("__cv_match_number"))
            val_pred = val_pred.with_columns(nw.lit(1).alias(self.validation_column_name))
            validation_dfs.append(val_pred)

            train_cut_off_match_number += step_matches
            train_df = df.filter(nw.col("__cv_match_number") < train_cut_off_match_number)

            if idx == self.n_splits - 2:
                validation_df = df.filter(nw.col("__cv_match_number") >= train_cut_off_match_number)
            else:
                validation_df = df.filter(
                    (nw.col("__cv_match_number") >= train_cut_off_match_number)
                    & (nw.col("__cv_match_number") < train_cut_off_match_number + step_matches)
                )

        out = nw.concat(validation_dfs)

        # de-dupe (train/val overlap from add_train_prediction) and restore original order
        out = out.unique(subset=["__cv_row_index"], keep="first").sort("__cv_row_index")

        if "__cv_row_index" in out.columns:
            out = out.drop("__cv_row_index")

        # "return_features" kept for API compatibility; now always returns full df anyway
        return out
