import copy

import narwhals.stable.v2 as nw
from narwhals.stable.v1.typing import IntoFrameT


class MatchKFoldCrossValidator:
    def __init__(
        self,
        match_id_column_name: str,
        date_column_name: str,
        target_column: str,
        estimator,
        prediction_column_name: str,
        n_splits: int = 3,
        min_validation_date: str | None = None,
        features: list[str] | None = None,
        binomial_probabilities_to_index1: bool = True,
    ):
        self.match_id_column_name = match_id_column_name
        self.date_column_name = date_column_name
        self.target_column = target_column
        self.estimator = estimator
        self.prediction_column_name = prediction_column_name
        self.n_splits = n_splits
        self.min_validation_date = min_validation_date
        self.features = features  # Optional: if None, will infer from DataFrame
        self.binomial_probabilities_to_index1 = binomial_probabilities_to_index1

    def _get_features(self, df):
        if self.features is not None:
            return self.features

        # Auto-detect if estimator is AutoPipeline with required_features
        if hasattr(self.estimator, "required_features"):
            return self.estimator.required_features

        # Fallback to current inference logic
        exclude_cols = {
            self.target_column,
            "__match_num",
            self.prediction_column_name,
            self.date_column_name,
            self.match_id_column_name,
        }

        return [c for c in df.columns if c not in exclude_cols]

    def _fit_estimator(self, est, train_df: IntoFrameT):
        features = self._get_features(train_df)
        X = train_df.select(features)
        y = train_df[self.target_column].to_list()
        est.fit(X, y)

    def _predict_smart(self, est, df: IntoFrameT) -> IntoFrameT:
        features = self._get_features(df)
        X = df.select(features)

        if hasattr(est, "predict_proba"):
            try:
                proba = est.predict_proba(X)

                if self.binomial_probabilities_to_index1:
                    if getattr(proba, "ndim", None) == 2:
                        n_cols = proba.shape[1]
                        values = proba[:, 1] if n_cols == 2 else proba.tolist()
                    else:
                        values = proba
                else:
                    values = proba.tolist() if getattr(proba, "ndim", None) == 2 else proba

            except AttributeError:
                values = est.predict(X)
        else:
            values = est.predict(X)

        out = df.with_columns(
            nw.new_series(
                name=self.prediction_column_name,
                values=values,
                backend=nw.get_native_namespace(df),
            )
        )

        if hasattr(est, "sklearn_pipeline"):
            Xt = X
            ok = True
            for name, step in est.sklearn_pipeline.steps:
                if name == "est":
                    break
                if not hasattr(step, "transform"):
                    ok = False
                    break
                Xt = step.transform(Xt)

            if ok and Xt is not None:
                feat_df = Xt if isinstance(Xt, nw.DataFrame) else nw.from_native(Xt)

                pred_col = self.prediction_column_name
                keep = [c for c in feat_df.columns if c != pred_col]

                if keep:
                    existing_cols = set(out.columns)
                    for c in keep:
                        if c in existing_cols:
                            continue
                        out = out.with_columns(
                            nw.new_series(
                                name=c,
                                values=feat_df[c].to_numpy(),
                                backend=nw.get_native_namespace(out),
                            )
                        )
                        existing_cols.add(c)

        return out

    @nw.narwhalify
    def generate_validation_df(
        self, df: IntoFrameT, add_training_predictions: bool = False
    ) -> IntoFrameT:
        df = df.sort([self.date_column_name, self.match_id_column_name])

        df = df.with_columns(
            (nw.col(self.match_id_column_name) != nw.col(self.match_id_column_name).shift(1))
            .cum_sum()
            .fill_null(0)
            .alias("__match_num")
        )

        if not self.min_validation_date:
            unique_dates = df[self.date_column_name].unique(maintain_order=True)
            median_number = len(unique_dates) // 2
            self.min_validation_date = unique_dates[median_number]

        max_m = df["__match_num"].max()
        min_m = df.filter(nw.col(self.date_column_name) >= self.min_validation_date)[
            "__match_num"
        ].min()

        step = (max_m - min_m) // self.n_splits
        if step <= 0:
            step = 1

        results = []
        curr_cut = min_m
        added_train_preds = False

        for i in range(self.n_splits):
            train_df = df.filter(nw.col("__match_num") < curr_cut)
            if len(train_df) == 0:
                raise ValueError("Traning data is empty")

            val_df = df.filter(
                (nw.col("__match_num") >= curr_cut) & (nw.col("__match_num") < curr_cut + step)
            )
            if len(val_df) == 0:
                raise ValueError("Val data is empty")

            if i == self.n_splits - 1:
                val_df = df.filter(nw.col("__match_num") >= curr_cut)

            est = copy.deepcopy(self.estimator)
            self._fit_estimator(est, train_df)

            if add_training_predictions and not added_train_preds:
                train_pred_df = self._predict_smart(est, train_df).with_columns(
                    nw.lit(0).alias("is_validation")
                )
                results.append(train_pred_df)
                added_train_preds = True

            pred_df = self._predict_smart(est, val_df).with_columns(
                nw.lit(1).alias("is_validation")
            )
            results.append(pred_df)

            curr_cut += step

        return nw.concat(results).drop("__match_num")
